"""M2 单 rank worker：通过 AdmissionScheduler 在两 rank Gloo 上执行场景。

由 ``run_m2.py`` 驱动，经 ``env://`` 初始化两 rank process group：:

    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 \
        python m2_worker.py --scenario fifo

每个 rank 向 stdout 打印单行 JSON：status、该 rank 的 ``launched_seq``
（per-group 发射序列，driver 据此对比两 rank 是否一致）与每个 collective
的 telemetry。与 M0 的区别是 collective 不再被直接调用，而是先提交
``CommIntent``，由 scheduler 校验 plan、执行 admission 后再调用原始
``dist.*`` 异步 collective。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    DirectLaunchExecutor,
    Plan,
    TaskKey,
)
from runtime_comm_scheduler.validate import ValidationError

# collective 使用的 tensor 大小（float32 元素数）
_N = 1024


def _key(ordinal: int) -> TaskKey:
    return TaskKey(0, 0, "dp", "dp", 0, 0, ordinal)


def _build_plan(scenario: str) -> Plan:
    ar = ("all_reduce", _N * 4)
    if scenario in ("fifo", "out_of_order_submit", "op_mismatch"):
        entries = ((_key(0), *ar), (_key(1), *ar))
    elif scenario == "fixed_reorder":
        entries = ((_key(1), *ar), (_key(0), *ar))
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return Plan(version=0, window_id=0, entries=tuple(entries))


def _make_all_reduce(key: TaskKey, rank: int, world: int):
    """构造 all_reduce intent 与验证函数（admission 后经 launch_fn 发射）。"""
    tensor = torch.ones(_N, dtype=torch.float32) * (rank + 1)

    def launch():
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

    def verify() -> bool:
        expected = world * (world + 1) / 2
        return bool(torch.all(tensor == expected).item())

    intent = CommIntent(
        key=key, op="all_reduce", tensor=tensor, process_group=None,
        num_bytes=_N * 4, launch_fn=launch,
    )
    return intent, verify


def _make_all_gather(key: TaskKey, rank: int, world: int):
    """构造 all_gather intent 与验证函数。"""
    tensor = torch.ones(_N, dtype=torch.float32) * (rank + 1)
    gathered = [torch.empty_like(tensor) for _ in range(world)]

    def launch():
        return dist.all_gather(gathered, tensor, async_op=True)

    def verify() -> bool:
        return all(bool(torch.all(g == (i + 1)).item())
                   for i, g in enumerate(gathered))

    intent = CommIntent(
        key=key, op="all_gather", tensor=tensor, process_group=None,
        num_bytes=_N * 4, launch_fn=launch, keepalive=tuple(gathered),
    )
    return intent, verify


def run(rank: int, world: int, scenario: str) -> dict:
    dist.init_process_group(backend="gloo")
    plan = _build_plan(scenario)
    sched = AdmissionScheduler(
        plan,
        local_group_ids=plan.group_ids(),
        executor=DirectLaunchExecutor(),
    )
    out: dict = {"rank": rank, "scenario": scenario, "status": "ok"}
    try:
        if scenario == "fifo":
            # 按 plan 顺序 [k0, k1] 提交。
            a, av = _make_all_reduce(_key(0), rank, world)
            b, bv = _make_all_reduce(_key(1), rank, world)
            wa = sched.submit(a)
            wb = sched.submit(b)
            assert wa.wait() and wb.wait()
            assert av() and bv()

        elif scenario == "fixed_reorder":
            # plan 是固定重排 [k1, k0]，按 plan 提交。
            b, bv = _make_all_reduce(_key(1), rank, world)
            a, av = _make_all_reduce(_key(0), rank, world)
            wb = sched.submit(b)
            wa = sched.submit(a)
            assert wb.wait() and wa.wait()
            assert bv() and av()

        elif scenario == "out_of_order_submit":
            # plan 顺序 [k0, k1]；故意先提交 k1：被 scheduler parked，
            # k0 提交后才按计划发射。这是 M2 与 M0 的关键差异。
            a, av = _make_all_reduce(_key(0), rank, world)
            b, bv = _make_all_reduce(_key(1), rank, world)
            wb = sched.submit(b)   # parked：队首 k0 尚未提交
            wa = sched.submit(a)   # 提交队首，drain 顺带发射 k1
            assert wb.wait() and wa.wait()
            assert bv() and av()

        elif scenario == "op_mismatch":
            # plan 对 k1 期望 all_reduce，但 intent 是 all_gather ->
            # fail-stop（ValidationError），两 rank 一致有界退出，
            # 不再像 M0 的 op_mismatch 那样挂起。
            a, av = _make_all_reduce(_key(0), rank, world)
            b, _bv = _make_all_gather(_key(1), rank, world)
            wa = sched.submit(a)
            try:
                sched.submit(b)
                out["status"] = "error"
                out["error"] = "expected ValidationError"
            except ValidationError as exc:
                out["status"] = "validation_error"
                out["error"] = str(exc)
            wa.wait()
            assert av()

        else:
            raise ValueError(f"unknown scenario: {scenario!r}")

        if scenario != "op_mismatch":
            sched.finish_window(timeout=10)
        out["launched_seq"] = {
            g: [k.as_list() for k in keys]
            for g, keys in sched.group_sequence_log().items()
        }
        out["timings"] = [
            {
                "key": t.key.as_list(),
                "ready_us": t.ready_record_ts,
                "admit_us": t.admit_ts,
                "submit_us": t.submit_ts,
                "complete_us": t.complete_ts,
                "actual_us": t.actual_duration_us,
            }
            for t in sched.timings()
        ]
    except Exception as exc:  # noqa: BLE001 - 记录任何异常后仍打印结果
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            sched.close()
        finally:
            dist.destroy_process_group()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    out = run(rank, world, args.scenario)
    print(json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
