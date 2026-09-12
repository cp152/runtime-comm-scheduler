"""M3 单 rank worker：两 rank NCCL 上的 CUDA event 语义 harness。

由 ``run_m3.py`` 驱动，NCCL backend。每个 rank 把场景跑在
``torch.profiler`` 下并导出 chrome trace，向 stdout 打单行 JSON（status、
``launched_seq``、每个 collective 的 ready/admit/submit/complete 时序与
completion-event 查询结果）。

场景覆盖 M3 的两个验收点：

- **stream dependency 正确**：``delayed_ready`` 中 producer 在 side stream
  上慢速生成 tensor 并记录 CUDA ready event，scheduler 在发射前让当前
  stream ``wait_event``，collective 排到 producer 之后（正确结果）；
  ``no_wait_ready`` 去掉该 event 作为对照，collective 与 producer 竞争
  （结果错误/不确定），证明依赖边是必需的。
- **stream-ordered wait**：``wait_stream_ordered`` 验证 ``Work.wait()`` 把
  NCCL completion dependency 接入 consumer stream，但不阻塞 CPU 等待 GPU
  物理完成。

其余场景（fifo/fixed_reorder/op_mismatch）与 M2 对齐，验证 NCCL 下
plan 校验与固定重排的安全性，错误场景 fail-stop 有界退出。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.profiler

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    Plan,
    TaskKey,
    TorchProcessGroupExecutor,
)
from runtime_comm_scheduler.validate import ValidationError

_N = 1024  # tensor 大小（float32 元素数）
# wait_stream_ordered 用较大的 collective 让依赖边保持可观测。
_WAIT_N = 64 * 1024 * 1024  # 256MB float32 -> all_reduce GPU 时间 ~20ms


def _key(ordinal: int) -> TaskKey:
    return TaskKey(0, 0, "dp", "dp", 0, 0, ordinal)


def _build_plan(scenario: str) -> Plan:
    n = _WAIT_N if scenario == "wait_stream_ordered" else _N
    ar = ("all_reduce", n * 4)
    if scenario == "fixed_reorder":
        entries = ((_key(1), *ar), (_key(0), *ar))
    elif scenario in ("delayed_ready", "no_wait_ready", "wait_stream_ordered"):
        entries = ((_key(0), *ar),)
    else:
        entries = ((_key(0), *ar), (_key(1), *ar))
    return Plan(version=0, window_id=0, entries=tuple(entries))


def _slow_fill(tensor: torch.Tensor, value: float, stream: torch.cuda.Stream) -> None:
    """在 side stream 上先跑一批 matmul 再填充 tensor，模拟慢 producer。

    matmul 与 ``fill_`` 都在 ``stream`` 上排队，与当前 stream 无依赖；
    tensor 在填充完成前保持 sentinel（调用方预先填 -1）。
    """
    with torch.cuda.stream(stream):
        a = torch.randn(2048, 2048, device="cuda")
        for _ in range(20):
            a = a @ a
        tensor.fill_(value)


def _warmup_nccl() -> None:
    """初始化 NCCL communicator 并预热 kernel。

    无预热时首 op 含 ~350ms 的 lazy init 延迟，会掩盖 no_wait_ready 的
    竞争演示并使 telemetry 失真。真实训练也有预热，所以预热是合理前提。
    """
    t = torch.ones(_N, dtype=torch.float32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()


def _make_all_reduce(key: TaskKey, rank: int, world: int):
    """普通 all_reduce intent：tensor 已就绪，无 ready event。"""
    tensor = torch.full((_N,), -1.0, device="cuda")
    tensor.fill_(rank + 1)

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
    """all_gather intent（只用于 op_mismatch 的 fail-stop 验证）。"""
    tensor = torch.full((_N,), -1.0, device="cuda")
    tensor.fill_(rank + 1)
    gathered = [torch.empty_like(tensor) for _ in range(world)]

    def launch():
        return dist.all_gather(gathered, tensor, async_op=True)

    intent = CommIntent(
        key=key, op="all_gather", tensor=tensor, process_group=None,
        num_bytes=_N * 4, launch_fn=launch, keepalive=tuple(gathered),
    )
    return intent


def _run(scenario: str, rank: int, world: int, local_rank: int, trace_dir: str) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    _warmup_nccl()  # 去掉首 op 的 lazy init 延迟（见函数注释）
    plan = _build_plan(scenario)
    sched = AdmissionScheduler(
        plan,
        local_group_ids=plan.group_ids(),
        executor=TorchProcessGroupExecutor(local_rank),
    )
    out: dict = {"rank": rank, "scenario": scenario, "status": "ok", "ops": []}

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        try:
            if scenario == "fifo":
                a, av = _make_all_reduce(_key(0), rank, world)
                b, bv = _make_all_reduce(_key(1), rank, world)
                wa = sched.submit(a)
                wb = sched.submit(b)
                assert wa.wait() and wb.wait()
                assert av() and bv()
                out["ops"] = [
                    {"key": 0, "ok": av()},
                    {"key": 1, "ok": bv()},
                ]

            elif scenario == "fixed_reorder":
                b, bv = _make_all_reduce(_key(1), rank, world)
                a, av = _make_all_reduce(_key(0), rank, world)
                wb = sched.submit(b)
                wa = sched.submit(a)
                assert wb.wait() and wa.wait()
                assert bv() and av()
                out["ops"] = [
                    {"key": 1, "ok": bv()},
                    {"key": 0, "ok": av()},
                ]

            elif scenario == "delayed_ready":
                # producer 慢速生成并记录 CUDA ready event；scheduler 在
                # 发射前让当前 stream wait_event，collective 排在其后。
                k = _key(0)
                tensor = torch.full((_N,), -1.0, device="cuda")
                prod_stream = torch.cuda.Stream()
                ready_ev = torch.cuda.Event()
                _slow_fill(tensor, rank + 1, prod_stream)
                ready_ev.record(prod_stream)
                intent = CommIntent(
                    key=k, op="all_reduce", tensor=tensor, process_group=None,
                    num_bytes=_N * 4,
                    launch_fn=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                                                      async_op=True),
                    ready_event=ready_ev,
                )
                w = sched.submit(intent)
                assert w.wait()
                ok = bool(torch.all(tensor == world * (world + 1) / 2).item())
                out["ops"] = [{"key": 0, "ok": ok}]

            elif scenario == "no_wait_ready":
                # 对照：同一慢 producer，但 intent 不带 ready event ->
                # collective 立即发射，与 producer 竞争，结果错误/不确定。
                k = _key(0)
                tensor = torch.full((_N,), -1.0, device="cuda")
                prod_stream = torch.cuda.Stream()
                _slow_fill(tensor, rank + 1, prod_stream)  # 不 record event
                intent = CommIntent(
                    key=k, op="all_reduce", tensor=tensor, process_group=None,
                    num_bytes=_N * 4,
                    launch_fn=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                                                      async_op=True),
                )
                w = sched.submit(intent)
                assert w.wait()
                ok = bool(torch.all(tensor == world * (world + 1) / 2).item())
                out["ops"] = [{"key": 0, "ok": ok}]

            elif scenario == "op_mismatch":
                a, av = _make_all_reduce(_key(0), rank, world)
                b = _make_all_gather(_key(1), rank, world)  # plan 期望 all_reduce
                wa = sched.submit(a)
                try:
                    sched.submit(b)
                    out["status"] = "error"
                    out["error"] = "expected ValidationError"
                except ValidationError as exc:
                    out["status"] = "validation_error"
                    out["error"] = str(exc)
                assert wa.wait() and av()
                out["ops"] = [{"key": 0, "ok": av()}]

            elif scenario == "wait_stream_ordered":
                # wait() 只把 NCCL completion dependency 接入 consumer stream。
                # CPU 返回时 consumer event 可以仍未触发；同步 consumer stream
                # 后依赖满足且 collective 输出可安全使用。
                k = _key(0)
                tensor = torch.full((_WAIT_N,), -1.0, device="cuda")
                tensor.fill_(rank + 1)

                def launch():
                    a = torch.randn(2048, 2048, device="cuda")
                    for _ in range(10):
                        a = a @ a
                    return dist.all_reduce(
                        tensor, op=dist.ReduceOp.SUM, async_op=True
                    )

                intent = CommIntent(
                    key=k, op="all_reduce", tensor=tensor, process_group=None,
                    num_bytes=_WAIT_N * 4, launch_fn=launch,
                )
                w = sched.submit(intent)
                consumer = torch.cuda.Stream()
                consumer_after = torch.cuda.Event()
                t0 = time.perf_counter()
                with torch.cuda.stream(consumer):
                    assert w.wait()
                    consumer_after.record()
                wait_s = round(time.perf_counter() - t0, 3)
                after_return = bool(consumer_after.query())
                consumer.synchronize()
                after_sync = bool(consumer_after.query())
                ok = bool(torch.all(tensor == world * (world + 1) / 2).item())
                out["ops"] = [
                    {"key": 0, "ok": ok,
                     "consumer_after_wait_return": after_return,
                     "consumer_after_stream_sync": after_sync,
                     "wait_s": wait_s},
                ]

            else:
                raise ValueError(f"unknown scenario: {scenario!r}")

            if scenario != "op_mismatch":
                sched.finish_window(timeout=30)
            out["launched_seq"] = {
                g: [k.as_list() for k in keys]
                for g, keys in sched.group_sequence_log().items()
            }
            for t in sched.timings():
                op = next(
                    (o for o in out["ops"] if o["key"] == t.key.ordinal), {}
                )
                op.update({
                    "ready_us": t.ready_record_ts,
                    "admit_us": t.admit_ts,
                    "submit_us": t.submit_ts,
                    "complete_us": t.complete_ts,
                    "actual_us": t.actual_duration_us,
                })
        except Exception as exc:  # noqa: BLE001 - 记录任何异常后仍打印结果
            out["status"] = "error"
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                sched.close()
            finally:
                dist.destroy_process_group()

    prof.export_chrome_trace(os.path.join(trace_dir, f"trace_r{rank}_{scenario}.json"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--trace-dir", default="/root/autodl-tmp")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # 进程经 CUDA_VISIBLE_DEVICES 只见一张卡，用 LOCAL_RANK 选设备。
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.makedirs(args.trace_dir, exist_ok=True)
    out = _run(args.scenario, rank, world, local_rank, args.trace_dir)
    print(json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
