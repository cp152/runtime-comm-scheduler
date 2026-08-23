"""M4 单 rank worker：两 rank NCCL 上的异步 admission worker harness。

由 ``run_m4.py`` 驱动，NCCL backend，``AdmissionScheduler(worker=True)``：
deferred launch 由专门的 worker 线程在显式 communication stream 上执行，
producer（主线程）的 ``submit`` 只校验 + park + 唤醒并立即返回。

场景覆盖 M4 验收点：

- **producer 提交后继续执行**：``producer_overlap`` 中主线程提交 N 个
  intent 后立即做 GPU 计算，worker 异步 drain；入队耗时应远小于 worker 的
  collective 总耗时，且计算与 collective 时间上重叠。
- **没有错误的 sequence divergence**：每个场景的 ``launched_seq`` 由 driver
  跨两 rank 对比；``out_of_order_submit`` 专门验证乱序提交经 worker 仍被
  强制为计划顺序。
- **没有错误的 stream dependency**：``delayed_ready`` 中 producer 在 side
  stream 记录 CUDA ready event，worker 在 comm stream 上 wait_event 后发射，
  collective 排在 producer 之后（数据正确）；``no_wait_ready`` 去掉该 event
  作为对照（竞争/不确定）。

其余场景（fifo/fixed_reorder/op_mismatch）与 M3 对齐，验证 worker 模式下
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

from runtime_comm_scheduler import AdmissionScheduler, CommIntent, Plan, TaskKey
from runtime_comm_scheduler.validate import ValidationError

_N = 1024           # 普通场景 tensor 大小（float32 元素数）
_BIG_N = 2 * 1024 * 1024  # 8MB，让 GPU 时间可测、依赖边可见
_OVERLAP_N = 16     # producer_overlap 的 intent 数量


def _key(ordinal: int) -> TaskKey:
    return TaskKey(0, 0, "dp", "dp", 0, 0, ordinal)


def _build_plan(scenario: str) -> Plan:
    if scenario == "producer_overlap":
        # 16 个 8MB all_reduce，覆盖 producer_overlap 提交的全部 intent。
        ar = ("all_reduce", _BIG_N * 4)
        entries = tuple((_key(i), *ar) for i in range(_OVERLAP_N))
        return Plan(version=0, window_id=0, entries=entries)
    n = _BIG_N if scenario in ("delayed_ready", "no_wait_ready") else _N
    ar = ("all_reduce", n * 4)
    if scenario == "fixed_reorder":
        entries = ((_key(1), *ar), (_key(0), *ar))
    else:
        entries = ((_key(0), *ar), (_key(1), *ar))
    return Plan(version=0, window_id=0, entries=tuple(entries))


def _slow_fill(tensor: torch.Tensor, value: float, stream: torch.cuda.Stream) -> None:
    """在 side stream 上先跑一批 matmul 再填充 tensor，模拟慢 producer。"""
    with torch.cuda.stream(stream):
        a = torch.randn(2048, 2048, device="cuda")
        for _ in range(20):
            a = a @ a
        tensor.fill_(value)


def _warmup_nccl() -> None:
    t = torch.ones(_N, dtype=torch.float32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()


def _make_all_reduce(key: TaskKey, rank: int, world: int, n: int = _N):
    """普通 all_reduce intent：tensor 已就绪，无 ready event。"""
    tensor = torch.full((n,), -1.0, device="cuda")
    tensor.fill_(rank + 1)

    def launch():
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

    def verify() -> bool:
        expected = world * (world + 1) / 2
        return bool(torch.all(tensor == expected).item())

    intent = CommIntent(
        key=key, op="all_reduce", tensor=tensor, process_group=None,
        num_bytes=n * 4, launch_fn=launch,
    )
    return intent, verify


def _run(scenario: str, rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    _warmup_nccl()  # 去掉首 op 的 lazy init 延迟（见 M3 注释）
    # M4：scheduler 持有显式 communication stream，worker 在其上发射。
    comm_stream = torch.cuda.Stream()
    sched = AdmissionScheduler(_build_plan(scenario), worker=True,
                               comm_stream=comm_stream)
    out: dict = {"rank": rank, "scenario": scenario, "status": "ok", "ops": []}

    try:
        if scenario == "fifo":
            a, av = _make_all_reduce(_key(0), rank, world)
            b, bv = _make_all_reduce(_key(1), rank, world)
            wa = sched.submit(a)
            wb = sched.submit(b)
            assert wa.wait() and wb.wait()
            assert av() and bv()
            out["ops"] = [{"key": 0, "ok": av()}, {"key": 1, "ok": bv()}]

        elif scenario == "fixed_reorder":
            b, bv = _make_all_reduce(_key(1), rank, world)
            a, av = _make_all_reduce(_key(0), rank, world)
            wb = sched.submit(b)
            wa = sched.submit(a)
            assert wb.wait() and wa.wait()
            assert bv() and av()
            out["ops"] = [{"key": 1, "ok": bv()}, {"key": 0, "ok": av()}]

        elif scenario == "out_of_order_submit":
            # 先提交 k1（非队首），worker 把它 parked；k0 提交后才按计划
            # [k0, k1] 发射。验证 worker 强制计划顺序（跨 rank 对比）。
            b, bv = _make_all_reduce(_key(1), rank, world)
            a, av = _make_all_reduce(_key(0), rank, world)
            wb = sched.submit(b)          # parked
            wa = sched.submit(a)          # worker drain -> k0 后 k1
            assert wa.wait() and wb.wait()
            assert av() and bv()
            out["ops"] = [{"key": 0, "ok": av()}, {"key": 1, "ok": bv()}]

        elif scenario == "delayed_ready":
            # producer 慢速生成并记录 CUDA ready event；worker 在 comm stream
            # 上 wait_event 后发射，collective 排在 producer 之后。
            k = _key(0)
            tensor = torch.full((_BIG_N,), -1.0, device="cuda")
            prod_stream = torch.cuda.Stream()
            ready_ev = torch.cuda.Event()
            _slow_fill(tensor, rank + 1, prod_stream)
            ready_ev.record(prod_stream)
            intent = CommIntent(
                key=k, op="all_reduce", tensor=tensor, process_group=None,
                num_bytes=_BIG_N * 4,
                launch_fn=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                                                  async_op=True),
                ready_event=ready_ev,
            )
            w = sched.submit(intent)
            assert w.wait()
            ok = bool(torch.all(tensor == world * (world + 1) / 2).item())
            out["ops"] = [{"key": 0, "ok": ok}]

        elif scenario == "no_wait_ready":
            # 对照：同一慢 producer，但 intent 不带 ready event -> worker
            # 立即发射，与 producer 竞争，结果错误/不确定。
            k = _key(0)
            tensor = torch.full((_BIG_N,), -1.0, device="cuda")
            prod_stream = torch.cuda.Stream()
            _slow_fill(tensor, rank + 1, prod_stream)  # 不 record event
            intent = CommIntent(
                key=k, op="all_reduce", tensor=tensor, process_group=None,
                num_bytes=_BIG_N * 4,
                launch_fn=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                                                  async_op=True),
            )
            w = sched.submit(intent)
            assert w.wait()
            ok = bool(torch.all(tensor == world * (world + 1) / 2).item())
            out["ops"] = [{"key": 0, "ok": ok}]

        elif scenario == "producer_overlap":
            # 验收点「producer 提交 intent 后可以继续执行」：主线程提交 N 个
            # intent（submit 只入队 + 唤醒，立即返回），随后做 GPU 计算；
            # worker 异步 drain。入队耗时 << worker collective 总耗时，且
            # 计算与 collective 时间上重叠。
            intents, verifies, works = [], [], []
            for i in range(_OVERLAP_N):
                it, vf = _make_all_reduce(_key(i), rank, world, _BIG_N)
                intents.append(it)
                verifies.append(vf)
            enq_t0 = time.perf_counter()
            for it in intents:
                works.append(sched.submit(it))
            enq_t1 = time.perf_counter()
            busy_at_start = not works[0].is_completed()
            compute_t0 = time.perf_counter()
            a = torch.randn(2048, 2048, device="cuda")
            for _ in range(30):
                a = a @ a
            compute_t1 = time.perf_counter()
            assert all(w.wait() for w in works)
            assert all(v() for v in verifies)
            out["ops"] = [
                {"key": i, "ok": v()} for i, v in enumerate(verifies)
            ]
            out["producer"] = {
                "enqueue_s": round(enq_t1 - enq_t0, 4),
                "worker_busy_at_compute_start": busy_at_start,
                "producer_compute_s": round(compute_t1 - compute_t0, 4),
            }

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

        else:
            raise ValueError(f"unknown scenario: {scenario!r}")

        out["launched_seq"] = {
            g: [k.as_list() for k in keys]
            for g, keys in sched.sequence_log().items()
        }
        for t in sched.timings():
            op = next(
                (o for o in out["ops"] if o["key"] == t.key.ordinal), {}
            )
            op.update({
                "ready_us": t.ready_ts,
                "admit_us": t.admit_ts,
                "submit_us": t.submit_ts,
                "complete_us": t.complete_ts,
                "actual_us": t.actual_duration_us,
            })
    except Exception as exc:  # noqa: BLE001 - 记录任何异常后仍打印结果
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sched.close()
        dist.destroy_process_group()
    return out


def _make_all_gather(key: TaskKey, rank: int, world: int):
    """all_gather intent（只用于 op_mismatch 的 fail-stop 验证）。"""
    tensor = torch.full((_N,), -1.0, device="cuda")
    tensor.fill_(rank + 1)
    gathered = [torch.empty_like(tensor) for _ in range(world)]

    def launch():
        return dist.all_gather(gathered, tensor, async_op=True)

    intent = CommIntent(
        key=key, op="all_gather", tensor=tensor, process_group=None,
        num_bytes=_N * 4, launch_fn=launch,
    )
    return intent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # 进程经 CUDA_VISIBLE_DEVICES 只见一张卡，用 LOCAL_RANK 选设备。
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    out = _run(args.scenario, rank, world, local_rank)
    print(json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
