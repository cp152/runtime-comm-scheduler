"""M4 决策关口 worker：验证 worker 线程提交 ProcessGroupNCCL 是否可靠。

由 ``run_m4_gate.py`` 驱动，两 rank NCCL，每个场景跑在一个子进程里，向
stdout 打单行 JSON。场景回答 phase1-plan M4 的第一个决策关口「Python
thread 提交 ProcessGroupNCCL 是否足以支撑 prototype？」：

- ``worker_basic``：非主线程（admission worker）串行提交并等待 N 个
  all_reduce，数据全部正确 —— 证明 worker 线程能驱动 ProcessGroupNCCL。
- ``producer_continues``：主线程（producer）把 intent 入队后立即返回，
  worker 异步 drain；入队耗时应远小于 worker 的 collective 总耗时，且
  producer 的 GPU 计算与 worker 的 collective 时间上重叠 —— 证明 submit
  不阻塞 producer（验收点 1）。
- ``concurrent_submit``：主线程与 worker 线程同时向同一 communicator 并发
  提交（各自独立 tensor），探测 c10d/NCCL 的线程安全边界。信息性场景：
  M4 正式设计中只有 worker 线程提交，此场景只是摸清并发提交的可靠性包络。

这些历史 gate 在最终数据校验前统一 ``torch.cuda.synchronize()``，仅用于排空
实验 workload；scheduler 的正常 consumer 依赖传递不应采用设备级同步。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

_M = 2 * 1024 * 1024  # 8MB float32 tensor -> 单卡 all_reduce 亚毫秒级
_ITER = 32  # 每场景 collective 数


def _key_err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _warmup_nccl() -> None:
    t = torch.ones(_M, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()


def _expected(world: int) -> float:
    return world * (world + 1) / 2


def _check(tensors, world: int) -> list[bool]:
    exp = _expected(world)
    return [bool(torch.all(t == exp).item()) for t in tensors]


def worker_basic(rank: int, world: int) -> dict:
    """非主线程串行提交 N 个 all_reduce 并等待，全部数据正确。"""
    tensors = [torch.full((_M,), rank + 1, device="cuda") for _ in range(_ITER)]
    span_s: list[float] = []
    err: str | None = None

    def run() -> None:
        nonlocal err
        try:
            for t in tensors:
                t0 = time.perf_counter()
                w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
                w.wait()
                span_s.append(round(time.perf_counter() - t0, 4))
        except BaseException as e:  # noqa: BLE001
            err = _key_err(e)

    th = threading.Thread(target=run, name="admission-worker")
    th.start()
    th.join()
    torch.cuda.synchronize()
    oks = _check(tensors, world)
    return {
        "ok": err is None and all(oks),
        "error": err,
        "iter": _ITER,
        "total_span_s": round(sum(span_s), 4),
        "per_collective_ms": [round(s * 1e3, 3) for s in span_s],
    }


def producer_continues(rank: int, world: int) -> dict:
    """producer 入队即返回；入队耗时远小于 worker 总耗时，GPU 计算与
    collective 时间上重叠。"""
    tensors = [torch.full((_M,), rank + 1, device="cuda") for _ in range(_ITER)]
    q: "queue.Queue[tuple[int, torch.Tensor] | None]" = __import__("queue").Queue()
    worker_span: dict = {}
    err: str | None = None

    def worker() -> None:
        nonlocal err
        try:
            first = last = None
            while True:
                item = q.get()
                if item is None:
                    break
                idx, t = item
                now = time.perf_counter()
                if first is None:
                    first = now
                w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
                w.wait()
                last = time.perf_counter()
                worker_span[idx] = {"first": first, "last": last}
        except BaseException as e:  # noqa: BLE001
            err = _key_err(e)

    th = threading.Thread(target=worker, name="admission-worker")
    th.start()

    # producer：入队全部 intent 并计时（submit 到返回的墙钟）。
    enq_t0 = time.perf_counter()
    for i, t in enumerate(tensors):
        q.put((i, t))
    enq_t1 = time.perf_counter()

    # producer 继续执行：GPU 计算（与 worker 的 collective 重叠）。
    compute_t0 = time.perf_counter()
    a = torch.randn(2048, 2048, device="cuda")
    for _ in range(40):
        a = a @ a
    compute_t1 = time.perf_counter()

    q.put(None)
    th.join()
    torch.cuda.synchronize()

    firsts = [v["first"] for v in worker_span.values()]
    lasts = [v["last"] for v in worker_span.values()]
    worker_first = min(firsts)
    worker_last = max(lasts)
    oks = _check(tensors, world)
    enqueue_s = round(enq_t1 - enq_t0, 4)
    worker_total_s = round(worker_last - worker_first, 4)
    overlap = worker_first < compute_t1 and worker_last > compute_t0
    return {
        "ok": err is None and all(oks) and overlap,
        "error": err,
        "iter": _ITER,
        "enqueue_s": enqueue_s,
        "worker_total_s": worker_total_s,
        "producer_compute_s": round(compute_t1 - compute_t0, 4),
        "overlap_with_worker": overlap,
        "worker_busy_at_compute_start": worker_first < compute_t0,
    }


def concurrent_submit(rank: int, world: int) -> dict:
    """主线程与 worker 同时向同一 communicator 并发提交（信息性场景）。"""
    half = _ITER // 2
    main_tensors = [torch.full((_M,), rank + 1, device="cuda") for _ in range(half)]
    worker_tensors = [torch.full((_M,), rank + 1, device="cuda") for _ in range(half)]
    barrier = threading.Barrier(2)
    err: str | None = None

    def worker() -> None:
        nonlocal err
        try:
            barrier.wait()
            for t in worker_tensors:
                w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
                w.wait()
        except BaseException as e:  # noqa: BLE001
            err = _key_err(e)

    th = threading.Thread(target=worker, name="second-submitter")
    th.start()
    try:
        barrier.wait()
        for t in main_tensors:
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
            w.wait()
    except BaseException as e:  # noqa: BLE001
        err = _key_err(e)
    th.join()
    torch.cuda.synchronize()
    oks = _check(main_tensors + worker_tensors, world)
    return {"ok": err is None and all(oks), "error": err, "iter": _ITER}


SCENARIOS = {
    "worker_basic": worker_basic,
    "producer_continues": producer_continues,
    "concurrent_submit": concurrent_submit,
}


def _run(scenario: str, rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    _warmup_nccl()
    out: dict = {"rank": rank, "scenario": scenario, "status": "ok"}
    try:
        out.update(SCENARIOS[scenario](rank, world))
    except Exception as exc:  # noqa: BLE001 - 记录任何异常后仍打印结果
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        dist.destroy_process_group()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    out = _run(args.scenario, rank, world, local_rank)
    print(json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
