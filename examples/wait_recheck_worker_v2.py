"""wait 语义重测 v2：孤立测量，消除 v1 的 back-to-back 污染。

v1 的问题：三个实验的 collective 背靠背发射，GPU 排队，导致「第一个 event 在
~2ms 触发、后两个在 ~290ms 触发」的模糊结果。v2 每段测量前
``torch.cuda.synchronize()`` 重置 GPU，每段只测一个 **fresh collective**，并把
「stream-ordered event」「裸 wait」「future wait」三者与「从发射到 device idle 的
真实时长」直接对照：

- **fill_s**：纯 fill 基线（当前流本地工作的耗时下限）。
- **S1**  ``ev_s`` + ``sync_after_ev_s``：当前流 event 是否追踪 collective 完成？
  - stream-ordered 生效 → ev_s≈真实时长，sync_after_ev_s≈0；
  - 未生效（event 只跟 fill）→ ev_s≈fill_s，sync_after_ev_s≈288ms（collective 仍在跑）。
- **S2**  ``wait_s`` + ``sync_after_wait_s``：裸 ``Work.wait()`` 返回时 GPU 是否完成？
  - 可靠 → wait_s≈真实时长，sync_after_wait_s≈0；
  - 提前返回 → wait_s≈0.01ms，sync_after_wait_s≈290ms（M3 发现 #2 复现）。
- **S3**  同 S2，测 ``get_future().wait()``。
- **S4**  ``sync_total_s``：collective 的真实 GPU 时长参照。

warmup 后加一个 2GiB 的 throwaway collective，吸收「首个大 collective」的潜在
初始化开销，确保 S1-S4 测的都是稳定态。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _poll_until(ev: torch.cuda.Event) -> float:
    """轮询 event 直到触发，返回墙钟秒数（非阻塞 query，不做 device sync）。"""
    t0 = time.perf_counter()
    while not ev.query():
        time.sleep(0.001)
    return time.perf_counter() - t0


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    # warmup：去掉 NCCL 首 op 的 lazy init 延迟。
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "world": world}
    s = torch.cuda.Stream()  # 非 legacy 显式流；所有测量都在它上面

    with torch.cuda.stream(s):
        # throwaway 大 collective：吸收首个大 all_reduce 的初始化/分配开销。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # fill 基线：当前流本地工作（collective 之前的 fill 自身耗时）。
        t = _fill(rank)
        ev = torch.cuda.Event()
        ev.record()
        fill_s = _poll_until(ev)
        torch.cuda.synchronize()
        out["fill_s"] = round(fill_s, 6)

        # S1：stream-ordered event 是否追踪 collective。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        ev = torch.cuda.Event()
        ev.record()
        q0 = bool(ev.query())
        ev_s = _poll_until(ev)
        t_ref = time.perf_counter()
        torch.cuda.synchronize()
        sync_after_ev_s = time.perf_counter() - t_ref
        out["S1"] = {
            "q0_immediate": q0,
            "ev_s": round(ev_s, 6),
            "sync_after_ev_s": round(sync_after_ev_s, 6),
        }

        # S2：裸 Work.wait()，看返回时 GPU 是否已完成。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        tw0 = time.perf_counter()
        ok = w.wait()
        wait_s = time.perf_counter() - tw0
        t_ref = time.perf_counter()
        torch.cuda.synchronize()
        sync_after_wait_s = time.perf_counter() - t_ref
        out["S2"] = {
            "wait_s": round(wait_s, 6),
            "sync_after_wait_s": round(sync_after_wait_s, 6),
            "ok": ok,
        }

        # S3：get_future().wait()。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        fut = w.get_future()
        tf0 = time.perf_counter()
        fut.wait()
        fut_s = time.perf_counter() - tf0
        t_ref = time.perf_counter()
        torch.cuda.synchronize()
        sync_after_fut_s = time.perf_counter() - t_ref
        out["S3"] = {
            "fut_s": round(fut_s, 6),
            "sync_after_fut_s": round(sync_after_fut_s, 6),
        }

        # S4：collective 真实时长参照（从发射到 device idle）。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        ts0 = time.perf_counter()
        torch.cuda.synchronize()
        sync_total_s = time.perf_counter() - ts0
        out["S4"] = {"sync_total_s": round(sync_total_s, 6)}

    dist.destroy_process_group()
    return out


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(json.dumps(_run(rank, world, local_rank)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
