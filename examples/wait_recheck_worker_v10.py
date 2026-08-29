"""读输出 vs 不读：sync 耗时是否被「读 collective 输出」改变？

v8 读输出后 sync=41µs；v9 不读时 sync=280ms。本 worker 直接对照：
- **no_read**：fill → all_reduce → 立即 sync（应 ≈280ms）。
- **read**：fill → all_reduce → 读 t.sum() → 立即 sync。
  - sync≈µs（读迫使 collective 完成，传输其实很快）→ 280ms 是 sync 触发的开销；
  - sync≈280ms（读无帮助）→ 280ms 是真实传输，v8 的 41µs 另有原因。

同时记录 read 分支里 ``t.sum()`` 的取值（正确性）与 event 时机。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29


def _poll(ev: torch.cuda.Event) -> float:
    t0 = time.perf_counter()
    while not ev.query():
        time.sleep(0.001)
    return time.perf_counter() - t0


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "expected": 3 * _N}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        torch.cuda.synchronize()
        t = torch.full((_N,), 1.0, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # no_read：不读，立即 sync。
        torch.cuda.synchronize()
        t = torch.full((_N,), float(rank + 1), device="cuda")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        ts0 = time.perf_counter()
        torch.cuda.synchronize()
        out["no_read_sync_s"] = round(time.perf_counter() - ts0, 6)

        # read：读 t.sum() 后再 sync。
        torch.cuda.synchronize()
        t = torch.full((_N,), float(rank + 1), device="cuda")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        sx = t.sum()
        ev = torch.cuda.Event()
        ev.record()
        ev_s = _poll(ev)
        ts0 = time.perf_counter()
        torch.cuda.synchronize()
        out["read_sync_s"] = round(time.perf_counter() - ts0, 6)
        out["read_ev_s"] = round(ev_s, 6)
        out["read_sum_correct"] = bool(sx.item() == 3 * _N)

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
