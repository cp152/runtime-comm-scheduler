"""隔离测试：``torch.cuda.synchronize()`` 本身是否慢（~280ms）？

v3/v4 悖论的关键假设之一：如果 sync 在**空闲**时也 ~280ms，则「280ms collective」
是假象（数据其实 ~3ms 就绪），一切矛盾消解；如果空闲 sync 是 µs 级，则 280ms 是
真实设备工作，sum@3ms 读到正确数据需要另一条未被理解的排序机制。

本 worker 对三种状态计时 sync：完全空闲、极小 op、2GiB fill（无 collective）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    out: dict = {"rank": rank}

    torch.cuda.synchronize()  # 预热

    # 完全空闲
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    out["idle_sync_s"] = round(time.perf_counter() - t0, 6)

    # 极小 op（0-dim 写）
    x = torch.zeros(1, device="cuda")
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    out["tiny_sync_s"] = round(time.perf_counter() - t0, 6)

    # 2GiB fill（无 collective）
    t = torch.full((_N,), 1.0, device="cuda")
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    out["fill_sync_s"] = round(time.perf_counter() - t0, 6)

    # fill + 极小 all_reduce（1024）
    if world > 1:
        dist.init_process_group(backend="nccl")
        t0 = torch.full((1024,), 1.0, device="cuda")
        w = dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True)
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        out["tiny_ar_sync_s"] = round(time.perf_counter() - t0, 6)
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
