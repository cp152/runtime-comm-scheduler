"""v19（nsys 靶子）：干净复刻 v14 rep0(首条 no-wait sum，预期阻塞)/rep1(快速脏读)，
不加 torch.profiler / sync-debug，只加 nvtx 范围，供 nsys 按 rank 逐调用打点定位
~500ms host 阻塞落在哪个 CUDA runtime API（Synchronize / LaunchKernel / Malloc）。
"""
import json
import os
import sys

import torch
import torch.distributed as dist

_N = 1 << 29
_EXP = 3 * _N


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "world": world, "expected": _EXP, "reps": []}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        for r in range(2):
            torch.cuda.nvtx.range_push(f"rep{r}_start")
            torch.cuda.synchronize()
            t = _fill(rank)
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
            torch.cuda.nvtx.range_pop()

            import time as _t

            torch.cuda.nvtx.range_push(f"rep{r}_sum")
            ta = _t.perf_counter()
            sx = t.sum()
            sum_cpu = (_t.perf_counter() - ta) * 1e3
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"rep{r}_item")
            tb = _t.perf_counter()
            v = sx.item()
            item_ms = (_t.perf_counter() - tb) * 1e3
            torch.cuda.nvtx.range_pop()

            tc = _t.perf_counter()
            torch.cuda.synchronize()
            sync_after = (_t.perf_counter() - tc) * 1e3

            out["reps"].append(
                {
                    "rep": r,
                    "sum_cpu_ms": round(sum_cpu, 4),
                    "item_block_ms": round(item_ms, 4),
                    "sync_after_item_ms": round(sync_after, 4),
                    "computed": v,
                    "correct": bool(v == _EXP),
                }
            )
            del t
            del sx

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
