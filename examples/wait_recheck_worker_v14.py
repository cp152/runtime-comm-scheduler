"""v3-form 重复性：fill → AR(async) → sum → item → sync，连跑 R 遍。

v11 vs v13 在同构场景（AR 在途、不 wait、直接 t.sum()）给出相反结果：
- v11 T_item_nowait：sum 挡 506ms（等完 collective，读到正确）；
- v13 S_C：sum 挡 0.042ms 即返回，item 挡 4.8ms，collective 到 sync 还在跑（529ms）——
  后者几乎肯定读到未 reduce 数据（v13 漏记 computed）。

本 worker 重复 R 遍，记录每遍：sum_cpu_ms / item_block_ms / sync_after_item_ms /
computed / correct(==3*N)。若正确性随遍变化或 sum_cpu 跳跃 → 行为是非确定/竞态，
v3「sum 读到正确值」本身是竞态的一个分支，不能当稳定结论。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29
_EXP = 3 * _N
_R = 6


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

        for i in range(_R):
            torch.cuda.synchronize()
            t = _fill(rank)
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

            ta = time.perf_counter()
            sx = t.sum()
            sum_cpu = (time.perf_counter() - ta) * 1e3

            tb = time.perf_counter()
            v = sx.item()
            item_block = (time.perf_counter() - tb) * 1e3

            tc = time.perf_counter()
            torch.cuda.synchronize()
            sync_after = (time.perf_counter() - tc) * 1e3

            out["reps"].append(
                {
                    "rep": i,
                    "sum_cpu_ms": round(sum_cpu, 4),
                    "item_block_ms": round(item_block, 4),
                    "sync_after_item_ms": round(sync_after, 4),
                    "computed": v,
                    "correct": bool(v == _EXP),
                }
            )

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
