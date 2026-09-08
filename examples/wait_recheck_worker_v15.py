"""交替验证：正确性是否严格跟随显式 w.wait()？

v14 显示：warmup 后第 1 个 collective 不 wait 也正确（sum 挡住等完），
第 2 个起不 wait 就读脏数据（sum 50µs 返回）。v15 交替 nowait/wait 8 遍：
- i=0 nowait：复现 v14 的「第 1 个特例正确」？
- i>=1：nowait → 脏数据；wait → 正确（严格跟随）？

每遍：sync → fill → AR(async) → (wait?) → sum → item → sync，记录 sum_cpu/item/
computed/correct。模式 = "wait"(i 奇) / "nowait"(i 偶)。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29
_EXP = 3 * _N
_R = 8


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
            mode = "nowait" if i % 2 == 0 else "wait"
            torch.cuda.synchronize()
            t = _fill(rank)
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

            wait_ms = 0.0
            if mode == "wait":
                ta = time.perf_counter()
                w.wait()
                wait_ms = (time.perf_counter() - ta) * 1e3

            tb = time.perf_counter()
            sx = t.sum()
            sum_cpu = (time.perf_counter() - tb) * 1e3

            tc = time.perf_counter()
            v = sx.item()
            item_block = (time.perf_counter() - tc) * 1e3

            torch.cuda.synchronize()

            out["reps"].append(
                {
                    "i": i,
                    "mode": mode,
                    "wait_ms": round(wait_ms, 4),
                    "sum_cpu_ms": round(sum_cpu, 4),
                    "item_block_ms": round(item_block, 4),
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
