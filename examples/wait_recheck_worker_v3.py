"""竞争测试：训练流发射 all_reduce 后，同行上紧跟的计算是否与 collective 竞争？

v2 已证明两条 CPU 完成信号（``Work.wait()`` / ``get_future().wait()``）都提前返回，
且当前流上裸 ``event.record()`` 不追踪 collective。但这两个都不能直接回答
admission-gate 架构最关键的前提：**训练流自己发射 all_reduce 后，后续计算是否
真的被排序在 collective 之后**。真实计算 op（读/写 tensor）会触发 c10d/allocator
的流依赖插入，裸 event.record() 不会——所以必须用真实竞争测试。

设计：s 流上 ``t = fill(rank+1)`` → ``all_reduce(t)``（真值应为 3）→ 立刻在 s 上
排 ``sx = t.sum()``（读 t）→ 记录 event 测量该计算实际执行的时机 → 全同步后取
``sx.item()``。

- stream-ordered（或 allocator 依赖）生效 → sum 等 collective 完成后才跑：
  ``ev_s≈280ms``，``sx = 3*_N``（读到 reduce 后）。
- 竞争 → sum 在 ~3ms 就跑，读到未 reduce 的 t：``ev_s≈3ms``，
  ``sx = (rank+1)*_N``（rank0 读到 1*_N，rank1 读到 2*_N）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32；真值 = (1+2)*_N = 3*_N


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _poll_until(ev: torch.cuda.Event) -> float:
    t0 = time.perf_counter()
    while not ev.query():
        time.sleep(0.001)
    return time.perf_counter() - t0


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "world": world, "expected": 3 * _N}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # throwaway 大 collective：吸收首个大 all_reduce 的初始化开销。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # 竞争测试：all_reduce 后同行紧跟一个读 t 的计算。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        sx = t.sum()  # 读 t 的真实计算 op（不是 event.record）
        ev = torch.cuda.Event()
        ev.record()  # 记录「sum 实际执行完成」的时刻
        ev_s = _poll_until(ev)
        torch.cuda.synchronize()
        val = sx.item()
        out["race"] = {
            "ev_s": round(ev_s, 6),
            "computed": val,
            "seen_reduced": bool(val == 3 * _N),
            "seen_unreduced_rank_fill": bool(val == (rank + 1) * _N),
        }

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
