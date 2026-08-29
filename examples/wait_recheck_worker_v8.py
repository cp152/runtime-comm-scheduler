"""最后一个判别：280ms 是真实传输，还是完成信号延迟？

若 2GiB all_reduce（AR1）的**数据** ~3ms 就绪、280ms 只是完成信号被延迟，则：
在 AR1 之后立刻发一个小 all_reduce（AR2，1024，排在 comm stream 的 AR1 之后），
并立刻读 AR2 的 tensor —— AR2 若真的排在 AR1 后且 AR1 占住 comm stream，AR2 未
执行，读到 pre-reduce 值（race）；若 comm stream 早早释放，AR2 已完成，读到 reduce
后值。

- ``p_ok``：AR2 的数据是否正确（race → False，有序 → True）。
- ``t_ok``：AR1 的 tensor 在同流读是否正确（v3/v4 已证 True，复测）。
- ``ev_s``：读 op 的执行时机（参照 fill ~3ms）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29
_P = 1024


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

    out: dict = {"rank": rank, "expected_p": 3 * _P, "expected_t": 3 * _N}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        torch.cuda.synchronize()
        t = torch.full((_N,), 1.0, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # AR1(2GiB) 后立刻 AR2(1024)，立刻读两者。
        torch.cuda.synchronize()
        t = torch.full((_N,), float(rank + 1), device="cuda")
        w1 = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)  # AR1
        p = torch.full((_P,), float(rank + 1), device="cuda")
        w2 = dist.all_reduce(p, op=dist.ReduceOp.SUM, async_op=True)  # AR2 排后
        sp = p.sum()  # 立刻读 AR2 的结果
        st = t.sum()  # 立刻读 AR1 的结果
        ev = torch.cuda.Event()
        ev.record()
        ev_s = _poll(ev)
        t_ref = time.perf_counter()
        torch.cuda.synchronize()
        sync_after_s = time.perf_counter() - t_ref
        out["race2"] = {
            "ev_s": round(ev_s, 6),
            "sync_after_s": round(sync_after_s, 6),
            "p_computed": sp.item(),
            "p_ok": bool(sp.item() == 3 * _P),
            "t_computed": st.item(),
            "t_ok": bool(st.item() == 3 * _N),
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
