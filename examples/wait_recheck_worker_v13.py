"""归因 t.sum() 的 CPU 阻塞：是等 collective，还是 sum 固有 CPU 成本？

v11/v12 观测：
- 无 collective 纯 sum：CPU 挡 ~14ms（S_ctrl_sum）；
- AR 在途、不 wait：sum 挡 ~506ms ≈ collective（T_item_nowait）；
- AR 在途、先 wait()（28µs）：sum 快速返回 0.09ms，item 才挡 510ms（T_item_afterwait）。

2×2 微判别（每场景 fresh 2GiB tensor、sync 隔离）：
S_A：fill → [GPU 在途] → sum      （复现 14ms？）
S_B：fill → sync(GPU 空) → sum     （若 ≈µs：14ms 是等 fill 在途；若仍 ~14ms：sum 固有）
S_C：fill → AR async → sum          （复现 506ms？= 等 collective 在途）
S_D：fill → AR async → wait → sum   （复现 0.09ms？= 有流序后 sum 不再挡）
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29


class TL:
    def __init__(self) -> None:
        self._items: list[tuple[str, float]] = []
        self._idx: dict[str, float] = {}

    def m(self, lb: str) -> None:
        self._idx[lb] = time.perf_counter()
        self._items.append((lb, self._idx[lb]))

    def dt(self, a: str, b: str) -> float:
        return (self._idx[b] - self._idx[a]) * 1e3

    def lines(self) -> list[dict]:
        base = self._items[0][1]
        out, prev = [], base
        for lb, t in self._items:
            out.append({"st": lb, "cum_ms": round((t - base) * 1e3, 4),
                        "dt_ms": round((t - prev) * 1e3, 4)})
            prev = t
        return out


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "world": world, "scen": {}}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        def scen(name: str, mode: str) -> None:
            tl = TL()
            torch.cuda.synchronize()
            tl.m("t0")
            t = _fill(rank)
            if mode == "B":
                torch.cuda.synchronize()  # GPU 排空后再 sum
            if mode in ("C", "D"):
                w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
                if mode == "D":
                    w.wait()
            tl.m("sum_in")
            sx = t.sum()
            tl.m("sum_out")
            v = sx.item()
            tl.m("item_out")
            torch.cuda.synchronize()
            tl.m("sync_out")
            out["scen"][name] = {
                "timeline": tl.lines(),
                "sum_cpu_ms": round(tl.dt("sum_in", "sum_out"), 4),
                "item_block_ms": round(tl.dt("sum_out", "item_out"), 4),
            }
            del t, sx

        scen("S_A_fill_inflight", "A")
        scen("S_B_fill_done", "B")
        scen("S_C_ar_inflight", "C")
        scen("S_D_ar_waited", "D")

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
