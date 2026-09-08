"""v11 判别的归属实验（2×3080 Ti 盒子）：

v11 发现：AR 后**不 wait** 时 CPU 挡在 ``t.sum()``（~collective 全时长），
**先 wait()**（28µs 即返回）后却挡在 ``sx.item()``。两个判别场景：

S_ctrl_sum —— 无 collective，仅 fill(2GiB) 后立即 ``t.sum()``：
  若 plain sum 本身 CPU 异步（~µs 返回、item 才挡 fill 时长）→ v11 里 t.sum()
  的 506ms 挡块是「消费 collective 在途输出」特有；
  若 plain sum 也挡 fill 时长（~2.4ms）→ sum 在 CPU 侧会同步等 GPU（与 collective
  无关，需另解释为什么 afterwait 里 sum 又快速返回）。

S_ev_after_ar —— AR 后立刻 ``ev.record()`` + 轮询（不消费输出）：
  老 3090 盒子此处 event ~3ms 提前触发（不追踪 collective）。本盒若 ev_s≈503ms
  → event 在 s 上真的排在 collective 之后（流序成立）；若 ~3ms → 老盒子现象复现，
  event 不追踪（此时 t.sum 挡块另有来源，需再查）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29
_EXP = 3 * _N


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


def _poll_until(ev: torch.cuda.Event) -> float:
    t0 = time.perf_counter()
    while not ev.query():
        time.sleep(0.001)
    return (time.perf_counter() - t0) * 1e3  # ms


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "world": world, "expected": _EXP, "scen": {}}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # throwaway 大 collective 吸收 init。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # S_ctrl_sum：无 collective，fill 后立即 sum → item。
        tl = TL()
        torch.cuda.synchronize()
        tl.m("t0")
        t = _fill(rank)
        tl.m("sum_in")
        sx = t.sum()
        tl.m("sum_out")
        v = sx.item()
        tl.m("item_out")
        torch.cuda.synchronize()
        tl.m("sync_out")
        out["scen"]["S_ctrl_sum"] = {
            "timeline": tl.lines(),
            "sum_cpu_ms": round(tl.dt("sum_in", "sum_out"), 4),
            "item_block_ms": round(tl.dt("sum_out", "item_out"), 4),
            "correct": bool(v == _EXP),
        }

        # S_ev_after_ar：AR 后立刻 ev.record + 轮询（不消费输出）。
        tl = TL()
        torch.cuda.synchronize()
        tl.m("t0")
        t = _fill(rank)
        tl.m("ar_launch")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        tl.m("ev_rec")
        ev = torch.cuda.Event()
        ev.record()
        ev_ms = _poll_until(ev)
        tl.m("poll_done")
        torch.cuda.synchronize()
        tl.m("sync_out")
        out["scen"]["S_ev_after_ar"] = {
            "timeline": tl.lines(),
            "ev_s": round(ev_ms, 4),
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
