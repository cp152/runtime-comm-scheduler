"""v16（一次性判别）：nowait 特例态的 500ms 卡点 = reduction 特有，还是任何 reader launch？

v11 T_item_nowait（nowait 读对）500ms 落在 t.sum()(+ev.record) 发射端；v14 rep0 /
v15 i0 同。想判别两点：
  (i)  卡在 reader kernel 内，还是卡在紧随的 ev.record() 内？
  (ii) 是 reduction（t.sum()）路径特有，还是「向带 pending NCCL 的同一流/张量发射
       任何 reader kernel」都卡？

做法：v14 的 preamble 精确复刻（保证第 1 个未 wait collective 进入特例态），
第 1 个 measured collective（rep0）让两 rank 用不同 reader，但 AR 仍配对：
  rank0 → reader = sum  （reduction：读 t，写 1 个标量）
  rank1 → reader = copy （非 reduction：读 t，写 2GiB out）
各自把 reader kernel 与 ev.record() 用独立 mark 拆开（reader_out / ev_rec），
再看 item/回读把剩余挡在哪、读对与否。

判定（rank 各自 rep0）：
  * sum 与 copy 都 reader_dt≈500ms → 特例态是「发射任何 reader 都被 host 串行」，
    与 reduction 无关；(ii) = 通用
  * 只有 sum reader_dt≈500ms、copy reader_dt≈2ms → reduction 路径内有隐藏 host 同步
  * 若 reader_dt 小且读到未 reduce（rank0 1N/rank1 非 3.0）→ 特例态本次未 arm，重试无效

rep1（同一 reader、第 2 个未 wait collective）= de-armed 常态对照（期望 v14 rep1：
reader_dt≈0.05ms、脏读、sync_after≈490ms）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32
_EXP = 3 * _N  # 两 rank fill 1.0/2.0，SUM 后 3.0


class TL:
    """轻量时间线：记录 (label, perf_counter)，可算 dt。"""

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
            out.append(
                {"st": lb, "cum_ms": round((t - base) * 1e3, 4),
                 "dt_ms": round((t - prev) * 1e3, 4)}
            )
            prev = t
        return out


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    reader = "sum" if rank == 0 else "copy"
    # copy reader 的输出缓冲（2GiB），一次性分配，避免在计时段内触发 alloc。
    out = torch.empty_like(_fill(rank))
    out.zero_()
    torch.cuda.synchronize()

    res: dict = {"rank": rank, "world": world, "expected": _EXP, "reader": reader, "reps": []}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # —— v14 精确 preamble：确保第 1 个未 wait collective 进特例态 ——
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        for r in range(2):  # rep0 = 特例候选，rep1 = de-armed 常态对照
            tl = TL()
            torch.cuda.synchronize()
            tl.m("t0")
            t = _fill(rank)
            tl.m("ar_launch")
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

            tl.m("reader_in")
            if reader == "sum":
                sx = t.sum()
            else:
                out.copy_(t)
            tl.m("reader_out")

            ev = torch.cuda.Event()
            ev.record()
            tl.m("ev_rec")

            tl.m("item_in")
            if reader == "sum":
                v = sx.item()
            else:
                v = out[0].item()  # 只读首元素做 host 回读（时序对称的 item）
            tl.m("item_out")

            torch.cuda.synchronize()
            tl.m("sync_out")

            rep = {
                "rep": r,
                "timeline": tl.lines(),
                "reader_dt_ms": round(tl.dt("reader_in", "reader_out"), 4),
                "evrec_dt_ms": round(tl.dt("reader_out", "ev_rec"), 4),
                "item_dt_ms": round(tl.dt("item_in", "item_out"), 4),
                "sync_after_ms": round(tl.dt("item_out", "sync_out"), 4),
                "readback0": float(v),
                "ev_fired_by_end": bool(ev.query()),
            }
            if reader == "copy":
                # sync 后的权威正确性检查（不计时）：copy 读到的 t 是否全为 3.0
                rep["all3"] = bool((out == 3.0).all().item())
            else:
                rep["correct"] = bool(v == _EXP)
            res["reps"].append(rep)
            del t
            if reader == "sum":
                del sx

    dist.destroy_process_group()
    return res


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(json.dumps(_run(rank, world, local_rank)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
