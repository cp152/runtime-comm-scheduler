"""v3 核心复刻 + 逐语句 CPU 时间线：CPU 到底在哪条语句被阻塞？

问题：v3 里 all_reduce(async) 后同行紧跟 ``t.sum()`` 能读到正确（reduce 后）数据，
但当前流 event 却 ~ms 就触发。疑点：
1. ``sum`` 是否真的与通信 stream 同步（被排序到 collective 之后）？
2. CPU 在哪条语句上真正被挡住（block）？

设计：给每场景每条语句打 CPU wall-clock 时间戳，dt 大 = CPU 阻塞。关键数字：
- ``block_item_ms``：``sx.item()`` 调用本身的 CPU 阻塞时长。若 sum 被排到 collective
  之后才执行，item 应挡到 collective 真完成（≈ AR 墙钟）；若数据 ~ms 就绪，item 只挡
  几 ms。
- ``sync_out`` 的 dt：item 返回后再 sync 还有没有剩余 GPU 工作（读是否「吞掉」trailing）。
- 参照：T_fill（fill 墙钟）、T_noread（不读输出，发完直接 sync）。

语义修正验证：Work.wait() 同步的是 current stream 与通信任务，不挡 CPU。
``T_item_afterwait`` 里 w.wait() 若 ~0ms 返回、其后 item 仍挡到 collective 完成，
即支持该模型（wait 只插入流序，CPU 何时真正等取决于你何时去读）。

注意：事件只 record 不轮询，避免 poll 自身阻塞污染时间线。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32
_EXP = 3 * _N  # 两 rank fill 值 1.0/2.0，SUM 后每元素 3.0


class TL:
    """轻量时间线：记录 (label, perf_counter)，可算 dt。"""

    def __init__(self) -> None:
        self._items: list[tuple[str, float]] = []
        self._idx: dict[str, float] = {}

    def m(self, lb: str) -> None:
        self._idx[lb] = time.perf_counter()
        self._items.append((lb, self._idx[lb]))

    def dt(self, a: str, b: str) -> float:
        """a→b 的毫秒墙钟。"""
        return (self._idx[b] - self._idx[a]) * 1e3

    def lines(self) -> list[dict]:
        base = self._items[0][1]
        out = []
        prev = base
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

    out: dict = {"rank": rank, "world": world, "expected": _EXP, "scen": {}}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # throwaway 大 collective：吸收首个大 all_reduce 的初始化开销。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # T_fill：fill 单独 GPU 墙钟（参照）。
        tl = TL()
        torch.cuda.synchronize()
        tl.m("t0")
        t = _fill(rank)
        tl.m("pre_sync")
        torch.cuda.synchronize()
        tl.m("post_sync")
        out["scen"]["T_fill"] = {
            "timeline": tl.lines(),
            "fill_gpu_ms": round(tl.dt("pre_sync", "post_sync"), 4),
        }

        # T_noread：fill → AR → 立即 sync（不读输出，参照 AR+fill 墙钟）。
        tl = TL()
        torch.cuda.synchronize()
        tl.m("t0")
        t = _fill(rank)
        tl.m("ar_launch")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        tl.m("pre_sync")
        torch.cuda.synchronize()
        tl.m("post_sync")
        out["scen"]["T_noread"] = {
            "timeline": tl.lines(),
            "sync_ms": round(tl.dt("pre_sync", "post_sync"), 4),
        }

        # T_item_nowait：v3 核心。AR 后不 wait，直接 sum → item → sync。
        tl = TL()
        torch.cuda.synchronize()
        tl.m("t0")
        t = _fill(rank)
        tl.m("ar_launch")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        tl.m("sum_launch")
        sx = t.sum()
        tl.m("ev_record")
        ev = torch.cuda.Event()
        ev.record()
        tl.m("item_in")
        v = sx.item()
        tl.m("item_out")
        torch.cuda.synchronize()
        tl.m("sync_out")
        out["scen"]["T_item_nowait"] = {
            "timeline": tl.lines(),
            "computed": v,
            "correct": bool(v == _EXP),
            "item_block_ms": round(tl.dt("item_in", "item_out"), 4),
            "sync_after_item_ms": round(tl.dt("item_out", "sync_out"), 4),
            "ev_fired_by_end": bool(ev.query()),
        }

        # T_item_afterwait：AR 后先 w.wait()（测它挡不挡 CPU），再 sum → item。
        tl = TL()
        torch.cuda.synchronize()
        tl.m("t0")
        t = _fill(rank)
        tl.m("ar_launch")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        tl.m("wait_in")
        w.wait()
        tl.m("wait_out")
        wc = w.is_completed()
        sx = t.sum()
        tl.m("item_in")
        v = sx.item()
        tl.m("item_out")
        torch.cuda.synchronize()
        tl.m("sync_out")
        out["scen"]["T_item_afterwait"] = {
            "timeline": tl.lines(),
            "computed": v,
            "correct": bool(v == _EXP),
            "wait_ms": round(tl.dt("wait_in", "wait_out"), 4),
            "is_completed_after_wait": bool(wc),
            "item_block_ms": round(tl.dt("item_in", "item_out"), 4),
            "sync_after_item_ms": round(tl.dt("item_out", "sync_out"), 4),
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
