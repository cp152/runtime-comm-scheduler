"""v3 悖论拆解：计算 op 在 ev_s≈3ms 执行却读到 reduce 后数据。

三个候选解释：
(a) 传输 ~3ms 完成、~277ms 是完成信号延迟（数据早已就绪）；
(b) 计算 op 实际被排序到 collective 之后执行，ev_s=3ms 测量有误；
(c) 其它（reordering / recordStream 机制）。

v4 用两组探测分离：
- **Test A**：AR1 后在 s 上排 ``ev_a``（AR1 发射点之后）→ ``sx = t.sum()`` →
  ``ev_b``（sum 之后）。若 sum 真被排序：ev_a≈3ms、ev_b≈280ms；若 sum 立即跑：
  ev_a≈ev_b≈3ms（则 3*N 只能来自 (a)）。
- **Test B**：AR1 后立刻在同一 process group 上发 AR2（1024，排到 comm stream 的
  AR1 之后），s 上记录 ev 轮询。AR2 的事件触发时刻反映 comm stream 何时被 AR1
  释放：ev≈3ms → AR1 数据 ~3ms 完、280ms 是信号延迟；ev≈280ms → AR1 真实占满
  comm stream（280ms 是真传输）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32；真值 = 3*_N


def _fill(rank: int, n: int = _N) -> torch.Tensor:
    return torch.full((n,), float(rank + 1), device="cuda")


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
        # throwaway 大 collective。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()

        # Test A：ev_a（AR 后）→ sum → ev_b（sum 后）。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        ev_a = torch.cuda.Event()
        ev_a.record()
        sx = t.sum()
        ev_b = torch.cuda.Event()
        ev_b.record()
        ev_a_s = _poll_until(ev_a)
        ev_b_s = _poll_until(ev_b)
        torch.cuda.synchronize()
        out["A"] = {
            "ev_a_s": round(ev_a_s, 6),
            "ev_b_s": round(ev_b_s, 6),
            "computed": sx.item(),
            "correct": bool(sx.item() == 3 * _N),
        }

        # Test B：AR1 后立刻发 AR2，探测 comm stream 何时空闲。
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)  # AR1
        p = _fill(rank, 1024)
        w2 = dist.all_reduce(p, op=dist.ReduceOp.SUM, async_op=True)  # AR2 排 AR1 后
        ev = torch.cuda.Event()
        ev.record()
        ev_ar2_s = _poll_until(ev)  # s 流排空时刻 ≈ comm stream 释放 AR1 后
        t_ref = time.perf_counter()
        torch.cuda.synchronize()
        sync_after_ar2_s = time.perf_counter() - t_ref
        out["B"] = {
            "ev_ar2_s": round(ev_ar2_s, 6),
            "sync_after_ar2_s": round(sync_after_ar2_s, 6),
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
