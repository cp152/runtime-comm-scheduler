"""验证：``dist.all_reduce`` 的 enqueue 是否 flush 调用方流（等输入生产完）。

v4 里 ``ev_a``（fill 后、all_reduce 后记录）8µs 触发，而 v6 证明 event 机制可靠、
all_reduce 不改 current stream——只剩一种解释：all_reduce 的 enqueue 内部把调用方
流同步了（fill 在返回前跑完）。若真如此，enqueue 墙钟 ≈ fill 时长（~3ms），而非
~µs 异步。

本 worker：
- **ctrl**：fill → event → poll（应 ~3ms，对照）。
- **enqueue_wall**：fill 后调 all_reduce，量 enqueue 墙钟。≈3ms → flush（阻塞）；
  ≈µs → 不阻塞（真异步，与 M4 producer_overlap 一致）。
- **ev_after_ar**：fill → all_reduce → event → poll。µs → fill 已被 flush；
  ~3ms → 未 flush。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29


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

    out: dict = {"rank": rank}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # 对照：fill → event（无 all_reduce）。
        torch.cuda.synchronize()
        t = torch.full((_N,), 1.0, device="cuda")
        ev = torch.cuda.Event()
        ev.record()
        out["ctrl_event_after_fill_s"] = round(_poll(ev), 6)
        torch.cuda.synchronize()

        # enqueue 墙钟：fill 后调 all_reduce，量返回耗时。
        torch.cuda.synchronize()
        t = torch.full((_N,), 1.0, device="cuda")
        tw0 = time.perf_counter()
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        out["enqueue_wall_s"] = round(time.perf_counter() - tw0, 6)
        ev = torch.cuda.Event()
        ev.record()
        out["ev_after_ar_s"] = round(_poll(ev), 6)
        torch.cuda.synchronize()

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
