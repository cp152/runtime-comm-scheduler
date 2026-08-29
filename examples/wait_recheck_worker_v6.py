"""控制实验：本环境下 ``torch.cuda.Event`` 的 record/query 是否可靠？

v4 的破绽：``ev_a``（all_reduce 后、sum 前记录）在 8µs 触发，但它排在 3ms 的 fill
之后——若 event 正常记在流 s 上，不可能 8µs 触发。两种可能：(a) event 机制本身坏
（query 提前返回）；(b) all_reduce 改变了 current_stream，event 记到了别的流上。

本 worker 分开测：
- **A**：s 上排一个 ~3ms 的 fill，之后 record event 并轮询。event 正常 → ~3ms；
  event 坏 → µs 级提前返回。
- **B**：all_reduce 前后检查 ``torch.cuda.current_stream() == s``，看是否被改变。
- **C**：空流（无任何工作）上 record event，应立即触发（作对照）。
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
        # C：空流事件（对照，应即时触发）。
        ev_c = torch.cuda.Event()
        ev_c.record()
        out["C_empty_stream_event_s"] = round(_poll(ev_c), 6)

        # A：fill (~3ms) 后的 event，应 ~3ms 触发。
        torch.cuda.synchronize()
        t = torch.full((_N,), 1.0, device="cuda")
        ev_a = torch.cuda.Event()
        ev_a.record()
        out["A_event_after_fill_s"] = round(_poll(ev_a), 6)
        torch.cuda.synchronize()

        # B：all_reduce 是否改变 current_stream。
        torch.cuda.synchronize()
        out["B_cur_is_s_before"] = torch.cuda.current_stream() == s
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        out["B_cur_is_s_after"] = torch.cuda.current_stream() == s
        out["B_cur_stream_eq_before_after"] = (
            torch.cuda.current_stream() == s
        )

        # A2：all_reduce 后 record 的 event（无本地工作），看是否也提前返回。
        ev_a2 = torch.cuda.Event()
        ev_a2.record()
        out["A2_event_after_ar_s"] = round(_poll(ev_a2), 6)

        # 验证 current stream 是否为 s 的下层指针（防止 == 重载掩盖差异）。
        out["B_cur_stream_cudaid"] = torch.cuda.current_stream().cuda_stream
        out["B_s_cudaid"] = s.cuda_stream

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
