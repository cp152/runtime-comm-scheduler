"""决定性吞吐测试：第一个大 collective 是否含一次性初始化（~280ms），
后续 2GiB all_reduce 的稳态真实时长是多少？

v8 显示有 throwaway 时 ``sync_after_s=41µs``（device 3.2ms 就空闲），而 v2 无
throwaway 时 ``sync_total_s≈280ms``。若「280ms 只是一次性 init」成立：
- first_sync ≈ 280ms（首个大 collective 的完整时长）
- steady_sync[i] ≈ ms 级（稳态）
若稳态仍是 ~280ms，则 v8 的 41µs 才是异常，collective 真实传输就是 280ms。

用 ``torch.cuda.synchronize()`` 墙钟（唯一可靠完成信号，v5 已证空闲时 µs 级）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # 首个大 collective（1024 warmup 之后），无 throwaway。
        torch.cuda.synchronize()
        t = torch.full((_N,), 1.0, device="cuda")
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        ts0 = time.perf_counter()
        torch.cuda.synchronize()
        out["first_sync_s"] = round(time.perf_counter() - ts0, 6)

        # 稳态：连续 3 个 2GiB all_reduce，各自 sync 计时。
        steady = []
        for i in range(3):
            torch.cuda.synchronize()
            t = torch.full((_N,), float(rank + 1), device="cuda")
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
            ts0 = time.perf_counter()
            torch.cuda.synchronize()
            steady.append(round(time.perf_counter() - ts0, 6))
        out["steady_sync_s"] = steady

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
