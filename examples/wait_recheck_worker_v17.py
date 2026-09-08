"""v17（用户假设探针）：首条 sum 的 host 阻塞是否 = CUDA 分配器偶然同步？

疑点：v13 S_A / v12 S_ctrl_sum 首条 sum（仅 fill 在途）阻塞 ~15ms，
首条 sum + AR 在途阻塞 ~AR 墙钟；之后所有 sum ~50µs 纯异步。用户假设可能是
CUDA 内存分配触发的 allocator 阻塞。验证手段：进程开头
``torch.cuda.set_sync_debug_mode("warn")``——若 sum 内部真的触发了同步原语
（如 cudaMalloc/cudaFree 引起的 device sync、或 named sync op），warn 模式会在
stderr 打 warning。

结构复刻 v14：warmup 后 rep0 = 首条 no-wait sum（预期阻塞+读对）、
rep1 = 后续 sum（预期 50µs+读脏）。每条 sum/item 前后向 stderr 打单调 ms mark，
warning 与 mark 交错出现即可对齐「warning 是否落在 sum 阻塞区间内」。

stderr mark 用 ``sys.stderr`` 直写，driver 需透传 stderr（见 run_wait_recheck_warn.py）。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32
_EXP = 3 * _N  # 两 rank fill 1.0/2.0，SUM 后 3.0
_PROC_T0 = time.perf_counter()


def _log(tag: str) -> None:
    sys.stderr.write(f"[{(time.perf_counter() - _PROC_T0) * 1e3:10.2f} ms] {tag}\n")
    sys.stderr.flush()


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    # —— 用户要求：测试代码开头开启 sync debug warn ——
    torch.cuda.set_sync_debug_mode("warn")
    _log("sync_debug_mode=warn 已开启")

    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()
    _log("init + 小 AR done")

    out: dict = {"rank": rank, "world": world, "expected": _EXP, "reps": []}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # v14 preamble：确保 rep0 进入「首条 sum + AR 在途」状态。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()
        _log("warmup 大 AR (waited) done")

        for r in range(2):
            torch.cuda.synchronize()
            t = _fill(rank)
            w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

            _log(f"rep{r}: --- sum IN ---")
            ta = time.perf_counter()
            sx = t.sum()
            sum_cpu = (time.perf_counter() - ta) * 1e3
            _log(f"rep{r}: --- sum OUT (sum_cpu_ms={sum_cpu:.2f}) ---")

            tb = time.perf_counter()
            v = sx.item()
            item_ms = (time.perf_counter() - tb) * 1e3
            _log(f"rep{r}: --- item OUT (item_ms={item_ms:.2f}) ---")

            tc = time.perf_counter()
            torch.cuda.synchronize()
            sync_after = (time.perf_counter() - tc) * 1e3

            out["reps"].append(
                {
                    "rep": r,
                    "sum_cpu_ms": round(sum_cpu, 4),
                    "item_block_ms": round(item_ms, 4),
                    "sync_after_item_ms": round(sync_after, 4),
                    "computed": v,
                    "correct": bool(v == _EXP),
                }
            )
            del t
            del sx

    torch.cuda.set_sync_debug_mode("default")
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
