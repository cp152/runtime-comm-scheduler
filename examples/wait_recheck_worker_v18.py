"""v18（profiler 探针）：首条 sum 的 ~500ms host 阻塞内部到底调了什么？

v17 用 sync_debug_mode(warn) 未在首条 sum 阻塞区间内抓到任何同步 warning →
排除了 torch「named sync」与 allocator 显式同步（后者 debug 模式会报）。
本 worker 用 torch.profiler(CPU+CUDA) 只包住首条 no-wait sum + item，
看 500ms self-time 落在哪个 op / 有没有 cudaMalloc / cudaFree /
cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize 事件。

结构复刻 v14：warmup 大 AR(waited) 后 rep0 = 首条 no-wait sum（预期阻塞读对）。
profiler 表打到 stderr（driver 用 run_wait_recheck_warn.py 透传），stdout 只有
最后一行 JSON。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

_N = 1 << 29  # 2 GiB float32
_EXP = 3 * _N  # 两 rank fill 1.0/2.0，SUM 后 3.0
_PROC_T0 = time.perf_counter()


def _log(tag: str) -> None:
    sys.stderr.write(f"[{(time.perf_counter() - _PROC_T0) * 1e3:10.2f} ms] {tag}\n")
    sys.stderr.flush()


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _emit_profiler(prof) -> None:
    """把 profiler 关键信息打到 stderr。"""
    err = sys.stderr
    w = err.write

    w("\n===== key_averages by self_cpu_time_total (top 60) =====\n")
    try:
        w(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60) + "\n")
    except Exception as e:  # noqa: BLE001
        w(f"  (key_averages failed: {e})\n")

    # 只看可能相关的 CPU 侧事件（含名字匹配运行时/分配/同步原语）
    keyw = (
        "Malloc", "Free", "Synchron", "Memcpy", "EventQuery", "EventRecord",
        "sum", "Sum", "reduce", "Reduce", "item", "copy_", "select", "empty",
    )
    w("\n===== 相关事件（name 匹配同步/分配/归约） ======")
    w("name | self_cpu_total_us | count | device_time_total_us\n")
    for e in prof.events():
        nm = e.name or ""
        if any(k in nm for k in keyw) and (e.self_cpu_time_total > 1 or e.device_time_total > 1):
            w(f"{nm} | {e.self_cpu_time_total:.1f} | {e.count} | {e.device_time_total:.1f}\n")
    w("===== end events =====\n")
    err.flush()


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    _log(f"rank{rank}: start")
    dist.init_process_group(backend="nccl")
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()
    _log("init + 小 AR done")

    out: dict = {"rank": rank, "world": world, "expected": _EXP, "profiled_rep0": {}}
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        # v14 preamble：确保 rep0 是「首条 sum + AR 在途」状态。
        torch.cuda.synchronize()
        t = _fill(rank)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize()
        _log("warmup 大 AR (waited) done")

        # rep0：首条 no-wait sum，包 profiler
        torch.cuda.synchronize()
        t = _fill(rank)
        w = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

        _log("profiler region IN (sum+item)")
        t_prof = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            sx = t.sum()
            v = sx.item()
        wall = (time.perf_counter() - t_prof) * 1e3
        _log(f"profiler region OUT wall_ms={wall:.2f}")
        _emit_profiler(prof)

        tc = time.perf_counter()
        torch.cuda.synchronize()
        sync_after = (time.perf_counter() - tc) * 1e3

        out["profiled_rep0"] = {
            "wall_ms": round(wall, 2),
            "computed": v,
            "correct": bool(v == _EXP),
            "sync_after_item_ms": round(sync_after, 4),
        }
        del t
        del sx

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
