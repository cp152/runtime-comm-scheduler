"""M0 单 rank worker：在 Gloo backend 下执行一个 collective 场景并记录时序。

由 ``run_m0.py`` 驱动，通过 env:// 初始化两 rank 的 process group：:

    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 \
        python m0_worker.py --scenario fifo

每个 rank 向 stdout 打印单行 JSON，记录每个 collective 的
``submit -> enqueue -> complete`` 时间戳，供 driver 汇总。

M0 的目的不是训练 workflow，而是观察真实 PyTorch distributed runtime
在 deferred collective submission 下的行为：哪些顺序变化是安全的、哪些
会静默通过、哪些会挂起。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

# collective 使用的 tensor 大小（float32 元素数）
_N = 1024


def _now() -> float:
    return time.perf_counter()


def _one_all_reduce(rank: int, world: int, ordinal: int) -> dict:
    """执行一次 all_reduce 并返回其事件时序。"""
    tensor = torch.ones(_N, dtype=torch.float32) * (rank + 1)
    events = [{"name": "submit", "ts": _now()}]
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    events.append({"name": "enqueue", "ts": _now()})
    work.wait()
    events.append({"name": "complete", "ts": _now()})
    # all_reduce(SUM): 每个 rank 结果是所有 rank 贡献之和 sum_{r}(r+1)。
    expected = world * (world + 1) / 2
    ok = bool(torch.all(tensor == expected).item())
    return {"ordinal": ordinal, "op": "all_reduce", "ok": ok, "events": events}


def _one_all_gather(rank: int, world: int, ordinal: int) -> dict:
    """执行一次 all_gather 并返回其事件时序。"""
    tensor = torch.ones(_N, dtype=torch.float32) * (rank + 1)
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    events = [{"name": "submit", "ts": _now()}]
    work = dist.all_gather(gathered, tensor, async_op=True)
    events.append({"name": "enqueue", "ts": _now()})
    work.wait()
    events.append({"name": "complete", "ts": _now()})
    # all_gather: gathered[i] 应是 rank i 的 tensor，其元素值为 (i+1)。
    ok = all(bool(torch.all(g == (i + 1)).item()) for i, g in enumerate(gathered))
    return {"ordinal": ordinal, "op": "all_gather", "ok": ok, "events": events}


def run_scenario(rank: int, world: int, scenario: str, delay: float) -> list[dict]:
    if scenario == "fifo":
        # 两个 rank 都按 [A, B] 顺序提交 -> 完成
        return [_one_all_reduce(rank, world, 0), _one_all_reduce(rank, world, 1)]

    if scenario == "fixed_reorder":
        # 两个 rank 都按计划 [B, A]（相对自然顺序 A->B 的固定重排）提交 -> 完成
        return [_one_all_reduce(rank, world, 1), _one_all_reduce(rank, world, 0)]

    if scenario == "identical_reorder_mismatch":
        # rank0 [A, B]，rank1 [B, A]，但都是相同的 all_reduce ->
        # 运行时无法区分，静默完成（证明逻辑重排必须由 plan 层检测）
        if rank == 0:
            return [_one_all_reduce(rank, world, 0), _one_all_reduce(rank, world, 1)]
        return [_one_all_reduce(rank, world, 1), _one_all_reduce(rank, world, 0)]

    if scenario == "op_mismatch":
        # rank0 [all_reduce, all_gather]，rank1 [all_gather, all_reduce]
        # -> 集合类型不兼容，两个 rank 都挂起
        if rank == 0:
            return [_one_all_reduce(rank, world, 0), _one_all_gather(rank, world, 1)]
        return [_one_all_gather(rank, world, 0), _one_all_reduce(rank, world, 1)]

    if scenario == "missing_collective":
        # rank0 提交 [A, B]，rank1 只提交 [A] -> A 完成，rank0 阻塞在 B
        if rank == 0:
            return [_one_all_reduce(rank, world, 0), _one_all_reduce(rank, world, 1)]
        return [_one_all_reduce(rank, world, 0)]

    if scenario == "delayed_ready":
        # rank1 延迟 producer ready，观察 rank0 的 all_reduce 等待时长
        if rank == 1:
            time.sleep(delay)
        return [_one_all_reduce(rank, world, 0)]

    raise ValueError(f"unknown scenario: {scenario!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    out: dict = {"rank": rank, "scenario": args.scenario, "status": "ok", "ops": []}
    try:
        out["ops"] = run_scenario(rank, world, args.scenario, args.delay)
    except Exception as exc:  # noqa: BLE001 - 记录任何异常后仍打印结果
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        dist.destroy_process_group()
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
