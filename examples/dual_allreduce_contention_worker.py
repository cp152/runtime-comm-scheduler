"""Per-rank worker for the dual-communicator NCCL contention benchmark.

The default process group is Gloo and is used only for control barriers.  Two
independent NCCL process groups contain the same two ranks.  Their all-reduces
are launched from different CUDA streams so the concurrent case can overlap on
the device while preserving the same host launch order on both ranks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

_MODES = ("single_a", "single_b", "sequential", "concurrent")


def _rotate_modes(round_index: int) -> tuple[str, ...]:
    offset = round_index % len(_MODES)
    return _MODES[offset:] + _MODES[:offset]


def _record_done(work, stream: torch.cuda.Stream) -> torch.cuda.Event:
    done = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        # WorkNCCL.wait installs the NCCL completion dependency into stream.
        if not work.wait():
            raise RuntimeError("WorkNCCL.wait() returned false")
        done.record()
    return done


def _run_trial(
    mode: str,
    rank: int,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    group_a,
    group_b,
    stream_a: torch.cuda.Stream,
    stream_b: torch.cuda.Stream,
) -> dict[str, float | bool | None]:
    tensor_a.fill_(float(rank + 1))
    tensor_b.fill_(float(rank + 1))
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    start.record()
    stream_a.wait_event(start)
    stream_b.wait_event(start)

    done_a = None
    done_b = None
    if mode == "single_a":
        with torch.cuda.stream(stream_a):
            work_a = dist.all_reduce(tensor_a, group=group_a, async_op=True)
        done_a = _record_done(work_a, stream_a)
    elif mode == "single_b":
        with torch.cuda.stream(stream_b):
            work_b = dist.all_reduce(tensor_b, group=group_b, async_op=True)
        done_b = _record_done(work_b, stream_b)
    elif mode == "sequential":
        with torch.cuda.stream(stream_a):
            work_a = dist.all_reduce(tensor_a, group=group_a, async_op=True)
        done_a = _record_done(work_a, stream_a)

        # Make B's caller stream wait for A. ProcessGroupNCCL propagates this
        # dependency to group B's internal NCCL stream without blocking CPU.
        stream_b.wait_event(done_a)
        with torch.cuda.stream(stream_b):
            work_b = dist.all_reduce(tensor_b, group=group_b, async_op=True)
        done_b = _record_done(work_b, stream_b)
    elif mode == "concurrent":
        with torch.cuda.stream(stream_a):
            work_a = dist.all_reduce(tensor_a, group=group_a, async_op=True)
        with torch.cuda.stream(stream_b):
            work_b = dist.all_reduce(tensor_b, group=group_b, async_op=True)
        # Submit both communicators before installing either completion wait so
        # short collectives get the largest overlap window host launch permits.
        done_a = _record_done(work_a, stream_a)
        done_b = _record_done(work_b, stream_b)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown mode {mode!r}")

    if done_a is not None:
        done_a.synchronize()
    if done_b is not None:
        done_b.synchronize()

    a_ms = None if done_a is None else float(start.elapsed_time(done_a))
    b_ms = None if done_b is None else float(start.elapsed_time(done_b))
    makespan_ms = max(value for value in (a_ms, b_ms) if value is not None)
    expected = float((rank + 1) + (2 - rank))  # 3.0 for a two-rank run
    correct_a = done_a is None or bool(torch.all(tensor_a == expected).item())
    correct_b = done_b is None or bool(torch.all(tensor_b == expected).item())
    dist.barrier()
    return {
        "makespan_ms": makespan_ms,
        "a_completion_ms": a_ms,
        "b_completion_ms": b_ms,
        "correct": correct_a and correct_b,
    }


def _run_size(
    size_mib: int,
    rank: int,
    group_a,
    group_b,
    rounds: int,
    warmup_rounds: int,
) -> dict:
    elements = size_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    tensor_a = torch.empty(elements, dtype=torch.float32, device="cuda")
    tensor_b = torch.empty_like(tensor_a)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    for warmup_index in range(warmup_rounds):
        for mode in _rotate_modes(warmup_index):
            _run_trial(
                mode,
                rank,
                tensor_a,
                tensor_b,
                group_a,
                group_b,
                stream_a,
                stream_b,
            )

    measurements = []
    for round_index in range(rounds):
        trial_by_mode = {}
        for mode in _rotate_modes(round_index):
            trial_by_mode[mode] = _run_trial(
                mode,
                rank,
                tensor_a,
                tensor_b,
                group_a,
                group_b,
                stream_a,
                stream_b,
            )
        measurements.append(trial_by_mode)
    return {
        "size_mib": size_mib,
        "rounds": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", default="1,16,64,256")
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    args = parser.parse_args()
    sizes_mib = tuple(int(value) for value in args.sizes_mib.split(","))
    if not sizes_mib or any(value <= 0 for value in sizes_mib):
        raise ValueError("--sizes-mib must contain positive integers")
    if args.rounds <= 0 or args.warmup_rounds < 0:
        raise ValueError("round counts must be positive/non-negative")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("this benchmark requires exactly two ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    dist.init_process_group(backend="gloo")
    group_a = dist.new_group(ranks=[0, 1], backend="nccl")
    group_b = dist.new_group(ranks=[0, 1], backend="nccl")
    try:
        # Force communicator initialization before any measured trial.
        for group in (group_a, group_b):
            warmup = torch.ones(1024, device="cuda")
            dist.all_reduce(warmup, group=group, async_op=True).wait()
            torch.cuda.synchronize()
        dist.barrier()

        results = [
            _run_size(
                size_mib,
                rank,
                group_a,
                group_b,
                args.rounds,
                args.warmup_rounds,
            )
            for size_mib in sizes_mib
        ]
        print(
            json.dumps(
                {
                    "rank": rank,
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "blocking_wait_env": os.environ.get("TORCH_NCCL_BLOCKING_WAIT"),
                    "sizes": results,
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group(group_b)
        dist.destroy_process_group(group_a)
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
