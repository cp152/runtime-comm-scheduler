"""Two-rank WorkNCCL physical-completion capability probe.

This is an experiment harness, not scheduler production code.  A CUDA sleep is
queued ahead of a real all-reduce so ``Work.is_completed()`` has a stable
interval in which it must remain false.  Consumer waiting is issued on an
explicit stream to distinguish stream dependency insertion from host blocking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

_N = 4 * 1024 * 1024  # 16 MiB per rank
_SLEEP_CYCLES = 500_000_000


def _launch_delayed_all_reduce(rank: int):
    tensor = torch.full((_N,), float(rank + 1), device="cuda")
    torch.cuda.current_stream().synchronize()
    launch_stream = torch.cuda.Stream()
    with torch.cuda.stream(launch_stream):
        torch.cuda._sleep(_SLEEP_CYCLES)
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    return tensor, work, launch_stream


def _safe_completed(work):
    try:
        return {"value": bool(work.is_completed()), "error": None}
    except BaseException as exc:  # noqa: BLE001 - capability result
        return {"value": None, "error": f"{type(exc).__name__}: {exc}"}


def _run_default(rank: int) -> dict:
    tensor, work, launch_stream = _launch_delayed_all_reduce(rank)
    before = _safe_completed(work)
    consumer = torch.cuda.Stream()
    after_wait = torch.cuda.Event()
    started = time.perf_counter()
    with torch.cuda.stream(consumer):
        wait_result = bool(work.wait())
        after_wait.record()
    wait_ms = (time.perf_counter() - started) * 1e3
    after = _safe_completed(work)
    event_after_return = bool(after_wait.query())

    poll_started = time.perf_counter()
    poll_values = [bool(work.is_completed()) for _ in range(64)]
    poll_ms = (time.perf_counter() - poll_started) * 1e3

    consumer.synchronize()
    final = _safe_completed(work)
    correct = bool(torch.all(tensor == 3.0).item())
    return {
        "before_wait": before,
        "wait_result": wait_result,
        "wait_host_ms": round(wait_ms, 3),
        "after_wait": after,
        "consumer_event_after_return": event_after_return,
        "poll_count": len(poll_values),
        "poll_host_ms": round(poll_ms, 3),
        "poll_all_false": not any(poll_values),
        "consumer_event_after_stream_sync": bool(after_wait.query()),
        "final": final,
        "correct": correct,
        "launch_stream": str(launch_stream),
    }


def _run_blocking(rank: int) -> dict:
    tensor, work, launch_stream = _launch_delayed_all_reduce(rank)
    before = _safe_completed(work)
    started = time.perf_counter()
    wait_result = bool(work.wait())
    wait_ms = (time.perf_counter() - started) * 1e3
    after = _safe_completed(work)
    correct = bool(torch.all(tensor == 3.0).item())
    return {
        "before_wait": before,
        "wait_result": wait_result,
        "wait_host_ms": round(wait_ms, 3),
        "after_wait": after,
        "correct": correct,
        "launch_stream": str(launch_stream),
    }


def _run_finite_timeout(rank: int) -> dict:
    _tensor, work, launch_stream = _launch_delayed_all_reduce(rank)
    before = _safe_completed(work)
    consumer = torch.cuda.Stream()
    started = time.perf_counter()
    result = None
    error = None
    try:
        with torch.cuda.stream(consumer):
            result = bool(work.wait(timeout=timedelta(milliseconds=10)))
    except BaseException as exc:  # noqa: BLE001 - behavior is the measurement
        error = f"{type(exc).__name__}: {exc}"
    wait_ms = (time.perf_counter() - started) * 1e3
    after = _safe_completed(work)
    if error is None:
        consumer.synchronize()
    return {
        "before_wait": before,
        "wait_result": result,
        "wait_error": error,
        "wait_host_ms": round(wait_ms, 3),
        "after_wait": after,
        "launch_stream": str(launch_stream),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("default", "blocking", "finite_timeout"), required=True
    )
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    warmup = torch.ones(1024, device="cuda")
    dist.all_reduce(warmup, async_op=True).wait()

    if args.scenario == "default":
        measurement = _run_default(rank)
    elif args.scenario == "blocking":
        measurement = _run_blocking(rank)
    else:
        measurement = _run_finite_timeout(rank)
    print(
        json.dumps(
            {
                "rank": rank,
                "scenario": args.scenario,
                "blocking_wait_env": os.environ.get("TORCH_NCCL_BLOCKING_WAIT"),
                **measurement,
            }
        ),
        flush=True,
    )
    if args.scenario == "finite_timeout":
        # A finite WorkNCCL timeout aborts the communicator in the qualified
        # PyTorch build.  Normal ProcessGroup teardown can then hang in the
        # watchdog path, so this disposable probe process exits without trying
        # to recover or reuse that communicator.
        os._exit(0)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
