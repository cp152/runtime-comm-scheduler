"""Run one rank of the unscheduled Phase 1 multi-job replay baseline."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _load_workload(path: Path) -> dict:
    workload = json.loads(path.read_text(encoding="utf-8"))
    if workload.get("schema_version") != 1:
        raise ValueError("unsupported workload schema_version")
    jobs = workload.get("jobs")
    if not jobs:
        raise ValueError("workload must contain at least one job")
    return workload


def _validate_job(job: dict, rank: int) -> None:
    required = {"job_id", "process_group_id", "ranks", "tasks"}
    missing = required.difference(job)
    if missing:
        raise ValueError(f"job is missing fields: {sorted(missing)}")
    if rank not in job["ranks"]:
        raise ValueError(f"rank {rank} is not a member of {job['job_id']}")
    task_ids = set()
    for task in job["tasks"]:
        task_id = task.get("task_id")
        if not task_id or task_id in task_ids:
            raise ValueError(f"duplicate or missing task_id in {job['job_id']}")
        task_ids.add(task_id)
        kind = task.get("kind")
        if kind == "compute":
            if float(task.get("duration_ms", -1)) < 0:
                raise ValueError(f"compute task {task_id} has invalid duration_ms")
        elif kind == "collective":
            if task.get("op") != "all_reduce":
                raise ValueError(f"unsupported collective in task {task_id}")
            if int(task.get("num_bytes", 0)) <= 0:
                raise ValueError(f"collective task {task_id} has invalid num_bytes")
        else:
            raise ValueError(f"unsupported task kind {kind!r}")


def _run_job(job: dict, rank: int, group, results: dict[str, dict]) -> None:
    stream = torch.cuda.Stream()
    events = []
    task_trace = []
    started = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        for task in job["tasks"]:
            task_started = time.perf_counter_ns()
            if task["kind"] == "compute":
                time.sleep(float(task["duration_ms"]) / 1000.0)
                task_completed = time.perf_counter_ns()
            else:
                elements = int(task["num_bytes"]) // torch.tensor([], dtype=torch.float32).element_size()
                tensor = torch.full((elements,), float(rank + 1), device="cuda")
                work = dist.all_reduce(tensor, group=group, async_op=True)
                if not work.wait():
                    raise RuntimeError(f"collective wait failed for {task['task_id']}")
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)
                events.append(event)
                task_completed = time.perf_counter_ns()
                task_trace.append(
                    {
                        "task_id": task["task_id"],
                        "kind": task["kind"],
                        "launch_ts_ns": task_started,
                        "submit_ts_ns": task_completed,
                        "num_bytes": task["num_bytes"],
                    }
                )
                continue
            task_trace.append(
                {
                    "task_id": task["task_id"],
                    "kind": task["kind"],
                    "start_ts_ns": task_started,
                    "complete_ts_ns": task_completed,
                }
            )
    for event in events:
        event.synchronize()
    completed = time.perf_counter_ns()
    results[job["job_id"]] = {
        "job_id": job["job_id"],
        "rank": rank,
        "start_ts_ns": started,
        "complete_ts_ns": completed,
        "makespan_ms": (completed - started) / 1_000_000,
        "tasks": task_trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    workload = _load_workload(args.workload)
    if workload["world_size"] != world_size:
        raise ValueError("workload world_size does not match process group")
    for job in workload["jobs"]:
        _validate_job(job, rank)

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("gloo")
    groups = {
        job["process_group_id"]: dist.new_group(job["ranks"], backend="nccl")
        for job in workload["jobs"]
    }
    results: dict[str, dict] = {}
    threads = [
        threading.Thread(
            target=_run_job,
            args=(job, rank, groups[job["process_group_id"]], results),
            name=job["job_id"],
        )
        for job in workload["jobs"]
    ]
    try:
        dist.barrier()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        dist.barrier()
        args.output.write_text(
            json.dumps(
                {
                    "workload_id": workload["workload_id"],
                    "rank": rank,
                    "jobs": [results[job["job_id"]] for job in workload["jobs"]],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())