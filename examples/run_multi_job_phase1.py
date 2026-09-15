"""Launch and summarize the unscheduled Phase 1 multi-job baseline."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKER = HERE / "multi_job_phase1_worker.py"
DEFAULT_WORKLOAD = HERE / "workloads" / "multi_job_phase1.json"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _launch(rank: int, world_size: int, workload: Path, output: Path, port: int):
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    return subprocess.Popen(
        [
            sys.executable,
            str(WORKER),
            "--workload",
            str(workload),
            "--output",
            str(output),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    world_size = int(workload["world_size"])
    with tempfile.TemporaryDirectory(prefix="phase1-replay-") as temp_dir:
        temp_path = Path(temp_dir)
        port = _free_port()
        processes = [
            _launch(rank, world_size, args.workload, temp_path / f"rank-{rank}.json", port)
            for rank in range(world_size)
        ]
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=world_size) as pool:
            futures = [pool.submit(process.communicate, args.timeout) for process in processes]
            for rank, (process, future) in enumerate(zip(processes, futures)):
                try:
                    _stdout, stderr = future.result()
                except subprocess.TimeoutExpired:
                    process.kill()
                    _stdout, stderr = process.communicate()
                    errors.append(f"rank {rank} timed out: {stderr[-1000:]}")
                if process.returncode != 0:
                    errors.append(f"rank {rank} exited {process.returncode}: {stderr[-1000:]}")
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1

        rank_results = [
            json.loads((temp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
            for rank in range(world_size)
        ]
        summary = {
            "workload_id": workload["workload_id"],
            "world_size": world_size,
            "jobs": {
                job_id: {
                    "makespan_ms": max(
                        result["jobs"][index]["makespan_ms"]
                        for result in rank_results
                    ),
                    "ranks": [result["jobs"][index] for result in rank_results],
                }
                for index, job_id in enumerate(
                    job["job_id"] for job in workload["jobs"]
                )
            },
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())