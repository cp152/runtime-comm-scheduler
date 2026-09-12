"""Run the M4.5 WorkNCCL capability gate on two local GPUs."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "m45_completion_gate_worker.py"
_SCENARIOS = ("default", "blocking", "finite_timeout")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_rank(rank: int, scenario: str, port: int):
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES=str(rank),
    )
    if scenario == "blocking":
        env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    else:
        env.pop("TORCH_NCCL_BLOCKING_WAIT", None)
    return subprocess.Popen(
        [sys.executable, str(_WORKER), "--scenario", scenario],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run_scenario(scenario: str) -> dict:
    port = _free_port()
    processes = [_run_rank(rank, scenario, port) for rank in range(2)]
    results = []
    errors = []
    for rank, process in enumerate(processes):
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            errors.append(f"rank {rank} timed out: {stderr[-500:]}")
            continue
        if process.returncode != 0:
            errors.append(
                f"rank {rank} exited {process.returncode}: {stderr[-1000:]}"
            )
            continue
        try:
            results.append(json.loads(stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError) as exc:
            errors.append(f"rank {rank} bad output ({exc}): {stdout[-1000:]}")
    return {"scenario": scenario, "results": results, "errors": errors}


def _accepted(run: dict) -> tuple[bool, str]:
    if run["errors"] or len(run["results"]) != 2:
        return False, "; ".join(run["errors"])
    results = run["results"]
    if run["scenario"] == "default":
        accepted = all(
            item["before_wait"]["value"] is False
            and item["wait_result"]
            and item["after_wait"]["value"] is False
            and not item["consumer_event_after_return"]
            and item["poll_all_false"]
            and item["consumer_event_after_stream_sync"]
            and item["final"]["value"]
            and item["correct"]
            for item in results
        )
        return accepted, "stream-ordered wait and nonblocking completion polling"
    if run["scenario"] == "blocking":
        accepted = all(
            item["before_wait"]["value"] is False
            and item["wait_result"]
            and item["after_wait"]["value"]
            and item["correct"]
            for item in results
        )
        return accepted, "blocking wait reaches physical completion"
    accepted = all(
        item["before_wait"]["value"] is False
        and (item["wait_result"] is not None or item["wait_error"] is not None)
        for item in results
    )
    return accepted, "finite-timeout behavior recorded without communicator reuse"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="optional JSON result path")
    parser.add_argument("--only", choices=_SCENARIOS)
    args = parser.parse_args()
    scenarios = (args.only,) if args.only else _SCENARIOS
    runs = [_run_scenario(scenario) for scenario in scenarios]
    ok = True
    for run in runs:
        accepted, reason = _accepted(run)
        run["accepted"] = accepted
        run["acceptance_reason"] = reason
        ok = ok and accepted
        print(json.dumps(run, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(runs, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
