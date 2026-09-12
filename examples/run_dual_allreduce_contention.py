"""Run and summarize the two-GPU, dual-communicator NCCL benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "dual_allreduce_contention_worker.py"
_MODES = ("single_a", "single_b", "sequential", "concurrent")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_rank(rank: int, args, port: int):
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES=str(rank),
    )
    # A blocking WorkNCCL wait would serialize the host launches and invalidate
    # the concurrent case. The benchmark always qualifies default wait mode.
    env.pop("TORCH_NCCL_BLOCKING_WAIT", None)
    command = [
        sys.executable,
        str(_WORKER),
        "--sizes-mib",
        args.sizes_mib,
        "--rounds",
        str(args.rounds),
        "--warmup-rounds",
        str(args.warmup_rounds),
    ]
    return subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _global_round(rank_results: list[dict], size_index: int, round_index: int) -> dict:
    combined = {}
    for mode in _MODES:
        rank_trials = [
            result["sizes"][size_index]["rounds"][round_index][mode]
            for result in rank_results
        ]
        combined[mode] = {
            "makespan_ms": max(trial["makespan_ms"] for trial in rank_trials),
            "a_completion_ms": max(
                trial["a_completion_ms"]
                for trial in rank_trials
                if trial["a_completion_ms"] is not None
            )
            if mode != "single_b"
            else None,
            "b_completion_ms": max(
                trial["b_completion_ms"]
                for trial in rank_trials
                if trial["b_completion_ms"] is not None
            )
            if mode != "single_a"
            else None,
            "correct": all(trial["correct"] for trial in rank_trials),
        }
    return combined


def _summarize(rank_results: list[dict]) -> list[dict]:
    summaries = []
    for size_index, size_result in enumerate(rank_results[0]["sizes"]):
        global_rounds = [
            _global_round(rank_results, size_index, round_index)
            for round_index in range(len(size_result["rounds"]))
        ]
        medians = {
            mode: statistics.median(
                round_result[mode]["makespan_ms"] for round_result in global_rounds
            )
            for mode in _MODES
        }
        isolated_sums = [
            round_result["single_a"]["makespan_ms"]
            + round_result["single_b"]["makespan_ms"]
            for round_result in global_rounds
        ]
        concurrent_ratios = [
            round_result["concurrent"]["makespan_ms"] / isolated_sum
            for round_result, isolated_sum in zip(global_rounds, isolated_sums)
        ]
        sequential_ratios = [
            round_result["sequential"]["makespan_ms"] / isolated_sum
            for round_result, isolated_sum in zip(global_rounds, isolated_sums)
        ]
        ratio = statistics.median(concurrent_ratios)
        if ratio > 1.05:
            verdict = "superadditive_contention"
        elif ratio < 0.95:
            verdict = "useful_overlap"
        else:
            verdict = "approximately_additive"
        concurrent_a = statistics.median(
            round_result["concurrent"]["a_completion_ms"]
            for round_result in global_rounds
        )
        concurrent_b = statistics.median(
            round_result["concurrent"]["b_completion_ms"]
            for round_result in global_rounds
        )
        summary = {
            "size_mib_per_allreduce": size_result["size_mib"],
            "rounds": len(global_rounds),
            "all_correct": all(
                trial[mode]["correct"]
                for trial in global_rounds
                for mode in _MODES
            ),
            "median_ms": {
                **{mode: round(value, 3) for mode, value in medians.items()},
                "isolated_sum": round(
                    statistics.median(isolated_sums), 3
                ),
                "concurrent_a_completion": round(concurrent_a, 3),
                "concurrent_b_completion": round(concurrent_b, 3),
            },
            "concurrent_over_isolated_sum": {
                "p10": round(_percentile(concurrent_ratios, 0.10), 4),
                "median": round(ratio, 4),
                "p90": round(_percentile(concurrent_ratios, 0.90), 4),
            },
            "sequential_over_isolated_sum_median": round(
                statistics.median(sequential_ratios), 4
            ),
            "concurrent_over_sequential_median": round(
                statistics.median(
                    round_result["concurrent"]["makespan_ms"]
                    / round_result["sequential"]["makespan_ms"]
                    for round_result in global_rounds
                ),
                4,
            ),
            "individual_slowdown_under_concurrency": {
                "a": round(concurrent_a / medians["single_a"], 4),
                "b": round(concurrent_b / medians["single_b"], 4),
            },
            "verdict": verdict,
        }
        summaries.append(summary)
    return summaries


def _print_table(summaries: list[dict]) -> None:
    print(
        "MiB   singleA   singleB   isolated_sum   sequential   concurrent   "
        "conc/sum       conc/seq   verdict"
    )
    for item in summaries:
        med = item["median_ms"]
        ratio = item["concurrent_over_isolated_sum"]
        print(
            f"{item['size_mib_per_allreduce']:>3} "
            f"{med['single_a']:>9.3f} "
            f"{med['single_b']:>9.3f} "
            f"{med['isolated_sum']:>14.3f} "
            f"{med['sequential']:>12.3f} "
            f"{med['concurrent']:>12.3f} "
            f"{ratio['median']:>7.3f} "
            f"[{ratio['p10']:.3f},{ratio['p90']:.3f}] "
            f"{item['concurrent_over_sequential_median']:>9.3f}   "
            f"{item['verdict']}"
        )


def _collect_rank(rank: int, process: subprocess.Popen, timeout: float):
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        _stdout, stderr = process.communicate()
        return None, f"rank {rank} timed out: {stderr[-2000:]}"
    if process.returncode != 0:
        return None, f"rank {rank} exited {process.returncode}: {stderr[-4000:]}"
    try:
        return json.loads(stdout.strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as exc:
        return None, f"rank {rank} bad output ({exc}): {stdout[-2000:]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", default="1,16,64,256")
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    port = _free_port()
    processes = [_run_rank(rank, args, port) for rank in range(2)]
    rank_results = []
    errors = []
    # Both ranks can emit more than a pipe buffer of raw samples. Drain their
    # pipes concurrently so one rank cannot block in write() while its peer is
    # waiting for it during ProcessGroup teardown.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_collect_rank, rank, process, args.timeout)
            for rank, process in enumerate(processes)
        ]
        for future in futures:
            result, error = future.result()
            if error is not None:
                errors.append(error)
            else:
                rank_results.append(result)

    if errors or len(rank_results) != 2:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    rank_results.sort(key=lambda result: result["rank"])
    summaries = _summarize(rank_results)
    _print_table(summaries)
    payload = {
        "environment": {
            key: rank_results[0][key]
            for key in ("device", "torch", "cuda", "nccl", "blocking_wait_env")
        },
        "summaries": summaries,
        "rank_results": rank_results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if all(item["all_correct"] for item in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
