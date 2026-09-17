#!/usr/bin/env python3
"""
Convert a chain-based DAG workload JSON into a `Workload`-style JSON file.

Usage:
    python convert_workload.py <input_dir> <output_dir>

For every *.json in <input_dir>, produce <stem>.json in <output_dir>.
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Reference values extracted from the "balanced" workload example.
# ---------------------------------------------------------------------------
BALANCED_COMPUTE_SAMPLES = (
    # job-0 producer_compute_s
    0.005, 0.002, 0.001,
    # job-0 consumer_compute_s
    0.002, 0.001, 0.001,
    # job-1 producer_compute_s
    0.002, 0.004, 0.001,
    # job-1 consumer_compute_s
    0.001, 0.001, 0.002,
)
BALANCED_AVG_COMPUTE_S = sum(BALANCED_COMPUTE_SAMPLES) / len(BALANCED_COMPUTE_SAMPLES)

# "balanced" uses 4096 bytes <-> 0.001 s, i.e. 4_096_000 bytes / second.
BALANCED_COMM_BYTES = 4096
BALANCED_COMM_S = 0.001
BYTES_PER_SECOND = BALANCED_COMM_BYTES / BALANCED_COMM_S

DEFAULT_OP = "all_reduce"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Preprocessing: drop zero-duration nodes and reconnect their neighbours
# ---------------------------------------------------------------------------
def preprocess_zero_duration(data: dict) -> dict:
    """Remove tasks with duration == 0, reconnecting predecessors and successors.

    The input is guaranteed to be a set of chains, but the function works for
    any DAG: for each removed node, every successor inherits the removed node's
    dependencies.
    """
    tasks = {t["id"]: dict(t) for t in data["tasks"]}

    while True:
        zero_ids = [tid for tid, t in tasks.items() if t["duration"] == 0]
        if not zero_ids:
            break

        # Recompute successors from the current task set.
        succ = {tid: [] for tid in tasks}
        for t in tasks.values():
            for dep in t["dependencies"]:
                if dep in succ:
                    succ[dep].append(t["id"])

        for zid in zero_ids:
            if zid not in tasks:
                continue
            z = tasks[zid]
            preds = list(z["dependencies"])

            for sid in succ.get(zid, []):
                if sid not in tasks:
                    continue
                s = tasks[sid]
                # Remove dependency on the zero-duration node.
                s["dependencies"] = [d for d in s["dependencies"] if d != zid]
                # Add the zero-duration node's predecessors.
                for p in preds:
                    if p not in s["dependencies"]:
                        s["dependencies"].append(p)

            del tasks[zid]

    new_data = dict(data)
    new_data["tasks"] = list(tasks.values())
    return new_data


# ---------------------------------------------------------------------------
# Graph processing
# ---------------------------------------------------------------------------
def build_chains(data: dict):
    """Return a list of chains (each chain is a list of task dicts)."""
    tasks = {t["id"]: t for t in data["tasks"]}

    succ = {t["id"]: [] for t in data["tasks"]}
    for t in data["tasks"]:
        for dep in t["dependencies"]:
            succ[dep].append(t["id"])

    chains = []
    for t in data["tasks"]:
        if t["dependencies"]:
            continue  # not a root
        chain = []
        cur = t["id"]
        while True:
            chain.append(tasks[cur])
            if not succ[cur]:
                break
            cur = succ[cur][0]  # DAG is guaranteed to be a set of chains
        chains.append(chain)
    return chains


def merge_adjacent(chain):
    """Merge consecutive tasks of the same kind (durations are summed)."""
    merged = []
    for task in chain:
        if merged and merged[-1]["kind"] == task["kind"]:
            merged[-1]["duration"] += task["duration"]
        else:
            merged.append(dict(task))
    return merged


def compute_scale(merged_chains) -> float:
    """Global scaling factor for durations so the average compute time
    matches the average compute time of the `balanced` workload."""
    total, count = 0, 0
    for chain in merged_chains:
        for task in chain:
            if task["kind"] == "compute":
                total += task["duration"]
                count += 1
    if count == 0 or total == 0:
        return 1.0
    avg = total / count
    return BALANCED_AVG_COMPUTE_S / avg


# ---------------------------------------------------------------------------
# Conversion into CollectiveComm blocks
# ---------------------------------------------------------------------------
def chain_to_comm_blocks(chain, scale: float):
    """Turn one merged chain into a list of CollectiveComm dicts."""
    blocks = []
    for i, task in enumerate(chain):
        if task["kind"] != "communication":
            continue

        producer = chain[i - 1] if i > 0 and chain[i - 1]["kind"] == "compute" else None
        consumer = (
            chain[i + 1]
            if i + 1 < len(chain) and chain[i + 1]["kind"] == "compute"
            else None
        )

        producer_s = producer["duration"] * scale if producer else 0.0
        consumer_s = consumer["duration"] * scale if consumer else 0.0
        est_s = task["duration"] * scale

        num_bytes = max(1, round(est_s * BYTES_PER_SECOND / 4)) * 4

        blocks.append(
            {
                "id": len(blocks),
                "num_bytes": num_bytes,
                "op": DEFAULT_OP,
                "producer_compute_s": producer_s,
                "consumer_compute_s": consumer_s,
                "estimated_comm_s": est_s,
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------
def round_s(x: float, ndigits: int = 10) -> float:
    """Round seconds to a stable precision for JSON output."""
    return round(x, ndigits)


def build_output(data: dict, merged_chains, scale: float) -> dict:
    workload_name = data.get("id", "workload")
    seed = data.get("metadata", {}).get("seed")
    if seed is None:
        seed = DEFAULT_SEED

    jobs = []
    for job_idx, chain in enumerate(merged_chains):
        blocks = chain_to_comm_blocks(chain, scale)
        comms = []
        for blk in blocks:
            comms.append(
                {
                    "id": blk["id"],
                    "num_bytes": blk["num_bytes"],
                    "op": blk["op"],
                    "producer_compute_s": round_s(blk["producer_compute_s"]),
                    "consumer_compute_s": round_s(blk["consumer_compute_s"]),
                    "estimated_comm_s": round_s(blk["estimated_comm_s"]),
                }
            )
        jobs.append(
            {
                "job_id": f"job-{job_idx}",
                "ranks": None,
                "communications": comms,
            }
        )

    return {
        "name": workload_name,
        "seed": seed,
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <input_dir> <output_dir>")
        return 1

    input_dir = Path(argv[1])
    output_dir = Path(argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 1

    for path in json_files:
        data = load_json(path)
        data = preprocess_zero_duration(data)   # <-- 新增：删除 0 时长节点
        chains = build_chains(data)
        merged_chains = [merge_adjacent(c) for c in chains]
        scale = compute_scale(merged_chains)

        out_obj = build_output(data, merged_chains, scale)
        out_path = output_dir / f"{path.stem}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(out_obj, fh, indent=2)
            fh.write("\n")
        print(f"[ok] {path.name} -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))