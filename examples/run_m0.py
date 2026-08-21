"""M0 driver：对每个场景启动两 rank Gloo 子进程，超时控制并汇总时序。

用法：:

    python run_m0.py                # 跑全部场景
    python run_m0.py --only fifo    # 只跑一个场景

对挂起场景（op_mismatch / missing_collective）用短超时杀掉，证明错误场景
有界退出，而非无限静默挂起。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "m0_worker.py")

# 场景 -> (超时秒数, 是否预期挂起)
SCENARIOS = {
    "fifo": (20, False),
    "fixed_reorder": (20, False),
    "identical_reorder_mismatch": (20, False),
    "op_mismatch": (6, True),
    "missing_collective": (6, True),
    "delayed_ready": (20, False),
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_rank(rank: int, world: int, scenario: str, port: int, delay: float):
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    cmd = [sys.executable, _WORKER, "--scenario", scenario, "--delay", str(delay)]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def run_scenario(scenario: str, delay: float) -> dict:
    timeout, expect_hang = SCENARIOS[scenario]
    port = _free_port()
    procs = [_run_rank(r, 2, scenario, port, delay) for r in range(2)]
    results = []
    timed_out = False
    for p in procs:
        try:
            out, err = p.communicate(timeout=timeout)
            if p.returncode != 0:
                results.append({"rank": None, "status": "crashed", "stderr": err[-500:]})
            else:
                try:
                    results.append(json.loads(out.strip().splitlines()[-1]))
                except json.JSONDecodeError:
                    results.append({"rank": None, "status": "bad_output", "out": out[-500:]})
        except subprocess.TimeoutExpired:
            timed_out = True
            p.kill()
            p.communicate()
    for p in procs:
        if p.poll() is None:
            p.kill()
    return {
        "scenario": scenario,
        "timeout_s": timeout,
        "expect_hang": expect_hang,
        "timed_out": timed_out,
        "results": results,
    }


def _summarize(r: dict) -> str:
    if r["timed_out"]:
        verdict = "TIMED_OUT (diverged/hung -> bounded exit via kill)"
    else:
        statuses = {x.get("status") for x in r["results"]}
        verdict = "COMPLETED" if statuses == {"ok"} else f"status={statuses}"
    lines = [f"  {r['scenario']:26s} -> {verdict}"]
    for x in r["results"]:
        if x.get("status") != "ok":
            lines.append(f"      rank={x.get('rank')} {x}")
            continue
        rank = x["rank"]
        for op in x["ops"]:
            ts = {e["name"]: e["ts"] for e in op["events"]}
            submit_to_enq = (ts["enqueue"] - ts["submit"]) * 1e6
            enq_to_done = (ts["complete"] - ts["enqueue"]) * 1e6
            lines.append(
                f"      r{rank} {op['op']}#{op['ordinal']} ok={op['ok']} "
                f"enqueue={submit_to_enq:.0f}us wait={enq_to_done:.0f}us"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single scenario name")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="delay for delayed_ready (seconds)")
    args = ap.parse_args()

    names = [args.only] if args.only else list(SCENARIOS)
    print(f"worker: {_WORKER}\npython: {sys.executable}\n")
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario {name!r}; choices={list(SCENARIOS)}")
            return 2
        r = run_scenario(name, args.delay)
        print(_summarize(r))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
