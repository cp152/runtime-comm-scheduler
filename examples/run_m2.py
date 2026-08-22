"""M2 driver：对每个场景启动两 rank Gloo 子进程并对比 sequence log。

用法：:

    python run_m2.py                # 跑全部场景
    python run_m2.py --only fifo    # 只跑一个场景

验收（phase1-plan M2）：FIFO 和安全的固定重排能完成，且所有 rank 的
sequence log 相同；错误场景（op_mismatch）以 fail-stop 有界退出，
不再像 M0 那样挂起。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "m2_worker.py")

# 场景 -> (超时秒数, 期望的 status)
SCENARIOS = {
    "fifo": (20, "ok"),
    "fixed_reorder": (20, "ok"),
    "out_of_order_submit": (20, "ok"),
    "op_mismatch": (20, "validation_error"),
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_rank(rank: int, world: int, scenario: str, port: int):
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    cmd = [sys.executable, _WORKER, "--scenario", scenario]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def run_scenario(scenario: str) -> dict:
    timeout, expect = SCENARIOS[scenario]
    port = _free_port()
    procs = [_run_rank(r, 2, scenario, port) for r in range(2)]
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
        "expect": expect,
        "timed_out": timed_out,
        "results": results,
    }


def _summarize(r: dict) -> str:
    results = r["results"]
    statuses = {x.get("status") for x in results}
    lines = [f"  {r['scenario']:20s} -> status={statuses} expect={r['expect']}"]
    for x in results:
        rank = x.get("rank")
        if x.get("status") != "ok":
            lines.append(f"      rank={rank} {x.get('error') or x}")
            continue
        seq = x["launched_seq"]
        seq_s = {g: [[str(y) for y in k] for k in keys] for g, keys in seq.items()}
        lines.append(f"      r{rank} launched_seq={seq_s}")
        for t in x["timings"]:
            lines.append(
                f"      r{rank} key={t['key'][-1]} ready={t['ready_us']} "
                f"admit={t['admit_us']} submit={t['submit_us']} "
                f"complete={t['complete_us']} actual={t['actual_us']}us"
            )
    if statuses == {r["expect"]} and r["expect"] == "ok":
        # 验收：ok 场景下所有 rank 的 sequence log 相同。
        seqs = {tuple(str(s) for s in x["launched_seq"].values()) for x in results}
        if len(seqs) == 1:
            lines.append("      ACCEPT: all ranks' sequence logs identical")
        else:
            lines.append("      REJECT: sequence logs diverge across ranks")
    elif statuses == {r["expect"]}:
        lines.append("      ACCEPT: all ranks failed consistently (bounded fail-stop)")
    else:
        lines.append("      REJECT: statuses do not match expectation")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single scenario name")
    args = ap.parse_args()

    names = [args.only] if args.only else list(SCENARIOS)
    print(f"worker: {_WORKER}\npython: {sys.executable}\n")
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario {name!r}; choices={list(SCENARIOS)}")
            return 2
        print(_summarize(run_scenario(name)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
