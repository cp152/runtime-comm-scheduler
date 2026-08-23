"""M4 决策关口 driver：对每个场景启动两 rank NCCL 子进程（各绑一张 3090）。

用法：:

    python run_m4_gate.py                 # 跑全部关口场景
    python run_m4_gate.py --only worker_basic

每个场景两 rank 各跑一张 GPU（``CUDA_VISIBLE_DEVICES=<rank>``），有超时
兜底（NCCL divergence 表现为无限挂起）。汇总后给出关口结论：worker 线程
提交 ProcessGroupNCCL 是否足以支撑异步 admission worker prototype。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "m4_gate_worker.py")

SCENARIOS = {
    "worker_basic": (60, "basic 非主线程提交"),
    "producer_continues": (90, "producer 入队即返回"),
    "concurrent_submit": (120, "双线程并发提交（信息性）"),
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
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES=str(rank),
    )
    cmd = [sys.executable, _WORKER, "--scenario", scenario]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def run_scenario(scenario: str, timeout: int) -> dict:
    port = _free_port()
    procs = [_run_rank(r, 2, scenario, port) for r in range(2)]
    results = []
    timed_out = False
    for p in procs:
        try:
            out, err = p.communicate(timeout=timeout)
            if p.returncode != 0:
                results.append({"status": "crashed", "stderr": err[-300:]})
            else:
                try:
                    results.append(json.loads(out.strip().splitlines()[-1]))
                except json.JSONDecodeError:
                    results.append({"status": "bad_output", "out": out[-300:]})
        except subprocess.TimeoutExpired:
            timed_out = True
            p.kill()
            p.communicate()
    for p in procs:
        if p.poll() is None:
            p.kill()
    return {"scenario": scenario, "timed_out": timed_out, "results": results}


def _summarize(r: dict) -> str:
    results = r["results"]
    statuses = {x.get("status") for x in results}
    lines = [f"  {r['scenario']:20s} -> status={statuses}"
             + ("  TIMED_OUT" if r["timed_out"] else "")]
    oks: set[bool] = set()
    for x in results:
        rank = x.get("rank")
        if x.get("status") != "ok":
            lines.append(f"      rank={rank} {x.get('error') or x}")
            continue
        oks.add(x["ok"])
        bits = [f"ok={x['ok']}"]
        for k in ("enqueue_s", "worker_total_s", "producer_compute_s",
                  "overlap_with_worker", "worker_busy_at_compute_start",
                  "total_span_s"):
            if x.get(k) is not None:
                bits.append(f"{k}={x[k]}")
        lines.append(f"      r{rank} " + " ".join(bits))
    return "\n".join(lines)


def _gate_verdict(r: dict) -> str:
    """决策关口：worker 线程提交是否可靠、producer 是否不被阻塞。"""
    if r["scenario"] == "concurrent_submit":
        note = "信息性场景（M4 设计中只有 worker 提交）；仅记录可靠性包络"
        if r["timed_out"] or any(x.get("status") != "ok" for x in r["results"]):
            return f"      GATE-NOTE: 双线程并发提交不可靠/超时。{note}"
        return f"      GATE-NOTE: 双线程并发提交通过。{note}"
    if r["timed_out"]:
        return "      GATE-REJECT: 场景超时（挂起）"
    if any(x.get("status") != "ok" or not x.get("ok") for x in r["results"]):
        return "      GATE-REJECT: 两 rank 存在失败"
    if r["scenario"] == "producer_continues":
        data = [x for x in r["results"] if x.get("status") == "ok"]
        enq = [x["enqueue_s"] for x in data]
        worker = [x["worker_total_s"] for x in data]
        overlap = all(x["overlap_with_worker"] for x in data)
        # 入队耗时应远小于 worker 的 collective 总耗时，且 GPU 计算与
        # collective 时间上重叠。
        if all(e < w * 0.1 for e, w in zip(enq, worker)) and overlap:
            return (f"      GATE-ACCEPT: producer 不被阻塞 "
                    f"(enqueue={enq}s << worker={worker}s, overlap={overlap})")
        return (f"      GATE-REJECT: enqueue={enq}s worker={worker}s "
                f"overlap={overlap}")
    return "      GATE-ACCEPT: worker 线程提交/等待可靠，数据正确"


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
        timeout, label = SCENARIOS[name]
        r = run_scenario(name, timeout)
        print(_summarize(r))
        print(_gate_verdict(r))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
