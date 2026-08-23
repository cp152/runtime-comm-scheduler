"""M4 driver：对每个场景启动两 rank NCCL 子进程（各绑一张 3090）并对比结果。

用法：:

    python run_m4.py                # 跑全部场景
    python run_m4.py --only fifo    # 只跑一个场景

每个场景两 rank 各跑一张 GPU（``CUDA_VISIBLE_DEVICES=<rank>``）。验收
（phase1-plan M4）：producer 提交 intent 后继续执行、consumer 通过
``ScheduledWork`` 等待；没有错误的 stream dependency 或 sequence divergence
（跨 rank 对比 ``launched_seq``）。``out_of_order_submit`` 额外要求乱序提交
经 worker 仍被强制为计划顺序。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "m4_worker.py")

# 场景 -> (超时秒数, 期望的 status)
SCENARIOS = {
    "fifo": (60, "ok"),
    "fixed_reorder": (60, "ok"),
    "out_of_order_submit": (60, "ok"),
    "delayed_ready": (60, "ok"),
    "no_wait_ready": (60, "ok"),          # status 仍为 ok，op.ok 预期 False（竞争）
    "producer_overlap": (60, "ok"),
    "op_mismatch": (60, "validation_error"),
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
        LOCAL_RANK="0",  # 每进程经 CUDA_VISIBLE_DEVICES 只见一张卡
        CUDA_VISIBLE_DEVICES=str(rank),
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
        for op in x["ops"]:
            parts = [f"k{op['key']} ok={op.get('ok')}"]
            if op.get("ready_us") is not None:
                parts.append(
                    f"ready={op['ready_us']} admit={op['admit_us']} "
                    f"submit={op['submit_us']} complete={op['complete_us']} "
                    f"actual={op['actual_us']}us"
                )
            lines.append(f"      r{rank} " + " ".join(parts))
        if x.get("producer"):
            p = x["producer"]
            lines.append(
                f"      r{rank} producer: enqueue={p['enqueue_s']}s "
                f"busy_at_compute_start={p['worker_busy_at_compute_start']} "
                f"compute={p['producer_compute_s']}s"
            )
    return "\n".join(lines)


def _accept(r: dict) -> tuple[bool, str]:
    """跨场景验收：sequence 一致、producer 不阻塞、无错误依赖/发散。"""
    if r["scenario"] == "no_wait_ready":
        oks = {x["ops"][0]["ok"] for x in r["results"] if x.get("status") == "ok"}
        note = (f"NOTE: 竞争演示，ok 预期 False/不确定；实际 oks={oks}")
        ok = not r["timed_out"] and {x.get("status") for x in r["results"]} == {"ok"}
        return ok, note
    if r["timed_out"]:
        return False, "超时（挂起）"
    if r["scenario"] == "op_mismatch":
        statuses = {x.get("status") for x in r["results"]}
        ok = statuses == {"validation_error"}
        return ok, ("" if ok else f"statuses={statuses}")
    statuses = {x.get("status") for x in r["results"]}
    if statuses != {"ok"}:
        return False, f"statuses={statuses}"
    seqs = {tuple(str(s) for s in x["launched_seq"].values()) for x in r["results"]}
    if len(seqs) != 1:
        return False, "sequence logs diverge across ranks"
    # out_of_order_submit：必须强制为计划顺序 [k0, k1]。
    if r["scenario"] == "out_of_order_submit":
        first = r["results"][0]["launched_seq"]["dp"]
        keys = [k[6] for k in first]  # as_list 的最后一个字段是 ordinal
        if keys != [0, 1]:
            return False, f"乱序提交未被强制为 [k0,k1]：{keys}"
    if r["scenario"] == "producer_overlap":
        prod = [x.get("producer") for x in r["results"]]
        if not all(p and p["worker_busy_at_compute_start"] for p in prod):
            return False, "producer 的 GPU 计算未与 worker collective 重叠"
    return True, ""


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
        r = run_scenario(name)
        print(_summarize(r))
        acc, note = _accept(r)
        if acc:
            verdict = "      ACCEPT" + (f": {note}" if note else
                                        ": sequence logs identical, no divergence")
        else:
            verdict = f"      REJECT: {note}"
        print(verdict)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
