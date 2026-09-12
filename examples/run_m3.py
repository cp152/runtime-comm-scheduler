"""M3 driver：对每个场景启动两 rank NCCL 子进程（各绑一张 3090）并对比结果。

用法：:

    python run_m3.py                # 跑全部场景
    python run_m3.py --only fifo    # 只跑一个场景

每个场景两 rank 各跑一张 GPU（``CUDA_VISIBLE_DEVICES=<rank>``）。验收
（phase1-plan M3/M4.5）：profiler/Nsight 证明 stream dependency 正确，且
``wait()`` 保持 stream-ordered 语义；FIFO/固定重排安全且 sequence log 相同。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "m3_worker.py")
_TRACE_DIR = os.environ.get("RCS_M3_TRACE_DIR", "/root/autodl-tmp/m3_traces")

# 场景 -> (超时秒数, 期望的 status)
SCENARIOS = {
    "fifo": (60, "ok"),
    "fixed_reorder": (60, "ok"),
    "delayed_ready": (60, "ok"),
    "no_wait_ready": (60, "ok"),          # status 仍为 ok，op.ok 预期 False（竞争）
    "op_mismatch": (60, "validation_error"),
    "wait_stream_ordered": (60, "ok"),
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
    cmd = [sys.executable, _WORKER, "--scenario", scenario,
           "--trace-dir", _TRACE_DIR]
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
            parts = [
                f"k{op['key']} ok={op.get('ok')}",
            ]
            if op.get("ready_us") is not None:
                parts.append(
                    f"ready={op['ready_us']} admit={op['admit_us']} "
                    f"submit={op['submit_us']} complete={op['complete_us']} "
                    f"actual={op['actual_us']}us"
                )
            if "consumer_after_wait_return" in op:
                parts.append(
                    f"after_wait_return={op['consumer_after_wait_return']} "
                    f"after_stream_sync={op['consumer_after_stream_sync']} "
                    f"wait_s={op.get('wait_s')}"
                )
            lines.append(f"      r{rank} " + " ".join(parts))
    # 验收判断
    if r["scenario"] == "no_wait_ready":
        oks = {x["ops"][0]["ok"] for x in results if x.get("status") == "ok"}
        lines.append(
            "      NOTE: no_wait_ready 是竞争演示，ok 预期 False/不确定；"
            f"实际 oks={oks}"
        )
        if statuses == {r["expect"]}:
            lines.append("      ACCEPT: both ranks completed (race demonstrated)")
        else:
            lines.append("      REJECT: statuses do not match")
    elif r["scenario"] == "wait_stream_ordered":
        oks = [x for x in results if x.get("status") == "ok" and x.get("ops")]
        returns = {x["ops"][0]["consumer_after_wait_return"] for x in oks}
        syncs = {x["ops"][0]["consumer_after_stream_sync"] for x in oks}
        if statuses == {"ok"} and returns == {False} and syncs == {True}:
            lines.append(
                "      ACCEPT: wait() inserts a consumer-stream dependency "
                "without waiting for physical GPU completion"
            )
        else:
            lines.append(
                f"      REJECT: after_return={returns} after_sync={syncs}"
            )
    elif statuses == {r["expect"]} and r["expect"] == "ok":
        seqs = {tuple(str(s) for s in x["launched_seq"].values()) for x in results}
        if len(seqs) == 1:
            lines.append("      ACCEPT: all ranks' sequence logs identical")
        else:
            lines.append("      REJECT: sequence logs diverge")
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
