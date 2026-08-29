"""wait 语义重测 v2 driver：两 rank NCCL 子进程（各绑一张 3090）+ 汇总打印。

用法（盒子上）：:

    /root/miniconda3/bin/python run_wait_recheck_v2.py

每个 rank 输出 S1/S2/S3/S4 测量 JSON；对比结论见
docs/todo-revisit-wait-semantics.md 的映射表。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(
    _HERE, sys.argv[1] if len(sys.argv) > 1 else "wait_recheck_worker_v2.py"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_rank(rank: int, world: int, port: int):
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK="0",  # 每进程经 CUDA_VISIBLE_DEVICES 只见一张卡
        CUDA_VISIBLE_DEVICES=str(rank),
    )
    return subprocess.Popen(
        [sys.executable, _WORKER], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def main() -> int:
    port = _free_port()
    procs = [_run_rank(r, 2, port) for r in range(2)]
    for p in procs:
        try:
            out, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            print("TIMEOUT: rank process hung")
            return 1
        line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            print(json.dumps(json.loads(line), indent=2))
        except json.JSONDecodeError:
            print("BAD OUTPUT:", out[-800:], err[-500:])
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
