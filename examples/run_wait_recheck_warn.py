"""v17 driver：与 run_wait_recheck_v2 相同，但把每个 rank 的 stderr（sync-debug
warning + 时间戳 mark）也完整透传到自己的 stderr，供分析 warning 是否落在
sum/item 阻塞区间内。用法：python run_wait_recheck_warn.py wait_recheck_worker_v17.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(
    _HERE, sys.argv[1] if len(sys.argv) > 1 else "wait_recheck_worker_v17.py"
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
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES=str(rank),
    )
    return subprocess.Popen(
        [sys.executable, _WORKER], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def main() -> int:
    port = _free_port()
    procs = [_run_rank(r, 2, port) for r in range(2)]
    for r, p in enumerate(procs):
        try:
            out, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            print(f"TIMEOUT: rank {r} process hung")
            return 1
        line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            print(json.dumps(json.loads(line), indent=2))
        except json.JSONDecodeError:
            print(f"BAD OUTPUT rank{r}:", out[-800:], err[-500:])
            return 1
        print(f"\n===== rank {r} STDERR (warning 与 mark 交错) =====")
        sys.stderr.flush()
        print(err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
