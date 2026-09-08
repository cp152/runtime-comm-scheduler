"""v19 nsys 启动器：按 rank 各包一层 nsys profile，跑完出 cuda_api_sum /
cuda_gpu_kern_sum 表。用法（盒子上 examples/ 目录）：
    /root/miniconda3/bin/python run_nsys_v19.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "wait_recheck_worker_v19.py")
_NSYS = "/usr/local/bin/nsys"
_PY = sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_cmd(rank: int, port: int) -> list[str]:
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK="0",
        CUDA_VISIBLE_DEVICES=str(rank),
    )
    cmd = [
        _NSYS, "profile", "--force-overwrite=true",
        "-o", f"/tmp/nsys_r{rank}",
        "-t", "cuda,osrt,nvtx",
        "--cuda-memory-usage=true",
        "--cuda-graph-trace=node",
        _PY, _WORKER,
    ]
    return cmd, env


def main() -> int:
    port = _free_port()
    procs = []
    for r in range(2):
        cmd, env = _rank_cmd(r, port)
        procs.append((r, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True)))
    for r, p in procs:
        try:
            out, err = p.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            print(f"TIMEOUT rank {r}")
            return 1
        line = out.strip().splitlines()[-1] if out.strip() else ""
        print(f"===== rank {r} worker JSON =====")
        print(line)
        if err.strip():
            print(f"----- rank {r} nsys stderr (tail) -----")
            print("\n".join(err.strip().splitlines()[-15:]))

    for r, _ in procs:
        print(f"\n===== rank {r} cuda_api_sum =====")
        subprocess.run(
            [_NSYS, "stats", "--report=cuda_api_sum", "--format=table",
             f"/tmp/nsys_r{r}.nsys-rep"],
            check=False,
        )
        print(f"===== rank {r} cuda_gpu_kern_sum =====")
        subprocess.run(
            [_NSYS, "stats", "--report=cuda_gpu_kern_sum", "--format=table",
             f"/tmp/nsys_r{r}.nsys-rep"],
            check=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
