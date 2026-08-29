"""wait 语义干净重测（docs/todo-revisit-wait-semantics.md 实验 1/2/3）。

两 rank NCCL，非 legacy 显式流，无 profiler。对 2 GiB all_reduce 分别测：

- **实验 1（stream-ordered 是否生效）**：发射后立即在当前流上 ``event.record``
  并 ``query()``——False 表示等待 NCCL（生效），True 表示立即触发（未生效）；
  再轮询到该 event 触发，得到真实 GPU 完成时刻 gt1。
- **实验 2（裸 ``Work.wait()``）**：墙钟 vs 同 collective 的 ground-truth gt2。
  若 wait_s << gt2，说明 wait 提前返回（M3 发现 #2 复现）。
- **实验 3（``Work.get_future().wait()``）**：墙钟 vs 同 collective 的 gt3。

每个实验用**独立的 fresh collective**（避免「被前一个实验等到完成」污染计时），
且各配一个发射后立即 record 的 event 作为该 collective 的真实完成参照。
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_N = 1 << 29  # 2 GiB float32


def _fill(rank: int) -> torch.Tensor:
    return torch.full((_N,), float(rank + 1), device="cuda")


def _poll_until(ev: torch.cuda.Event) -> float:
    """轮询 event 直到触发，返回墙钟秒数（非阻塞 query，不做 device sync）。"""
    t0 = time.perf_counter()
    while not ev.query():
        time.sleep(0.001)
    return time.perf_counter() - t0


def _run(rank: int, world: int, local_rank: int) -> dict:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    # warmup：去掉 NCCL 首 op 的 lazy init 延迟（~350ms）。
    t0 = torch.full((1024,), 1.0, device="cuda")
    dist.all_reduce(t0, op=dist.ReduceOp.SUM, async_op=True).wait()

    out: dict = {"rank": rank, "world": world}
    s = torch.cuda.Stream()  # 非 legacy 显式流；所有测量都在它上面（避免隐式同步）

    with torch.cuda.stream(s):
        # 实验 1：发射后立即 query 当前流上的 event；再轮询拿真实完成时刻。
        t1 = _fill(rank)
        w1 = dist.all_reduce(t1, op=dist.ReduceOp.SUM, async_op=True)
        ev1 = torch.cuda.Event()
        ev1.record()
        q0 = bool(ev1.query())
        gt1 = _poll_until(ev1)
        ok1 = w1.wait()
        out["exp1"] = {"q0_immediate": q0, "gt1_s": round(gt1, 6), "ok1": ok1}

        # 实验 2：裸 Work.wait() 墙钟 + 独立 ground-truth event。
        t2 = _fill(rank)
        w2 = dist.all_reduce(t2, op=dist.ReduceOp.SUM, async_op=True)
        ev2 = torch.cuda.Event()
        ev2.record()
        tw0 = time.perf_counter()
        ok2 = w2.wait()
        wait_s = time.perf_counter() - tw0
        gt2 = _poll_until(ev2)
        out["exp2"] = {"wait_s": round(wait_s, 6), "gt2_s": round(gt2, 6), "ok2": ok2}

        # 实验 3：get_future().wait() 墙钟 + 独立 ground-truth event。
        t3 = _fill(rank)
        w3 = dist.all_reduce(t3, op=dist.ReduceOp.SUM, async_op=True)
        ev3 = torch.cuda.Event()
        ev3.record()
        fut = w3.get_future()
        tf0 = time.perf_counter()
        fut.wait()
        fut_s = time.perf_counter() - tf0
        gt3 = _poll_until(ev3)
        out["exp3"] = {"fut_s": round(fut_s, 6), "gt3_s": round(gt3, 6)}

    dist.destroy_process_group()
    return out


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(json.dumps(_run(rank, world, local_rank)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
