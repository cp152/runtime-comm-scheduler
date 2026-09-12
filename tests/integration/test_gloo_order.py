"""Three-rank Gloo harness for overlapping process-group plan projections."""

from __future__ import annotations

import json
import multiprocessing
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import pytest

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    DirectLaunchExecutor,
    Plan,
    TaskKey,
)


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
    except PermissionError:
        pytest.skip("sandbox does not permit the local Gloo rendezvous socket")


def _key(group_id: str, ordinal: int) -> TaskKey:
    return TaskKey(0, 0, "mixed", group_id, 0, 0, ordinal)


def _worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    groups = {
        "a": dist.new_group([0, 1]),
        "b": dist.new_group([1, 2]),
        "c": dist.new_group([2, 0]),
    }
    members = {"a": {0, 1}, "b": {1, 2}, "c": {2, 0}}
    keys = [
        _key("a", 0),
        _key("b", 0),
        _key("c", 0),
        _key("a", 1),
        _key("b", 1),
        _key("c", 1),
    ]
    plan = Plan(
        version=0,
        window_id=0,
        entries=tuple((key, "all_reduce", 4) for key in keys),
    )
    local_groups = {group_id for group_id, ranks in members.items() if rank in ranks}
    scheduler = AdmissionScheduler(
        plan,
        local_group_ids=local_groups,
        executor=DirectLaunchExecutor(),
    )
    tensors = {}
    works = []
    try:
        # Reverse submission ensures the scheduler, not arrival order, enforces
        # each rank's projection of the shared A -> B -> C host plan.
        for key in reversed(plan.local_projection(local_groups)):
            tensor = torch.tensor([float(rank + 1)])
            tensors[key] = tensor
            group = groups[key.process_group_id]
            intent = CommIntent(
                key=key,
                op="all_reduce",
                tensor=tensor,
                process_group=group,
                num_bytes=4,
                launch_fn=lambda tensor=tensor, group=group: dist.all_reduce(
                    tensor, group=group, async_op=True
                ),
            )
            works.append(scheduler.submit(intent))

        assert all(work.wait() for work in works)
        scheduler.finish_window(timeout=10)
        expected = {"a": 3.0, "b": 5.0, "c": 4.0}
        assert all(
            tensor.item() == expected[key.process_group_id]
            for key, tensor in tensors.items()
        )
        result = {
            "rank": rank,
            "plan_digest": plan.digest(),
            "local_groups": sorted(local_groups),
            "launch_log": [key.process_group_id for key in scheduler.sequence_log()],
            "group_log": {
                group: [key.ordinal for key in sequence]
                for group, sequence in scheduler.group_sequence_log().items()
            },
        }
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        try:
            scheduler.close()
        finally:
            dist.destroy_process_group()


def test_overlapping_groups_follow_shared_plan_projection(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    processes = [
        ctx.Process(target=_worker, args=(rank, 3, port, str(tmp_path)))
        for rank in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise AssertionError("Gloo projection harness timed out")
        assert process.exitcode == 0

    results = [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text())
        for rank in range(3)
    ]
    assert [result["launch_log"] for result in results] == [
        ["a", "c", "a", "c"],
        ["a", "b", "a", "b"],
        ["b", "c", "b", "c"],
    ]
    assert len({result["plan_digest"] for result in results}) == 1
    member_ranks = {"a": (0, 1), "b": (1, 2), "c": (2, 0)}
    for group, ranks in member_ranks.items():
        assert [results[rank]["group_log"][group] for rank in ranks] == [
            [0, 1],
            [0, 1],
        ]
    for result in results:
        nonlocal_groups = {"a", "b", "c"}.difference(result["local_groups"])
        assert nonlocal_groups.isdisjoint(result["group_log"])
