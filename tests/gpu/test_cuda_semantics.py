"""M4.5 CUDA tests for producer/gate/NCCL/consumer stream semantics."""

from __future__ import annotations

import time

import pytest
import torch
import torch.distributed as dist

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    Plan,
    TaskKey,
    TorchProcessGroupExecutor,
)

_N = 1024


def _key(ordinal=0, group_id="dp"):
    return TaskKey(0, 0, "dp", group_id, 0, 0, ordinal)


def _plan(keys, num_bytes=_N * 4):
    return Plan(
        version=0,
        window_id=0,
        entries=tuple((key, "all_reduce", num_bytes) for key in keys),
    )


def _scheduler(plan):
    return AdmissionScheduler(
        plan,
        local_group_ids=plan.group_ids(),
        executor=TorchProcessGroupExecutor(torch.cuda.current_device()),
    )


def _wait_bound(work, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if work.is_bound:
            return
        time.sleep(0.002)
    raise AssertionError("work was not bound")


@pytest.fixture(scope="module")
def pg():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0
    )
    warm = torch.ones(1024, device="cuda")
    dist.all_reduce(warm, async_op=True).wait()
    yield
    dist.destroy_process_group()


def test_scheduler_bridges_producer_event_to_nccl(pg):
    key = _key()
    tensor = torch.full((_N,), -1.0, device="cuda")
    producer = torch.cuda.Stream()
    ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        value = torch.randn(2048, 2048, device="cuda")
        for _ in range(30):
            value = value @ value
        tensor.fill_(1.0)
        ready.record()

    intent = CommIntent(
        key=key,
        op="all_reduce",
        tensor=tensor,
        process_group=None,
        num_bytes=_N * 4,
        launch_fn=lambda: dist.all_reduce(tensor, async_op=True),
        ready_event=ready,
        device=tensor.device,
    )
    scheduler = _scheduler(_plan([key]))
    work = scheduler.submit(intent)

    consumer = torch.cuda.Stream()
    after = torch.cuda.Event()
    with torch.cuda.stream(consumer):
        assert work.wait()
        after.record()
    consumer.synchronize()
    assert after.query()
    with torch.cuda.stream(consumer):
        assert torch.all(tensor == 1.0).item()
    scheduler.finish_window(timeout=2)
    scheduler.close()


def test_wait_is_stream_ordered_and_does_not_synchronize_device(pg, monkeypatch):
    key = _key()
    tensor = torch.ones(_N, device="cuda")
    scheduler = _scheduler(_plan([key]))
    unrelated = torch.cuda.Stream()
    unrelated_done = torch.cuda.Event()
    with torch.cuda.stream(unrelated):
        torch.cuda._sleep(1_000_000_000)
        unrelated_done.record()

    def launch():
        value = torch.randn(2048, 2048, device="cuda")
        for _ in range(50):
            value = value @ value
        return dist.all_reduce(tensor, async_op=True)

    work = scheduler.submit(
        CommIntent(
            key=key,
            op="all_reduce",
            tensor=tensor,
            process_group=None,
            num_bytes=_N * 4,
            launch_fn=launch,
            device=tensor.device,
        )
    )
    _wait_bound(work)

    def forbidden_synchronize(*args, **kwargs):
        raise AssertionError("ScheduledWork.wait performed a device synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_synchronize)
    consumer = torch.cuda.Stream()
    after = torch.cuda.Event()
    with torch.cuda.stream(consumer):
        assert work.wait()
        after.record()
    assert not after.query()
    consumer.synchronize()
    assert after.query()
    assert not unrelated_done.query()
    unrelated.synchronize()
    scheduler.finish_window(timeout=2)
    scheduler.close()


def test_different_groups_do_not_share_gate_stream_dependency(pg):
    """Host launch A->B must not imply producer-ready A -> gate B."""

    class EventWork:
        def __init__(self, done):
            self.done = done

        def wait(self, timeout=None):
            torch.cuda.current_stream().wait_event(self.done)
            return True

        def is_completed(self):
            return self.done.query()

    a, b = _key(0, "a"), _key(0, "b")
    plan = _plan([a, b])
    executor = TorchProcessGroupExecutor(torch.cuda.current_device())
    scheduler = AdmissionScheduler(
        plan,
        local_group_ids={"a", "b"},
        executor=executor,
    )

    slow_producer = torch.cuda.Stream()
    a_ready = torch.cuda.Event()
    with torch.cuda.stream(slow_producer):
        value = torch.randn(2048, 2048, device="cuda")
        for _ in range(80):
            value = value @ value
        a_ready.record()

    a_done, b_done = torch.cuda.Event(), torch.cuda.Event()

    def launch_a():
        a_done.record()
        return EventWork(a_done)

    def launch_b():
        b_done.record()
        return EventWork(b_done)

    wa = scheduler.submit(
        CommIntent(a, "all_reduce", None, None, _N * 4, launch_a, ready_event=a_ready)
    )
    wb = scheduler.submit(
        CommIntent(b, "all_reduce", None, None, _N * 4, launch_b)
    )
    _wait_bound(wa)
    _wait_bound(wb)
    b_done.synchronize()
    assert b_done.query()
    assert not a_done.query()
    assert executor.gate_stream("a") != executor.gate_stream("b")
    assert scheduler.sequence_log() == [a, b]

    a_done.synchronize()
    scheduler.finish_window(timeout=2)
    scheduler.close()


def test_raw_collective_still_works_after_scheduler_close(pg):
    key = _key()
    tensor = torch.ones(_N, device="cuda")
    scheduler = _scheduler(_plan([key]))
    work = scheduler.submit(
        CommIntent(
            key=key,
            op="all_reduce",
            tensor=tensor,
            process_group=None,
            num_bytes=_N * 4,
            launch_fn=lambda: dist.all_reduce(tensor, async_op=True),
            device=tensor.device,
        )
    )
    assert work.wait()
    scheduler.finish_window(timeout=2)
    scheduler.close()

    raw = torch.full((_N,), 2.0, device="cuda")
    assert dist.all_reduce(raw, async_op=True).wait()
    assert torch.all(raw == 2.0).item()
