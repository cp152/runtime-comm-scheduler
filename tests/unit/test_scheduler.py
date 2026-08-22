"""M2 unit tests: AdmissionScheduler 的 admission 状态机与 ScheduledWork。

用 ``FakeWork`` 模拟底层 ``Work``，不依赖 Gloo/NCCL。覆盖：
FIFO 发射顺序、乱序提交强制、重复拒绝、plan 校验失败、outstanding 上限、
延迟 ready、telemetry 单调性、ScheduledWork 的延迟绑定语义。
"""

from __future__ import annotations

import threading

import pytest

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    IntentState,
    Plan,
    TaskKey,
)
from runtime_comm_scheduler.validate import ValidationError


class FakeWork:
    """模拟底层分布式 Work：显式 complete() 后 wait/is_completed 才成功。"""

    def __init__(self):
        self._done = threading.Event()

    def complete(self) -> None:
        self._done.set()

    def wait(self, timeout=None) -> bool:
        return self._done.wait(timeout)

    def is_completed(self) -> bool:
        return self._done.is_set()


def _key(ordinal=0, process_group_id="dp"):
    return TaskKey(
        iteration=0, microbatch=0, parallelism="dp",
        process_group_id=process_group_id, layer_id=0, bucket_id=0,
        ordinal=ordinal,
    )


def _plan(keys, op="all_reduce", num_bytes=8):
    return Plan(version=0, window_id=0, entries=tuple((k, op, num_bytes) for k in keys))


def _intent(key, launch, op="all_reduce", num_bytes=8, ready_event=None):
    return CommIntent(
        key=key, op=op, tensor=None, process_group=None, num_bytes=num_bytes,
        launch_fn=launch, ready_event=ready_event,
    )


def _mk(created, works, key):
    """构造并登记一个 FakeWork，把 key 记入发射日志。"""
    fake = FakeWork()
    created.append(key)
    works[key] = fake
    return fake


# --- 发射顺序与强制 ------------------------------------------------


def test_fifo_launch_order():
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]))
    created, works = [], {}
    i0 = _intent(k0, lambda: _mk(created, works, k0))
    i1 = _intent(k1, lambda: _mk(created, works, k1))
    sched.submit(i0)
    sched.submit(i1)
    assert created == [k0, k1]
    assert sched.sequence_log() == {"dp": [k0, k1]}


def test_out_of_order_submission_enforced():
    # plan 顺序 [k0, k1]，先提交 k1 会被 parked，k0 提交后才按计划发射。
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]))
    created, works = [], {}
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    assert created == []          # k1 不是队首，parked
    assert not w1.is_completed()
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    assert created == [k0, k1]    # k0 提交后 drain 顺带发射 k1
    works[k0].complete()
    works[k1].complete()
    assert w0.wait()
    assert w1.wait()
    assert sched.sequence_log() == {"dp": [k0, k1]}


def test_missing_head_parks_pending_intent():
    # 队首 k0 从未提交，k1 即使提交也一直 parked；ScheduledWork 未发射。
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]))
    created, works = [], {}
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    assert created == []
    assert w1.wait(timeout=0.05) is False
    sched.close()


# --- 校验与 fail-stop ------------------------------------------------


def test_duplicate_rejected():
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]))
    created, works = [], {}
    sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    with pytest.raises(ValidationError, match="duplicate"):
        sched.submit(_intent(k0, lambda: _mk(created, works, k0)))


def test_validation_error_op_mismatch():
    # intent 的 op 与 plan 不一致 -> fail-stop，不发射。
    k0 = _key(0)
    plan = _plan([k0], op="all_reduce")
    sched = AdmissionScheduler(plan)
    created, works = [], {}
    with pytest.raises(ValidationError, match="op mismatch"):
        sched.submit(_intent(k0, lambda: _mk(created, works, k0), op="all_gather"))
    assert created == []


def test_validation_error_key_not_in_plan():
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]))
    with pytest.raises(ValidationError, match="not in plan"):
        sched.submit(_intent(_key(9), lambda: None))


# --- outstanding 上限 ------------------------------------------------


def test_outstanding_limit():
    k0, k1, k2 = _key(0), _key(1), _key(2)
    sched = AdmissionScheduler(_plan([k0, k1, k2]), max_outstanding=1)
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    assert created == [k0]
    # 在途已满，k1/k2 都被 parked 而非发射。
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    w2 = sched.submit(_intent(k2, lambda: _mk(created, works, k2)))
    assert created == [k0]
    # k0 完成（wait 触发完成回调）后 slot 释放，drain 顺带发射 k1。
    works[k0].complete()
    assert w0.wait()
    assert created == [k0, k1]
    # k1 完成后再发射 k2。
    works[k1].complete()
    assert w1.wait()
    assert created == [k0, k1, k2]
    works[k2].complete()
    assert w2.wait()


# --- 延迟 ready --------------------------------------------------


def test_delayed_ready():
    k0 = _key(0)
    ready = threading.Event()
    sched = AdmissionScheduler(_plan([k0]))
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0), ready_event=ready))
    assert created == []          # producer 未 ready
    ready.set()
    sched.nudge()                 # ready 后显式 nudge 触发发射
    assert created == [k0]
    works[k0].complete()
    assert w0.wait()


# --- telemetry ----------------------------------------------------


def test_telemetry_monotonic_and_duration():
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]))
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    works[k0].complete()
    assert w0.wait()
    (t,) = sched.timings()
    assert t.key == k0
    assert t.ready_ts <= t.admit_ts <= t.submit_ts <= t.complete_ts
    assert t.actual_duration_us is not None and t.actual_duration_us >= 0


def test_telemetry_records_ready_for_parked():
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]))
    created, works = [], {}
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    works[k0].complete()
    works[k1].complete()
    w1.wait()
    t0, t1 = sched.timings()
    assert t0.ready_ts is not None and t1.ready_ts is not None
    assert t1.admit_ts is not None  # parked 的 k1 最终也被发射并记录


# --- ScheduledWork 状态 ---------------------------------------------


def test_work_is_completed_requires_underlying():
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]))
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    assert not w0.is_completed()
    works[k0].complete()
    assert w0.is_completed()


def test_work_wait_bounded_timeout_before_launch():
    # 未发射（parked）的 work：wait 带超时返回 False，不阻塞。
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]))
    created, works = [], {}
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    assert w1.wait(timeout=0.05) is False


def test_intent_lifecycle_transitions():
    # 校验生命周期推进：CREATED -> ... -> COMPLETED。
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]))
    created, works = [], {}
    intent = _intent(k0, lambda: _mk(created, works, k0))
    assert intent.state is IntentState.CREATED
    w0 = sched.submit(intent)
    assert intent.state is IntentState.SUBMITTED
    works[k0].complete()
    assert w0.wait()
    assert intent.state is IntentState.COMPLETED
