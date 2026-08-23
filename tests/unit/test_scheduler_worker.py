"""M4 unit tests: AdmissionScheduler worker 模式（异步 admission worker）。

用 ``FakeWork`` 模拟底层 ``Work``，不依赖 Gloo/NCCL（worker 模式在无 CUDA
时 ``comm_stream=None``，不加 stream 上下文，仅验证线程化发射路径）。覆盖：
producer 提交不阻塞、乱序提交强制（经 worker）、延迟 ready 门控、outstanding
上限、worker launch 抛错 fail-stop、``get_future()``、``close()`` join。

并发路径的确定性写法：用 launch_fn 内部阻塞的 gate 断言「尚未发射」，用
轮询 ``sequence_log()``/created 断言「最终发射」，避免与 worker 抢跑。
"""

from __future__ import annotations

import threading
import time

import pytest

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
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


def _wait_for(pred, timeout=2.0) -> None:
    """轮询直到 ``pred()`` 为真（worker 是异步的，需要最终态断言）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition not met within {timeout}s")


def test_worker_submit_does_not_block_launch():
    """producer 提交不阻塞：launch_fn 在内部 gate 上阻塞时，submit 已返回
    且 worker 尚未发射；放行 gate 后 worker 才发射并 bind。"""
    k0 = _key(0)
    gate = threading.Event()
    created, works = [], {}

    def launch():
        gate.wait()
        return _mk(created, works, k0)

    sched = AdmissionScheduler(_plan([k0]), worker=True)
    w0 = sched.submit(_intent(k0, launch))
    assert created == []          # worker 卡在 gate 上，尚未发射
    assert not w0.is_completed()
    gate.set()
    _wait_for(lambda: created == [k0])
    works[k0].complete()
    assert w0.wait(timeout=1)     # launch 后 bind，wait 走 FakeWork
    assert sched.sequence_log() == {"dp": [k0]}
    sched.close()


def test_worker_out_of_order_submission_enforced():
    """经 worker 仍强制计划顺序：先提交 k1 被 parked，k0 提交后 worker
    按 [k0, k1] 发射，sequence log 与同步模式一致。"""
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]), worker=True)
    created, works = [], {}
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    assert created == []          # k1 非队首，worker drain 无事可做
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    _wait_for(lambda: created == [k0, k1])
    works[k0].complete()
    works[k1].complete()
    assert w0.wait(timeout=1)
    assert w1.wait(timeout=1)
    assert sched.sequence_log() == {"dp": [k0, k1]}
    sched.close()


def test_worker_delayed_ready_gates_launch():
    """延迟 ready（threading.Event）在 worker 模式下同样门控发射。"""
    k0 = _key(0)
    ready = threading.Event()
    sched = AdmissionScheduler(_plan([k0]), worker=True)
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0), ready_event=ready))
    assert created == []          # producer 未 ready，worker 不发射
    ready.set()
    sched.nudge()                 # ready 后唤醒 worker
    _wait_for(lambda: created == [k0])
    works[k0].complete()
    assert w0.wait(timeout=1)
    sched.close()


def test_worker_outstanding_limit():
    """outstanding 上限经 worker 同样生效：槽位释放后唤醒 worker 发射下一个。"""
    k0, k1, k2 = _key(0), _key(1), _key(2)
    sched = AdmissionScheduler(_plan([k0, k1, k2]), max_outstanding=1, worker=True)
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    _wait_for(lambda: created == [k0])
    w1 = sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    w2 = sched.submit(_intent(k2, lambda: _mk(created, works, k2)))
    assert created == [k0]        # 在途已满，k1/k2 parked
    works[k0].complete()
    assert w0.wait(timeout=1)     # 完成回调释放槽位并唤醒 worker
    _wait_for(lambda: created == [k0, k1])
    works[k1].complete()
    assert w1.wait(timeout=1)
    _wait_for(lambda: created == [k0, k1, k2])
    works[k2].complete()
    assert w2.wait(timeout=1)
    sched.close()


def test_worker_launch_error_fail_stop_on_next_submit():
    """worker 内 launch_fn 抛错：worker 停止并记录错误，下一次 submit 在
    训练线程上重新抛出（fail-stop，而非静默挂起）。"""
    k0, k1 = _key(0), _key(1)

    def boom():
        raise RuntimeError("launch boom")

    sched = AdmissionScheduler(_plan([k0, k1]), worker=True)
    sched.submit(_intent(k0, boom))
    _wait_for(lambda: sched._worker_error is not None)
    with pytest.raises(RuntimeError, match="launch boom"):
        sched.submit(_intent(k1, lambda: None))
    # worker 已停止：close() 不再需要 join 存活线程
    with pytest.raises(RuntimeError, match="launch boom"):
        sched.close()


def test_worker_get_future_resolves_on_completion():
    """get_future()：wait 完成前 not done，wait 后 done 且 result 为 True。"""
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]), worker=True)
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    fut = w0.get_future()
    assert not fut.done()
    _wait_for(lambda: created == [k0])   # 等 worker 发射并登记 FakeWork
    works[k0].complete()
    assert w0.wait(timeout=1)
    assert fut.done()
    assert fut.result(timeout=0) is True
    assert fut.cancelled() is False
    sched.close()


def test_worker_close_joins_thread():
    """close() 停止并 join worker 线程。"""
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]), worker=True)
    created, works = [], {}
    sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    _wait_for(lambda: created == [k0])
    sched.close()
    assert not sched._thread.is_alive()


def test_worker_sequence_log_snapshot():
    """worker 模式下 sequence_log 与 timings 可安全快照（持锁读）。"""
    k0, k1 = _key(0), _key(1)
    sched = AdmissionScheduler(_plan([k0, k1]), worker=True)
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    sched.submit(_intent(k1, lambda: _mk(created, works, k1)))
    _wait_for(lambda: created == [k0, k1])  # 等 worker 发射两个 intent
    works[k0].complete()
    works[k1].complete()
    assert w0.wait(timeout=1)
    assert sched.sequence_log() == {"dp": [k0, k1]}
    assert [t.key for t in sched.timings()] == [k0, k1]
    sched.close()


def test_worker_duplicate_and_validation_still_fail_stop():
    """worker 模式下 plan 校验与重复提交仍 fail-stop（发射前）。"""
    k0 = _key(0)
    sched = AdmissionScheduler(_plan([k0]), worker=True)
    created, works = [], {}
    sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    with pytest.raises(ValidationError, match="duplicate"):
        sched.submit(_intent(k0, lambda: _mk(created, works, k0)))
    with pytest.raises(ValidationError, match="op mismatch"):
        sched.submit(_intent(k0, lambda: _mk(created, works, k0), op="all_gather"))
    sched.close()
