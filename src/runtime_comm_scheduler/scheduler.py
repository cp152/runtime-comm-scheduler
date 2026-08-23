"""AdmissionScheduler：训练线程内的同步 admission facade（M2），M4 增加
可选的异步 admission worker。

两种模式：

- ``worker=False``（默认，M2/M3 行为）：``submit`` 在训练线程内同步校验
  plan、park、并就地 drain 发射。不加锁，假定单线程使用。保留为既有
  harness 与单元测试的兼容路径。
- ``worker=True``（M4）：deferred launch 移到专门的 worker 线程。训练线程
  的 ``submit`` 只做校验 + park + 唤醒，立即返回；worker 线程被唤醒后按
  每个 process group 的队首序列 drain 并发射。共享状态用一把锁保护；
  发射时（若提供 ``comm_stream``）在显式 communication stream 上下文中
  调用 ``launch_fn``，使 collective 排在 producer 的 CUDA ready event 之后
  （§4.5 的 producer stream -> ready event -> communication stream）。

约束（architecture.md §5.3）：
1. 可以 delay task，但不能在 producer ready 前提交；
2. 不能在本地跳过或重排同一个 process group 的计划子序列；
3. outstanding collective 数量不超过上限；
4. collective 提交给底层后不可取消/重排。

M4 记录的限制：
- worker 线程发射时持有调度锁（发射是 async_op 的 enqueue，约百微秒）。
  若 launch_fn 阻塞（如 NCCL rendezvous），训练线程的 submit 会等锁。
- 同一 communicator 绝不能由多个线程并发提交（M4 关口实测会打挂进程）；
  本设计只有 worker 一个提交线程，天然避开。
- worker 内 launch_fn 抛错时捕获到 ``_worker_error``，下一次
  submit/nudge/观察接口重新抛出（fail-stop）。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Optional

from .intent import CommIntent, IntentState, TaskKey
from .plan import Plan
from .telemetry import CommTiming, now_us
from .validate import ValidationError, validate_intent
from .work import ScheduledWork


class AdmissionScheduler:
    def __init__(
        self,
        plan: Plan,
        *,
        max_outstanding: int = 0,
        worker: bool = False,
        comm_stream: Optional[Any] = None,
    ) -> None:
        """``max_outstanding=0`` 表示不限制同时在途的 collective 数量。

        ``worker=True`` 时启动 admission worker 线程；``comm_stream`` 是
        worker 发射 collective 所用的显式 CUDA communication stream（M4）。
        未提供且 CUDA 可用时自动创建一个；CUDA 不可用（如无 GPU 的单元测试）
        时 worker 不加 stream 上下文，仅做线程化发射。
        """
        self._plan = plan
        self._max_outstanding = max_outstanding
        self._worker_mode = worker
        self._lock = threading.Lock()
        self._worker_error: Optional[BaseException] = None
        self._comm_stream = comm_stream
        self._group_ids: tuple[str, ...] = plan.group_ids()
        # 每个 group 尚未发射的 plan key 序列（头是下一个必须发射的 collective）。
        self._remaining: dict[str, deque[TaskKey]] = {
            g: deque(plan.group_sequence(g)) for g in self._group_ids
        }
        # 已提交但未发射的 intent/work（parked，等待成为队首或 ready）。
        self._pending: dict[str, dict[TaskKey, CommIntent]] = {
            g: {} for g in self._group_ids
        }
        self._pending_work: dict[str, dict[TaskKey, ScheduledWork]] = {
            g: {} for g in self._group_ids
        }
        # 已发射未完成；``_launched`` 是该 group 的 sequence log（发射顺序）。
        self._inflight: dict[str, list[tuple[ScheduledWork, CommTiming]]] = {
            g: [] for g in self._group_ids
        }
        self._launched: dict[str, list[TaskKey]] = {g: [] for g in self._group_ids}
        self._timings: list[CommTiming] = []
        if worker:
            self._init_worker()

    # ---- worker 线程生命周期 -------------------------------------------

    def _init_worker(self) -> None:
        """启动 admission worker：wake/stop 两个 Event + 显式 comm stream。"""
        if self._comm_stream is None:
            self._comm_stream = self._make_comm_stream()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop, name="admission-worker", daemon=True
        )
        self._thread.start()

    @staticmethod
    def _make_comm_stream():
        """CUDA 可用时创建一个非 legacy 显式 stream，否则返回 None。

        默认（legacy）stream 会与所有其他 stream 隐式同步（M3 实测），
        会把 producer 与 collective 的依赖边掩盖掉；M4 要求显式 stream。
        """
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return torch.cuda.Stream()

    def _worker_loop(self) -> None:
        """等待唤醒，逐个 group drain；launch_fn 抛错时记录并停止。"""
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                with self._lock:
                    for g in self._group_ids:
                        self._drain(g)
            except BaseException as exc:  # noqa: BLE001 - fail-stop 需要暴露
                with self._lock:
                    self._worker_error = exc
                self._stop.set()
                return

    def _check_worker_error(self) -> None:
        """worker 曾因 launch_fn 抛错而停止时，在训练线程上重新抛出。"""
        if self._worker_error is not None:
            raise self._worker_error

    # ---- 公共接口 --------------------------------------------------------

    def submit(self, intent: CommIntent) -> ScheduledWork:
        """提交 intent，返回延迟绑定的 ``ScheduledWork``。

        intent 必须与 plan 一致（key 存在、op/num_bytes 匹配）且未被提交过；
        违反任一条件即抛出 :class:`ValidationError`（fail-stop，而非静默）。

        worker 模式下：校验 + park 后唤醒 worker 并立即返回（不阻塞）；
        实际发射由 worker 线程完成。返回值可能尚未发射：只有成为对应 group
        的队首、producer ready 且 outstanding 未达上限时才会被发射。
        """
        validate_intent(intent, self._plan)
        if self._worker_mode:
            self._check_worker_error()
            with self._lock:
                work = self._submit_locked(intent)
            self._wake.set()
            return work
        return self._submit_locked(intent)

    def nudge(self, group_id: Optional[str] = None) -> None:
        """重新检查 admission 条件（ready_event 就绪或 outstanding 释放后调用）。

        worker 模式：只唤醒 worker 重跑全部 group 的 drain（group_id 不再
        区分，drain 全部是超集）。同步模式：就地按 group drain。
        """
        if self._worker_mode:
            self._check_worker_error()
            self._wake.set()
            return
        ids = (group_id,) if group_id is not None else self._group_ids
        for g in ids:
            self._drain(g)

    def close(self) -> None:
        """释放 scheduler 资源；worker 模式停止并 join worker 线程。"""
        if not self._worker_mode:
            return
        self._check_worker_error()
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5)

    # ---- 只读观察 -------------------------------------------------------

    def sequence_log(self) -> dict[str, list[TaskKey]]:
        """每个 process group 已发射 key 的序列（跨 rank 对比的验收对象）。"""
        if self._worker_mode:
            with self._lock:
                return {g: list(keys) for g, keys in self._launched.items()}
        return {g: list(keys) for g, keys in self._launched.items()}

    def timings(self) -> list[CommTiming]:
        """按提交顺序返回所有 intent 的时序记录。"""
        if self._worker_mode:
            with self._lock:
                return list(self._timings)
        return list(self._timings)

    # ---- 内部状态机 ------------------------------------------------------

    def _submit_locked(self, intent: CommIntent) -> ScheduledWork:
        """submit 的核心：校验重复、推进生命周期、park、同步模式就地 drain。

        调用方需持有 ``_lock``（worker 模式）或保证单线程（同步模式）。
        """
        g = intent.key.process_group_id
        self._check_not_duplicate(intent.key, g)
        if intent.state == IntentState.CREATED:
            intent.transition(IntentState.READY)
        elif intent.state != IntentState.READY:
            raise ValidationError(
                f"intent {intent.key.as_list()} in state {intent.state.name} "
                f"cannot be submitted"
            )
        timing = CommTiming(key=intent.key, ready_ts=now_us())
        self._timings.append(timing)
        work = ScheduledWork(
            intent, timing,
            on_complete=lambda: self._on_work_complete(g, timing),
        )
        self._pending[g][intent.key] = intent
        self._pending_work[g][intent.key] = work
        if not self._worker_mode:
            self._drain(g)
        return work

    def _check_not_duplicate(self, key: TaskKey, g: str) -> None:
        if key in self._pending[g] or key in self._pending_work[g]:
            raise ValidationError(f"duplicate submit of {key.as_list()}")
        if key in self._launched[g]:
            raise ValidationError(f"duplicate submit of {key.as_list()}")

    def _is_ready(self, intent: CommIntent) -> bool:
        ev = intent.ready_event
        if ev is None:
            return True
        if isinstance(ev, threading.Event):
            return ev.is_set()
        # CUDA event：ready 由 GPU 侧 stream 顺序保证（_sync_ready_event），
        # 不需要 CPU 侧阻塞，因此这里直接放行。
        return True

    def _drain(self, g: str) -> None:
        """尽可能把队首且已 ready 的 pending intent 发射出去。

        同步模式由 submit/nudge/on_complete 直接调用（单线程）；worker 模式
        由 ``_worker_loop`` 在持有 ``_lock`` 的情况下调用。两个模式下
        ``_drain`` 本身都不取锁，由调用方保证。
        """
        while self._remaining[g]:
            head = self._remaining[g][0]
            intent = self._pending[g].get(head)
            if intent is None:
                break  # 队首尚未提交（parked 后面直到队首到达）
            if not self._is_ready(intent):
                break  # producer 未 ready，不允许提前提交
            if self._max_outstanding and len(self._inflight[g]) >= self._max_outstanding:
                break  # 在途数量达上限
            self._remaining[g].popleft()
            del self._pending[g][head]
            work = self._pending_work[g].pop(head)
            self._admit_and_launch(g, intent, work)

    def _admit_and_launch(self, g: str, intent: CommIntent, work: ScheduledWork) -> None:
        intent.transition(IntentState.WAITING_FOR_ADMISSION)
        intent.transition(IntentState.ADMITTED)
        intent.transition(IntentState.SUBMITTED)
        timing = work.timing
        timing.admit_ts = now_us()
        timing.submit_ts = now_us()
        if intent.launch_fn is None:
            raise ValidationError(
                f"intent {intent.key.as_list()} has no launch_fn"
            )
        if self._comm_stream is not None:
            # M4：在显式 communication stream 上下文中发射。ready event 的
            # wait_event 落在该 stream 上，collective 经 c10d 捕获该 stream
            # 状态 -> producer stream -> comm stream 的依赖边（§4.5）。
            import torch

            with torch.cuda.stream(self._comm_stream):
                self._sync_ready_event(intent)
                underlying = intent.launch_fn()
        else:
            self._sync_ready_event(intent)
            underlying = intent.launch_fn()
        work.bind(underlying)
        self._inflight[g].append((work, timing))
        self._launched[g].append(intent.key)

    def _sync_ready_event(self, intent: CommIntent) -> None:
        """发射前让当前 stream 等待 producer 的 CUDA ready event（M3）。

        把 collective 排在 producer 的 GPU 工作之后（stream-ordered），
        而不是靠 CPU 阻塞；这样 stream dependency 由 GPU 侧 event 保证，
        在 profiler/Nsight 上可见为 wait_event -> collective 的依赖边。
        ``threading.Event`` 的 CPU ready 已在 ``_is_ready`` 处理，这里跳过。
        """
        ev = intent.ready_event
        if ev is None or isinstance(ev, threading.Event):
            return
        if hasattr(ev, "record") and hasattr(ev, "query"):  # torch.cuda.Event
            import torch  # 惰性导入，避免包在无 torch 环境下 import 失败

            torch.cuda.current_stream().wait_event(ev)

    def _on_work_complete(self, g: str, timing: CommTiming) -> None:
        """底层 Work 完成时由 ``ScheduledWork`` 回调：释放在途槽位并继续 drain。"""
        if self._worker_mode:
            with self._lock:
                self._inflight[g] = [
                    (w, t) for (w, t) in self._inflight[g] if t is not timing
                ]
            self._wake.set()  # 释放槽位后唤醒 worker 继续 drain
            return
        self._inflight[g] = [
            (w, t) for (w, t) in self._inflight[g] if t is not timing
        ]
        self._drain(g)
