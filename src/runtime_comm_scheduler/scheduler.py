"""AdmissionScheduler：训练线程内的同步 admission facade（M2）。

路径：``submit intent -> 校验 plan -> admission -> 调用原始 async
collective``。暂不引入 worker thread 或替代 CUDA stream；本类假定单线程
使用（训练线程），不加锁，M4 引入异步 worker 时再处理并发。

约束（architecture.md §5.3）：
1. 可以 delay task，但不能在 producer ready 前提交；
2. 不能在本地跳过或重排同一个 process group 的计划子序列；
3. outstanding collective 数量不超过上限；
4. collective 提交给底层后不可取消/重排。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from .intent import CommIntent, IntentState, TaskKey
from .plan import Plan
from .telemetry import CommTiming, now_us
from .validate import ValidationError, validate_intent
from .work import ScheduledWork


class AdmissionScheduler:
    def __init__(self, plan: Plan, *, max_outstanding: int = 0) -> None:
        """``max_outstanding=0`` 表示不限制同时在途的 collective 数量。"""
        self._plan = plan
        self._max_outstanding = max_outstanding
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

    # ---- 公共接口 --------------------------------------------------------

    def submit(self, intent: CommIntent) -> ScheduledWork:
        """提交 intent，返回延迟绑定的 ``ScheduledWork``。

        intent 必须与 plan 一致（key 存在、op/num_bytes 匹配）且未被提交过；
        违反任一条件即抛出 :class:`ValidationError`（fail-stop，而非静默）。
        返回值可能尚未发射：只有成为对应 group 的队首、producer ready 且
        outstanding 未达上限时才实际调用 ``launch_fn``。
        """
        validate_intent(intent, self._plan)
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
        self._drain(g)
        return work

    def nudge(self, group_id: Optional[str] = None) -> None:
        """重新检查 admission 条件（ready_event 就绪或 outstanding 释放后调用）。"""
        ids = (group_id,) if group_id is not None else self._group_ids
        for g in ids:
            self._drain(g)

    def close(self) -> None:
        """释放 scheduler 资源；M2 为同步 facade，无后台资源，仅作边界。"""

    # ---- 只读观察 -------------------------------------------------------

    def sequence_log(self) -> dict[str, list[TaskKey]]:
        """每个 process group 已发射 key 的序列（跨 rank 对比的验收对象）。"""
        return {g: list(keys) for g, keys in self._launched.items()}

    def timings(self) -> list[CommTiming]:
        """按提交顺序返回所有 intent 的时序记录。"""
        return list(self._timings)

    # ---- 内部状态机 ------------------------------------------------------

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
        return True  # CUDA event 的 ready 语义在 M3 实现

    def _drain(self, g: str) -> None:
        """尽可能把队首且已 ready 的 pending intent 发射出去。"""
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
        underlying = intent.launch_fn()
        work.bind(underlying)
        self._inflight[g].append((work, timing))
        self._launched[g].append(intent.key)

    def _on_work_complete(self, g: str, timing: CommTiming) -> None:
        """底层 Work 完成时由 ``ScheduledWork`` 回调：释放在途槽位并继续 drain。"""
        self._inflight[g] = [
            (w, t) for (w, t) in self._inflight[g] if t is not timing
        ]
        self._drain(g)
