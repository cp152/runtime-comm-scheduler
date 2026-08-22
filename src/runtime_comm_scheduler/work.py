"""ScheduledWork：延迟绑定的分布式 Work 兼容边界。

M2 的创建时机在 admission 之前：intent 提交时立刻返回 ``ScheduledWork``，
但底层 collective 可能尚未发射（parked）。发射时由 scheduler 调用
:meth:`bind` 把原始 ``Work`` 挂上来，``wait``/``is_completed`` 再透传到
底层 Work。完成时记录 telemetry 并把 intent 推进到 ``COMPLETED``。
"""

from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any, Callable, Optional

from .intent import CommIntent, IntentState
from .telemetry import CommTiming, now_us


class ScheduledWork:
    """延迟绑定的底层分布式 Work 的包装边界。"""

    def __init__(
        self,
        intent: CommIntent,
        timing: CommTiming,
        *,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self._intent = intent
        self._timing = timing
        self._on_complete = on_complete
        self._underlying: Optional[Any] = None
        self._launched = threading.Event()
        self._lock = threading.Lock()
        self._completed = False

    @property
    def key(self):
        return self._intent.key

    @property
    def timing(self) -> CommTiming:
        return self._timing

    def bind(self, underlying: Any) -> None:
        """把已发射 collective 返回的底层 ``Work`` 绑定到本包装对象。"""
        with self._lock:
            if self._underlying is not None:
                raise RuntimeError(f"ScheduledWork {self._intent.key} already bound")
            self._underlying = underlying
        self._launched.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待发射与完成；``timeout`` 期间未发射或未完成时返回 False。

        ``timeout=None`` 表示无限等待——对应 M0 观察到的缺失 collective
        挂起语义，但在这里是 scheduler 层的有界契约，由 harness 的超时兜底。
        """
        if not self._launched.wait(timeout):
            return False
        with self._lock:
            underlying = self._underlying
        if underlying is None:
            return False
        # c10d Work.wait 只接受 timedelta；None 表示无限等待（不传参）。
        if timeout is None:
            done = underlying.wait()
        else:
            done = underlying.wait(timeout=timedelta(seconds=timeout))
        if done:
            self._mark_completed()
        return done

    def is_completed(self) -> bool:
        """已发射且底层 Work 已完成（未发射时恒为 False）。"""
        with self._lock:
            if self._underlying is None:
                return False
            return bool(self._underlying.is_completed())

    def get_future(self):
        """M4 引入 future/异步完成通知之前不可用。"""
        raise NotImplementedError("ScheduledWork future is deferred to M4")

    def _mark_completed(self) -> None:
        with self._lock:
            if self._completed:
                return
            self._completed = True
        self._timing.complete_ts = now_us()
        if (
            self._timing.submit_ts is not None
            and self._timing.complete_ts is not None
        ):
            self._timing.actual_duration_us = float(
                self._timing.complete_ts - self._timing.submit_ts
            )
        self._intent.transition(IntentState.COMPLETED)
        if self._on_complete is not None:
            self._on_complete()
