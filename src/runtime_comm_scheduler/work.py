"""ScheduledWork：延迟绑定的分布式 Work 兼容边界。

M2 的创建时机在 admission 之前：intent 提交时立刻返回 ``ScheduledWork``，
但底层 collective 可能尚未发射（parked）。发射时由 scheduler 调用
:meth:`bind` 把原始 ``Work`` 挂上来，``wait``/``is_completed`` 再透传到
底层 Work。完成时记录 telemetry 并把 intent 推进到 ``COMPLETED``。

M3 起 ``wait`` 对 c10d Work 在底层 wait 后补一次设备同步（见
:func:`_ensure_gpu_complete`）：本环境实测裸 ``WorkNCCL.wait()`` 会在
collective 的 GPU 工作完成前就返回，只透传无法兑现"不提前返回"的契约。
"""

from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any, Callable, Optional

from .intent import CommIntent, IntentState
from .telemetry import CommTiming, now_us


def _ensure_gpu_complete(underlying: Any) -> None:
    """补偿 c10d Work 在 NCCL 下不可靠的完成语义（M3 发现）。

    torch 2.12.1+cu130 / NCCL comm stream 上实测：裸 ``WorkNCCL.wait()``
    对 4GB all_reduce 也在 ~0.01ms 内返回，而 GPU 工作要等后续
    ``torch.cuda.synchronize()`` 才排空（~570ms）。即 wait 返回不意味着
    collective 完成。这里对真正的 c10d Work 补一次设备同步，保证
    ``ScheduledWork.wait()`` 的完成信号真实反映 GPU 完成。
    代价是每次 wait 全设备同步；M4 换成 comm-stream completion event 消除。
    非 c10d Work（测试里的 FakeWork 等）或未启用 CUDA 时不做任何事。
    """
    try:
        import torch  # 惰性导入，包不强制依赖 torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    mod = type(underlying).__module__ or ""
    # c10d 的 Work（NCCL/Gloo）定义在 torch._C._distributed_c10d 下。
    if not (mod.startswith("torch._C") or mod.startswith("torch.distributed")):
        return
    torch.cuda.synchronize()


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
            # 底层 wait 返回不等于 GPU 完成（M3 发现），先补同步再标记完成。
            _ensure_gpu_complete(underlying)
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
