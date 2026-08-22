"""通信事件时序 schema 与单调时钟。

M2 起 scheduler 在 submit/admit/submit-complete 边界记录微秒时间戳；
``now_us`` 是单调时钟（不受系统时间调整影响），供调度与实验对比使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .intent import TaskKey


def now_us() -> int:
    """返回单调时钟的微秒时间戳（跨 rank 不可比，仅本 rank 相对比较）。"""
    return time.perf_counter_ns() // 1000


@dataclass
class CommTiming:
    key: TaskKey
    ready_ts: Optional[int] = None
    admit_ts: Optional[int] = None
    submit_ts: Optional[int] = None
    complete_ts: Optional[int] = None
    predicted_duration_us: Optional[float] = None
    actual_duration_us: Optional[float] = None
