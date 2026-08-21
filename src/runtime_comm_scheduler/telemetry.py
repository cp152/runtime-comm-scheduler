"""Runtime event schema placeholder."""

from dataclasses import dataclass
from typing import Optional

from .intent import TaskKey


@dataclass
class CommTiming:
    key: TaskKey
    ready_ts: Optional[int] = None
    admit_ts: Optional[int] = None
    submit_ts: Optional[int] = None
    complete_ts: Optional[int] = None
    predicted_duration_us: Optional[float] = None
    actual_duration_us: Optional[float] = None
