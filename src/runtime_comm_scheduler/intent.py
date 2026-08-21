"""通信任务语义接口骨架。

plan 是由 CommIntent 的确定性元数据和 TaskKey 序列构成的概念，不额外
引入独立的 planned-intent 运行时对象。本模块只包含 schema；M0 通过后
才开始加入 scheduler 行为。
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class TaskKey:
    iteration: int
    microbatch: int
    parallelism: str
    process_group_id: str
    layer_id: int
    bucket_id: int
    ordinal: int


@dataclass
class CommIntent:
    """运行时绑定的通信意图，同时承载可进入 plan 的语义元数据。"""

    key: TaskKey
    op: str
    tensor: Any
    process_group: Any
    num_bytes: int
    launch_fn: Optional[Callable[..., Any]] = None
    producer: Optional[str] = None
    consumer: Optional[str] = None
    ready_event: Any = None
