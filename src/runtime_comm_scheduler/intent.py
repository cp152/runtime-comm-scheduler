"""通信任务语义接口与确定性身份。

plan 是由 ``CommIntent`` 的确定性元数据和 ``TaskKey`` 序列构成的概念。本模块包含 schema、确定性
hash 以及任务生命周期状态。

``TaskKey`` 的字段是所有 rank 都能本地生成、且必须确定性的：禁止使用
object identity、指针、随机 ID、tensor address 或本地 timestamp。
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def stable_dumps(obj: Any) -> str:
    """返回跨进程稳定、可复现的 JSON 序列化。

    使用 ``sort_keys`` 与 ``separators`` 消除 dict 键序与默认空格带来的
    差异；对象必须只包含可 JSON 化的确定性字段。
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class IntentState(enum.Enum):
    """任务生命周期状态。

    ``CREATED -> READY -> WAITING_FOR_ADMISSION -> ADMITTED
    -> SUBMITTED -> COMPLETED``。``SUBMITTED`` 是不可逆边界。
    """

    CREATED = "created"
    READY = "ready"
    WAITING_FOR_ADMISSION = "waiting_for_admission"
    ADMITTED = "admitted"
    SUBMITTED = "submitted"
    COMPLETED = "completed"


# 允许的单调前向转移。状态只能前进，不能回退（``SUBMITTED`` 之后不可取消）。
_TRANSITIONS: dict[IntentState, tuple[IntentState, ...]] = {
    IntentState.CREATED: (IntentState.READY,),
    IntentState.READY: (IntentState.WAITING_FOR_ADMISSION,),
    IntentState.WAITING_FOR_ADMISSION: (IntentState.ADMITTED,),
    IntentState.ADMITTED: (IntentState.SUBMITTED,),
    IntentState.SUBMITTED: (IntentState.COMPLETED,),
    IntentState.COMPLETED: (),
}


@dataclass(frozen=True)
class TaskKey:
    """一个逻辑 collective 的确定性身份（跨 rank 一致）"""

    iteration: int          # 训练 iteration 编号
    microbatch: int         # 当前 microbatch 编号
    parallelism: str        # DP/TP/PP 等训练语义类别
    process_group_id: str   # 稳定逻辑 process group 标识，决定顺序一致性边界
    layer_id: int           # collective 对应的逻辑 layer
    bucket_id: int          # layer 内 gradient/parameter bucket 编号
    ordinal: int            # 其他字段相同时的重复 collective 稳定序号

    def as_list(self) -> list[Any]:
        """按字段顺序展平为可 JSON 化的列表，用于确定性 hash。"""
        return [
            self.iteration,
            self.microbatch,
            self.parallelism,
            self.process_group_id,
            self.layer_id,
            self.bucket_id,
            self.ordinal,
        ]

    def canonical(self) -> str:
        """跨 rank 一致的确定性序列化。"""
        return stable_dumps(self.as_list())

    def digest(self) -> str:
        """该 key 的确定性 hash（hex）。"""
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


@dataclass
class CommIntent:
    """运行时绑定的通信意图，同时承载可进入 plan 的语义元数据。

    可进入 plan 的共享部分：``key``、``op``、``num_bytes``（以及
    ``key`` 中已含的 parallelism / process_group_id 等）。仅本 rank 的
    运行时绑定：``tensor``、``process_group``、``launch_fn``、
    ``ready_event``，这些不能作为 plan 内容广播。
    """

    key: TaskKey                                        # 对应的 TaskKey（跨 rank 一致）
    op: str                                             # collective 类型：all_reduce / reduce_scatter 等
    tensor: Any                                         # 本 rank 的实际通信 tensor
    process_group: Any                                  # 本 rank 的实际 PyTorch ProcessGroup handle
    num_bytes: int                                      # 通信数据量，用于准入和 telemetry
    launch_fn: Optional[Callable[..., Any]] = None      # admission 后调用原始 collective 的函数
    producer: Optional[str] = None                      # 训练 DAG 生产者描述（可选）
    consumer: Optional[str] = None                      # 训练 DAG 消费者描述（可选）
    ready_event: Any = None                             # CUDA ready event，表示 tensor 已可通信
    state: IntentState = field(default=IntentState.CREATED)

    def transition(self, new_state: IntentState) -> None:
        """把 intent 推进到 ``new_state``，拒绝非法（回退/跳跃）转移。"""
        allowed = _TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"illegal state transition {self.state.name} -> "
                f"{new_state.name}; allowed={[s.name for s in allowed]}"
            )
        self.state = new_state
