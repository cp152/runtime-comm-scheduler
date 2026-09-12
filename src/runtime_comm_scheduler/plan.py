"""Plan 的轻量表示与确定性 hash。

Plan 是带版本的概念性共享执行计划，由确定性 ``CommIntent`` 元数据和有序
``TaskKey`` 序列构成。plan entry 是 ``(TaskKey, op, num_bytes)`` 三元组的元数据，
``TaskKey`` 中已包含 parallelism / process_group_id 等身份字段。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Collection, Iterable

from .intent import CommIntent, TaskKey, stable_dumps

#: 每个 plan entry 是确定性元数据三元组：key、collective/op 类型、字节数。
#: 不使用 ``CommIntent`` 本体，因为其 tensor/launcher/event 是仅本 rank 的
#: 运行时对象，不能进入可跨 rank 广播的 plan。
PlanEntry = tuple[TaskKey, str, int]


@dataclass(frozen=True)
class Plan:
    """带版本、带窗口的共享规范文档，不是每 rank 的执行计划。

    ``entries`` 是有序的共享载体；同一 process group 的诱导子序列由
    :meth:`group_sequence` 按 ``process_group_id`` 过滤得到，是唯一跨 rank
    一致的顺序不变量。每 rank 的实际执行序列是运行时投影，不存入 plan。
    :meth:`digest` 的语义是文档身份，不是执行顺序。
    """

    version: int
    window_id: int
    entries: tuple[PlanEntry, ...]

    @property
    def keys(self) -> tuple[TaskKey, ...]:
        return tuple(e[0] for e in self.entries)

    def metadata(self, key: TaskKey) -> tuple[str, int] | None:
        """返回 plan 中 ``key`` 对应的 (op, num_bytes)，缺失时返回 None。"""
        for k, op, num_bytes in self.entries:
            if k == key:
                return (op, num_bytes)
        return None

    def group_sequence(self, group_id: str) -> tuple[TaskKey, ...]:
        """给定 process group 的诱导子序列（保持 plan 全局顺序）。"""
        return tuple(
            k for (k, _op, _nb) in self.entries if k.process_group_id == group_id
        )

    def group_ids(self) -> tuple[str, ...]:
        """按首次出现顺序返回所有 process group id。"""
        seen: list[str] = []
        for k, _op, _nb in self.entries:
            if k.process_group_id not in seen:
                seen.append(k.process_group_id)
        return tuple(seen)

    def local_projection(
        self,
        local_group_ids: Collection[str],
    ) -> tuple[TaskKey, ...]:
        """返回共享 plan 在当前 rank 所参与 group 上的有序投影。

        不参与某个 process group 的 rank 直接跳过其 task，不创建占位
        collective。返回序列保持 ``entries`` 的全局 host launch 顺序。
        """
        groups = frozenset(local_group_ids)
        return tuple(k for k in self.keys if k.process_group_id in groups)

    def digest(self) -> str:
        """plan 的确定性 hash，覆盖 version、window_id 与有序 entry 元数据。"""
        payload = [
            self.version,
            self.window_id,
            [[k.as_list(), op, nb] for (k, op, nb) in self.entries],
        ]
        return hashlib.sha256(stable_dumps(payload).encode("utf-8")).hexdigest()


def plan_from_intents(
    version: int,
    window_id: int,
    intents: Iterable[CommIntent],
) -> Plan:
    """从已排序的 intent 序列构建 Plan，只保留确定性元数据。"""
    return Plan(
        version=version,
        window_id=window_id,
        entries=tuple((i.key, i.op, i.num_bytes) for i in intents),
    )
