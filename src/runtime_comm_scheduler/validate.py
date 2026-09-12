"""Plan 与 CommIntent 的校验。

M1 验收覆盖五类问题：重复 key、缺失 key、metadata mismatch、plan hash
mismatch、per-group sequence validation。所有校验函数在发现不一致时显式
抛出 :class:`ValidationError`，而不是静默继续——这是正确性模型的核心
要求（docs/design/architecture.md §6：主动暴露 divergence，而非等待 NCCL 超时）。
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .intent import CommIntent, TaskKey
from .plan import Plan


class ValidationError(Exception):
    """collective 序列与当前 plan 不一致时抛出。"""


def check_plan_integrity(plan: Plan) -> None:
    """校验 plan 自身：不允许重复 key。"""
    duplicates = duplicate_keys(plan)
    if duplicates:
        raise ValidationError(
            f"plan v{plan.version} contains duplicate keys: "
            f"{[k.as_list() for k in duplicates]}"
        )


def duplicate_keys(plan: Plan) -> list[TaskKey]:
    """返回 plan 中出现多于一次的 key（按首次出现顺序）。"""
    counts = Counter(plan.keys)
    seen: set[TaskKey] = set()
    dupes: list[TaskKey] = []
    for k in plan.keys:
        if counts[k] > 1 and k not in seen:
            seen.add(k)
            dupes.append(k)
    return dupes


def missing_keys(plan: Plan, observed: Iterable[TaskKey]) -> list[TaskKey]:
    """返回 plan 中预期但未被 ``observed`` 提交的 key（保持 plan 顺序）。"""
    observed_set = set(observed)
    return [k for k in plan.keys if k not in observed_set]


def unexpected_keys(plan: Plan, observed: Iterable[TaskKey]) -> list[TaskKey]:
    """返回 ``observed`` 中出现但 plan 中没有的 key。"""
    plan_set = set(plan.keys)
    return [k for k in observed if k not in plan_set]


def metadata_mismatch(intent: CommIntent, plan: Plan) -> str | None:
    """返回 metadata 不匹配的描述；一致时返回 None。"""
    m = plan.metadata(intent.key)
    if m is None:
        return f"key {intent.key.as_list()} not in plan v{plan.version}"
    op, num_bytes = m
    if intent.op != op:
        return (
            f"op mismatch for {intent.key.as_list()}: "
            f"intent={intent.op!r} plan={op!r}"
        )
    if intent.num_bytes != num_bytes:
        return (
            f"num_bytes mismatch for {intent.key.as_list()}: "
            f"intent={intent.num_bytes} plan={num_bytes}"
        )
    return None


def validate_intent(intent: CommIntent, plan: Plan) -> None:
    """校验单个 intent 与 plan 一致：key 存在且 metadata 匹配。"""
    err = metadata_mismatch(intent, plan)
    if err is not None:
        raise ValidationError(err)


def check_hash_match(a: Plan, b: Plan) -> None:
    """校验两个 plan 的 hash 一致；不一致时抛出。"""
    if a.digest() != b.digest():
        raise ValidationError(
            f"plan hash mismatch: v{a.version} {a.digest()} vs "
            f"v{b.version} {b.digest()}"
        )


def check_group_sequence_consistency(
    group_id: str,
    a: Plan,
    b: Plan,
) -> None:
    """校验两个 plan 在 ``group_id`` 上的诱导子序列一致。

    这是跨 rank 一致性的核心：同一 process group 内所有成员 rank 必须
    看到相同的 collective 顺序。
    """
    seq_a = a.group_sequence(group_id)
    seq_b = b.group_sequence(group_id)
    if seq_a != seq_b:
        raise ValidationError(
            f"group {group_id!r} sequence diverged:\n"
            f"  a={[k.as_list() for k in seq_a]}\n"
            f"  b={[k.as_list() for k in seq_b]}"
        )


def validate_group_sequences(plans: Sequence[Plan]) -> None:
    """校验多个 rank 的 plan 在每个 process group 上的诱导序列两两一致。

    ``plans`` 是各 rank 安装的（本应相同的）plan；第一个作为基准。
    """
    if len(plans) < 2:
        return
    base = plans[0]
    # 覆盖所有 group（以基准 plan 的 group 集合为准，并集则更严格）。
    group_ids: set[str] = set()
    for p in plans:
        group_ids.update(p.group_ids())
    for other in plans[1:]:
        check_hash_match(base, other)
        for gid in group_ids:
            check_group_sequence_consistency(gid, base, other)
