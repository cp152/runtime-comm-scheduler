"""M1 unit tests: plan hash + 五类校验（重复/缺失/metadata/hash/sequence）。"""

import pytest

from runtime_comm_scheduler import CommIntent, Plan, TaskKey
from runtime_comm_scheduler import validate as v
from runtime_comm_scheduler.validate import ValidationError


def _key(ordinal=0, process_group_id="dp", **kw):
    base = dict(
        iteration=0,
        microbatch=0,
        parallelism="dp",
        process_group_id=process_group_id,
        layer_id=0,
        bucket_id=0,
        ordinal=ordinal,
    )
    base.update(kw)
    return TaskKey(**base)


def _intent(ordinal=0, op="all_reduce", num_bytes=8, process_group_id="dp"):
    return CommIntent(
        key=_key(ordinal=ordinal, process_group_id=process_group_id),
        op=op,
        tensor=None,
        process_group=None,
        num_bytes=num_bytes,
    )


def _plan(entries, version=0, window_id=0):
    return Plan(version=version, window_id=window_id, entries=tuple(entries))


# --- plan hash ---


def test_plan_digest_stable_and_version_sensitive():
    p1 = _plan([(_key(0), "all_reduce", 8)])
    p2 = _plan([(_key(0), "all_reduce", 8)])
    assert p1.digest() == p2.digest()
    p3 = _plan([(_key(0), "all_reduce", 8)], version=1)
    assert p1.digest() != p3.digest()


def test_plan_digest_sensitive_to_order_and_metadata():
    base = _plan([(_key(0), "all_reduce", 8), (_key(1), "all_reduce", 8)])
    reordered = _plan([(_key(1), "all_reduce", 8), (_key(0), "all_reduce", 8)])
    assert base.digest() != reordered.digest()

    changed_op = _plan([(_key(0), "reduce_scatter", 8), (_key(1), "all_reduce", 8)])
    assert base.digest() != changed_op.digest()

    changed_bytes = _plan([(_key(0), "all_reduce", 16), (_key(1), "all_reduce", 8)])
    assert base.digest() != changed_bytes.digest()


# --- duplicate key ---


def test_duplicate_keys_detected():
    plan = _plan([(_key(0), "all_reduce", 8), (_key(0), "all_reduce", 8)])
    assert v.duplicate_keys(plan) == [_key(0)]
    with pytest.raises(ValidationError):
        v.check_plan_integrity(plan)


def test_no_duplicate_keys_when_clean():
    plan = _plan([(_key(0), "all_reduce", 8), (_key(1), "all_reduce", 8)])
    assert v.duplicate_keys(plan) == []
    v.check_plan_integrity(plan)  # 不抛


# --- missing / unexpected key ---


def test_missing_keys_detected():
    plan = _plan([(_key(0), "all_reduce", 8), (_key(1), "all_reduce", 8)])
    observed = [_key(0)]
    assert v.missing_keys(plan, observed) == [_key(1)]


def test_unexpected_keys_detected():
    plan = _plan([(_key(0), "all_reduce", 8)])
    observed = [_key(0), _key(1)]
    assert v.unexpected_keys(plan, observed) == [_key(1)]


# --- metadata mismatch ---


def test_metadata_mismatch_op_and_bytes():
    plan = _plan([(_key(0), "all_reduce", 8)])
    ok = _intent(0, op="all_reduce", num_bytes=8)
    assert v.metadata_mismatch(ok, plan) is None

    bad_op = _intent(0, op="all_gather", num_bytes=8)
    assert "op mismatch" in v.metadata_mismatch(bad_op, plan)

    bad_bytes = _intent(0, op="all_reduce", num_bytes=16)
    assert "num_bytes mismatch" in v.metadata_mismatch(bad_bytes, plan)


def test_validate_intent_raises_on_missing_key():
    plan = _plan([(_key(0), "all_reduce", 8)])
    with pytest.raises(ValidationError):
        v.validate_intent(_intent(1), plan)


# --- plan hash mismatch ---


def test_check_hash_match_raises_on_mismatch():
    a = _plan([(_key(0), "all_reduce", 8)])
    b = _plan([(_key(0), "all_reduce", 16)])
    with pytest.raises(ValidationError, match="plan hash mismatch"):
        v.check_hash_match(a, b)


# --- per-group sequence validation ---


def test_group_sequence_order_preserved():
    plan = _plan(
        [
            (_key(0, process_group_id="dp"), "all_reduce", 8),
            (_key(1, process_group_id="tp"), "all_reduce", 8),
            (_key(2, process_group_id="dp"), "all_reduce", 8),
        ]
    )
    assert plan.group_sequence("dp") == (
        _key(0, process_group_id="dp"),
        _key(2, process_group_id="dp"),
    )
    assert plan.group_sequence("tp") == (_key(1, process_group_id="tp"),)


def test_group_sequence_divergence_raises():
    a = _plan(
        [
            (_key(0, process_group_id="dp"), "all_reduce", 8),
            (_key(1, process_group_id="dp"), "all_reduce", 8),
        ]
    )
    b = _plan(
        [
            (_key(1, process_group_id="dp"), "all_reduce", 8),
            (_key(0, process_group_id="dp"), "all_reduce", 8),
        ]
    )
    with pytest.raises(ValidationError, match="diverged"):
        v.check_group_sequence_consistency("dp", a, b)


def test_validate_group_sequences_consistent():
    good = _plan([(_key(0), "all_reduce", 8), (_key(1), "all_reduce", 8)])
    v.validate_group_sequences([good, good])  # 不抛


def test_validate_group_sequences_raises_on_hash_mismatch():
    good = _plan([(_key(0), "all_reduce", 8)])
    bad = _plan([(_key(0), "all_reduce", 16)])
    with pytest.raises(ValidationError):
        v.validate_group_sequences([good, bad])
