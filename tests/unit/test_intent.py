"""M1 unit tests: TaskKey determinism, CommIntent lifecycle."""

import pytest

from runtime_comm_scheduler import CommIntent, IntentState, TaskKey
from runtime_comm_scheduler.intent import stable_dumps


def _key(**overrides):
    base = dict(
        iteration=0,
        microbatch=0,
        parallelism="dp",
        process_group_id="dp_group",
        layer_id=0,
        bucket_id=0,
        ordinal=0,
    )
    base.update(overrides)
    return TaskKey(**base)


def test_task_key_digest_is_deterministic():
    k1 = _key()
    k2 = _key()
    assert k1.digest() == k2.digest()
    # 同一 key 重复计算稳定
    assert k1.digest() == k1.digest()


def test_task_key_distinguishes_every_field():
    fields = ["iteration", "microbatch", "parallelism", "process_group_id",
              "layer_id", "bucket_id", "ordinal"]
    base = _key().digest()
    for f in fields:
        overrides = {f: _key().__dict__[f]}
        # 逐个字段翻转，应产生不同的 digest
        if isinstance(overrides[f], int):
            overrides[f] += 1
        else:
            overrides[f] += "_x"
        assert _key(**overrides).digest() != base, f"field {f} not hashed"


def test_task_key_equality_and_hash():
    assert _key() == _key()
    assert hash(_key()) == hash(_key())
    assert _key(ordinal=1) != _key(ordinal=0)


def test_stable_dumps_sorts_dict_keys():
    assert stable_dumps({"b": 1, "a": 2}) == stable_dumps({"a": 2, "b": 1})


def test_lifecycle_forward_transitions():
    intent = CommIntent(key=_key(), op="all_reduce", tensor=None,
                        process_group=None, num_bytes=8)
    path = [
        IntentState.READY,
        IntentState.WAITING_FOR_ADMISSION,
        IntentState.ADMITTED,
        IntentState.SUBMITTED,
        IntentState.COMPLETED,
    ]
    for s in path:
        intent.transition(s)
    assert intent.state == IntentState.COMPLETED


@pytest.mark.parametrize(
    "bad_state",
    [
        IntentState.COMPLETED,  # CREATED -> COMPLETED 跳跃
        IntentState.CREATED,     # 回退
        IntentState.SUBMITTED,   # CREATED -> SUBMITTED 跳跃
    ],
)
def test_lifecycle_rejects_illegal_transition(bad_state):
    intent = CommIntent(key=_key(), op="all_reduce", tensor=None,
                        process_group=None, num_bytes=8)
    with pytest.raises(ValueError):
        intent.transition(bad_state)


def test_lifecycle_no_transition_after_completed():
    intent = CommIntent(key=_key(), op="all_reduce", tensor=None,
                        process_group=None, num_bytes=8)
    for s in [
        IntentState.READY,
        IntentState.WAITING_FOR_ADMISSION,
        IntentState.ADMITTED,
        IntentState.SUBMITTED,
        IntentState.COMPLETED,
    ]:
        intent.transition(s)
    # COMPLETED 是终态
    with pytest.raises(ValueError):
        intent.transition(IntentState.SUBMITTED)


@pytest.mark.parametrize(
    "path",
    [
        [],
        [IntentState.READY],
        [IntentState.READY, IntentState.WAITING_FOR_ADMISSION],
        [
            IntentState.READY,
            IntentState.WAITING_FOR_ADMISSION,
            IntentState.ADMITTED,
        ],
        [
            IntentState.READY,
            IntentState.WAITING_FOR_ADMISSION,
            IntentState.ADMITTED,
            IntentState.SUBMITTED,
        ],
    ],
)
def test_lifecycle_can_fail_from_every_nonterminal_stage(path):
    intent = CommIntent(
        key=_key(), op="all_reduce", tensor=None, process_group=None, num_bytes=8
    )
    for state in path:
        intent.transition(state)
    intent.transition(IntentState.FAILED)
    assert intent.state is IntentState.FAILED
    with pytest.raises(ValueError):
        intent.transition(IntentState.COMPLETED)
