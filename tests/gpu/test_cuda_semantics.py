"""M3 GPU tests: CUDA event ready 语义与 wait() 不提前返回。

依赖 CUDA，无 CUDA 时整体 skip。用 world_size=1 的 NCCL 进程组在单卡上
直接验证 scheduler 的 stream-dependency 机制（真实 NCCL 两 rank 行为由
``examples/run_m3.py`` 的 harness 覆盖）。

测试不依赖竞态数据结果：world_size=1 的 trivial all_reduce 实测恒在
producer fill 之后执行，无法用"读 sentinel"证明无依赖。改为用事件触发
顺序确定性验证依赖边是否存在（带 event -> 当前 stream 被阻塞；不带 ->
立即推进），数据竞态由两 rank harness 的 ``no_wait_ready`` 覆盖。

两个前提（均为 M3 实测）：
- 默认（legacy）stream 会与所有其他 stream 隐式同步，掩盖 wait_event 依赖；
  测试用非 legacy 显式流模拟真实场景（collective 走独立 comm stream，无
  隐式同步）。
- NCCL 首 op 含 ~350ms lazy init，会掩盖依赖边；fixture 先预热 communicator。
"""

from __future__ import annotations

import pytest

import torch
import torch.distributed as dist

from runtime_comm_scheduler import AdmissionScheduler, CommIntent, Plan, TaskKey

_N = 1024


def _key(ordinal=0):
    return TaskKey(0, 0, "dp", "dp", 0, 0, ordinal)


def _plan(keys):
    return Plan(version=0, window_id=0,
                entries=tuple((k, "all_reduce", _N * 4) for k in keys))


@pytest.fixture(scope="module")
def pg():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0
    )
    torch.cuda.set_device(0)
    # 预热 NCCL communicator：首 op 的 lazy init 会掩盖依赖边。
    warm = torch.ones(1024, device="cuda")
    dist.all_reduce(warm, async_op=True).wait()
    yield
    dist.destroy_process_group()


def test_scheduler_waits_cuda_ready_event(pg):
    """带 ready event：scheduler 发射前让当前 stream wait_event，collective
    排在慢 producer 之后，数据正确。依赖边用事件触发顺序验证：submit 后
    当前 stream 被阻塞（after_ev 未触发），wait 后已触发。"""
    k = _key()
    tensor = torch.full((_N,), -1.0, device="cuda")  # sentinel：未就绪
    prod_stream = torch.cuda.Stream()
    ready_ev = torch.cuda.Event()
    with torch.cuda.stream(prod_stream):
        a = torch.randn(2048, 2048, device="cuda")
        for _ in range(30):
            a = a @ a  # 慢 producer（~30ms）
        tensor.fill_(1.0)
        ready_ev.record(prod_stream)  # fill 完成才触发

    intent = CommIntent(
        key=k, op="all_reduce", tensor=tensor, process_group=None,
        num_bytes=_N * 4,
        launch_fn=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True),
        ready_event=ready_ev,
    )
    sched = AdmissionScheduler(_plan([k]))
    # 非 legacy 显式流：默认流会与 producer 流隐式同步，掩盖依赖边。
    sched_stream = torch.cuda.Stream()
    with torch.cuda.stream(sched_stream):
        work = sched.submit(intent)
        after_ev = torch.cuda.Event()
        after_ev.record(sched_stream)  # 排到 wait_event 依赖之后
        assert not after_ev.query()    # 依赖边：当前 stream 被 ready_ev 阻塞
    assert work.wait()
    assert after_ev.query()            # wait() 后 producer 已结束，事件触发
    # 依赖正确：collective 排在 producer 之后，读到 1.0 而不是 sentinel。
    assert torch.all(tensor == 1.0).item()


def test_without_ready_event_no_stream_dependency(pg):
    """对照：不带 ready event 时 scheduler 不插入 stream 依赖，submit 后
    当前 stream 立即推进（after_ev 立即触发），collective 不被 producer 阻塞。"""
    k = _key()
    tensor = torch.full((_N,), -1.0, device="cuda")
    prod_stream = torch.cuda.Stream()
    with torch.cuda.stream(prod_stream):
        a = torch.randn(2048, 2048, device="cuda")
        for _ in range(30):
            a = a @ a
        tensor.fill_(1.0)              # 不 record ready event

    intent = CommIntent(
        key=k, op="all_reduce", tensor=tensor, process_group=None,
        num_bytes=_N * 4,
        launch_fn=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True),
    )
    sched = AdmissionScheduler(_plan([k]))
    sched_stream = torch.cuda.Stream()
    with torch.cuda.stream(sched_stream):
        work = sched.submit(intent)
        after_ev = torch.cuda.Event()
        after_ev.record(sched_stream)  # 当前 stream 无阻塞
        assert after_ev.query()        # 无依赖：立即推进
    assert work.wait()


def test_wait_does_not_return_early(pg):
    """wait() 不提前返回：提交后 GPU 尚未完成时 completion event 未触发；
    即使 submit 立即返回，wait() 也会阻塞到 GPU 工作真正完成。"""
    k = _key()
    tensor = torch.full((_N,), 1.0, device="cuda")
    comp_ev = torch.cuda.Event()

    def launch():
        # 先在当前 stream 排一段慢 GPU 工作（约 30ms），使提交后、wait 前
        # 的查询确定性地落在 GPU 忙碌期（world_size=1 的 all_reduce 本身太快）。
        a = torch.randn(2048, 2048, device="cuda")
        for _ in range(30):
            a = a @ a
        w = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        comp_ev.record(torch.cuda.current_stream())  # 排到 collective 之后
        return w

    intent = CommIntent(
        key=k, op="all_reduce", tensor=tensor, process_group=None,
        num_bytes=_N * 4, launch_fn=launch,
    )
    sched = AdmissionScheduler(_plan([k]))
    work = sched.submit(intent)
    assert not comp_ev.query()     # submit 立即返回，但 GPU 工作未完成
    assert work.wait()
    # M3 契约：wait() 返回即保证 GPU 工作完成，事件无需再额外同步就已触发
    # （本环境裸 WorkNCCL.wait() 会在 GPU 完成前返回，ScheduledWork 内补同步）。
    assert comp_ev.query()
