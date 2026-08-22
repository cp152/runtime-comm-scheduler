# Runtime Communication Scheduler

面向单个混合并行训练任务的 runtime-adaptive collective scheduling 研究原型。项目研究如何在训练框架、PyTorch distributed runtime、NCCL 和网络之间建立可逐步下沉的 schedule layer，使训练 DAG 语义能够影响真实 collective 的执行。

本项目与 `SimAI/simai-flow-scheduler` 互补：SimAI 已用于 flow-level trace replay、带宽分配和 policy 验证；本项目研究真实 training stack 中究竟能够观察、控制和安全重排哪些通信事件。

## 项目范围与长期目标

当前聚焦单个 dense Megatron-style training job。DP、TP、PP 都可以作为训练语义来源，但第一个真实集成点只处理 DP gradient synchronization。multi-tenant 暂不纳入项目接口和实现，未来如有需要再扩展。

长期目标是支持从 high-level collective admission 到更细粒度 data-plane control 的演进：

```text
framework semantic scheduling
  -> ProcessGroup-level enforcement
  -> NCCL communicator/channel/chunk control
  -> NIC/network traffic control
```

当前不预设必须在哪一层结束。每次向下扩展都应由上一层机制不足以实现目标 policy 的证据驱动。

## 整体架构

```text
训练框架语义适配层
  - 识别 layer、microbatch、DP/TP/PP 角色、producer 和 consumer
  - 绑定本地 tensor、CUDA ready event 和原始 collective launcher
                         |
                         v
每 rank 的 schedule layer
  - 根据共享执行计划校验 CommIntent
  - 执行 runtime admission、delay 和安全的跨 process group 发射仲裁
  - 返回 ScheduledWork 并记录 telemetry
                         |
                         v
ProcessGroup / NCCL 数据面
  - 执行已经通过 admission 的 collective
  - 初期完全复用原始 ProcessGroupNCCL 和 NCCL
  - 后续可成为更细粒度控制的修改对象
                         |
                         v
GPU 互连 / NIC / 网络
```

前两层有意分离：训练框架是唯一掌握 training-DAG 语义的层级；schedule layer 是实际 gate collective submission 的执行点。

## Plan + Admission 调度模型

调度器不依赖单一 priority queue，而是结合两个互补机制。

### Plan：基础执行计划

Plan 是一个概念上的、带版本的基础执行计划，描述一个 scheduling window（初期为一个 iteration）的预期执行结构。它由 `CommIntent` 的确定性元数据和有序的 `TaskKey` 序列组成，不额外引入独立的 intent 类型：

- 每个预期 collective 的稳定 `TaskKey`；
- 每个 rank 上的有序 task 序列；
- 每个 process group 诱导出的 collective 子序列；
- collective 类型、字节数、训练语义和预测时延等 `CommIntent` 元数据。

Plan 在各 rank 间共享，初期只允许在 iteration 之间的安全边界切换。它建立了保证 collective 正确性所需的顺序契约。

### Dynamic Admission：运行时准入

运行时，训练框架在每个 rank 创建本地 `CommIntent`，将计划中的 `TaskKey` 绑定到实际 tensor、CUDA ready event 和原始 `torch.distributed` launcher。Admission 层可以：

- 等待 producer ready；
- 延迟计划中的 collective；
- 限制 outstanding collective 数量；
- 在不同 process group 之间选择 ready task 的发射时机；
- 记录运行时测量，为下一版本 plan 提供数据。

Admission 不能在本地独立改变同一个 process group 内的 collective 顺序。只有在每个受影响 process group 的诱导序列仍然对其所有成员 rank 一致时，才允许进行跨 group 仲裁。

## 任务生命周期

```text
CommIntent
  -> READY
  -> WAITING_FOR_ADMISSION
  -> ADMITTED
  -> SUBMITTED
  -> COMPLETED
```

`READY`、`SUBMITTED` 和 `COMPLETED` 是不同事件。scheduler 只能在 `SUBMITTED` 之前介入；collective 一旦提交给 NCCL，就不能在当前层级取消或抢占。

## 目录结构

- `docs/architecture.md`：长期维护的系统边界、事件模型、机制和正确性约束。
- `docs/phase1-plan.md`：当前 Phase 1 的目标、设计、里程碑和验收标准。
- `docs/experiments/`：实验记录和 profiler/Nsight 产物索引。
- `src/runtime_comm_scheduler/`：机制接口和后续实现。
- `src/runtime_comm_scheduler/adapters/`：框架语义适配器，首先适配 Megatron。
- `tests/`：单元测试以及 Gloo/NCCL distributed harness。
- `examples/`：最小可运行示例。
- `configs/`：预留给后续实验配置。

## 当前状态

- **M0（Gloo 行为 harness）已完成**：两 rank 场景覆盖 FIFO、固定重排、延迟
  ready 与 divergence 挂起，记录见 [docs/experiments/m0-m4.md](docs/experiments/m0-m4.md)。
- **M1（核心 schema 与 plan 校验）已完成**：确定性 `TaskKey`、`CommIntent`
  生命周期、`Plan` 表示与五类校验。
- **M2（V0 同步 admission）已完成**：`AdmissionScheduler` 的 submit → 校验 →
  准入 → 发射路径，两 rank Gloo harness 重现 FIFO 与固定重排且 sequence log
  一致，乱序提交被强制为计划顺序，错误场景 fail-stop 有界退出。
- **待开发**：NCCL/CUDA event、异步 admission worker、Megatron DP adapter 与
  plan 版本切换，详见 [docs/phase1-plan.md](docs/phase1-plan.md)。
