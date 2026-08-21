# Phase 1 计划：Collective Plan 与 Dynamic Admission

更新时间：2026-08-21

## 1. Phase 目标

Phase 1 的目标是在真实 PyTorch distributed stack 中建立第一个可运行的 schedule layer。它是机制阶段，不是 policy 阶段。

本阶段需要证明：训练框架可以将带训练语义的 `CommIntent` 交给每 rank 的 scheduler；scheduler 可以在调用原始 collective 前，根据共享的基础 plan 实施安全的 runtime admission；系统可以返回正确的 `ScheduledWork` 并记录 telemetry。

本阶段暂不追求 iteration time 优化。预期产物是一个正确、可观测、可关闭并可回退的机制，后续 runtime-adaptive policy 将建立在它之上。

## 2. 范围

### 包含

- 单个 dense training job；
- 每个 scheduling window 的共享、带版本基础 plan；
- 本地 `CommIntent` 创建和校验；
- runtime admission、delay 和 outstanding work 上限；
- 在不破坏 per-process-group 顺序的前提下进行跨 group 发射仲裁；
- 使用原始 Gloo/NCCL collective；
- `ScheduledWork`、CUDA event 和 telemetry 机制；
- 最后接入 Megatron DP gradient synchronization。

### 不包含

- 具体 criticality、slack 或 flow-priority policy；
- multi-tenant admission、fairness 或 quota；
- collective 内部 chunk 调度或抢占；
- packet/flow 可见性或带宽分配；
- 修改 PyTorch、ProcessGroup、NCCL 或网络源码；
- 本阶段接入 TP、PP、MoE、FSDP 或 ZeRO。

## 3. Phase 架构

```text
iteration 之间的共享 plan 路径
  runtime telemetry -> 生成/更新 plan -> 各 rank 安装同一版本

iteration 内的本地执行路径
  framework adapter -> CommIntent -> plan 校验 -> admission
      -> 原始 torch.distributed async collective -> ScheduledWork
      -> completion telemetry
```

这里的 plan 是概念上的基础执行计划，不额外引入 `PlannedIntent`。它由 `CommIntent` 中可确定性生成的元数据和有序 `TaskKey` 序列组成；实际 tensor、stream 和 launcher 仍然是每 rank 的本地绑定。

Phase 1 从单个 process group 和固定 plan 开始。接口保留跨 process group 的可能性，但不提前实现复杂的全局动态重排。

## 4. 需要实现的机制

### 4.1 确定性身份

`TaskKey` 使用所有 rank 都能生成的字段标识一个逻辑 collective：

```text
iteration, microbatch, parallelism, process_group_id,
layer_id, bucket_id, ordinal
```

禁止使用 object identity、pointer、随机 ID、tensor address 或本地 timestamp。

字段含义：

| 字段 | 含义 |
|---|---|
| `iteration` | 训练 iteration 编号。 |
| `microbatch` | 当前 microbatch 编号。 |
| `parallelism` | DP/TP/PP 等训练语义类别。 |
| `process_group_id` | 稳定的逻辑 process group 标识，决定顺序一致性的边界。 |
| `layer_id` | collective 对应的逻辑 layer。 |
| `bucket_id` | layer 内 gradient/parameter bucket 编号。 |
| `ordinal` | 处理其他字段相同的重复 collective 的稳定序号。 |

`TaskKey` 负责回答“这是哪个逻辑 collective”，用于 plan 顺序、跨 rank 校验和 telemetry 对齐。

### 4.2 `CommIntent` 运行时绑定

`CommIntent` 负责回答“本 rank 如何执行这个 collective”，字段包括：

| 字段 | 含义 |
|---|---|
| `key` | 对应的 `TaskKey`。 |
| `op` | `all_reduce`、`reduce_scatter` 等 collective 类型。 |
| `tensor` | 本 rank 的实际通信 tensor。 |
| `process_group` | 本 rank 的实际 PyTorch ProcessGroup handle。 |
| `num_bytes` | 通信数据量，用于准入和 telemetry。 |
| `launch_fn` | admission 后调用原始 collective 的函数。 |
| `producer` / `consumer` | 可选的训练 DAG 生产者和消费者描述。 |
| `ready_event` | 可选的 CUDA ready event，表示 tensor 已可通信。 |

`TaskKey` 必须跨 rank 一致；`CommIntent` 可以包含 tensor、event 和 launcher 等 rank-local 对象，因此不能直接在 rank 间共享。

### 4.3 基础执行计划

plan 只作为概念上的共享状态，不在当前代码骨架中固定独立的 plan class 或序列化接口。它至少需要包含版本、窗口、`CommIntent` 元数据对应的有序 key 序列以及 hash。第一阶段所有 rank 使用相同的静态 plan；plan 构建和 rank-0 分发在静态机制通过后再加入。

本地 scheduler 只有在 intent 的 key 和不变量元数据与当前计划相符时，才允许它进入 admission。

### 4.4 Dynamic Admission

Admission 是独立于基础 plan 的运行时机制。intent ready 后可以暂存在 pending 表中，只有满足 plan 和本地运行条件才被提交。

初始条件只包括：

- producer ready；
- 当前 plan position；
- outstanding collective 数量上限；
- harness 注入的测试 delay。

后续可以替换 admission criterion，但不能破坏 per-process-group sequence invariant。

### 4.5 Work 与 stream 语义

`ScheduledWork` 表示底层 PyTorch `Work` 创建前后的统一等待边界，至少支持 `wait()` 和 `is_completed()`。

异步版本必须保持：

```text
producer stream -> ready event -> communication stream
communication completion -> completion event -> consumer stream
```

### 4.6 Telemetry

每个 collective 记录：

```text
ready_ts, admit_ts, submit_ts, complete_ts,
predicted_duration, actual_duration
```

初期 telemetry 用于机制调试，后续作为 iteration-to-iteration plan 更新的输入。

## 5. 必须保持的不变量

1. 每个 process group 在所有成员 rank 上看到相同的逻辑 `TaskKey` 序列。
2. collective 不能在 producer ready 前提交。
3. `ScheduledWork.wait()` 不能在底层 collective 完成前返回。
4. 本地 intent 与当前 plan 不一致时不能静默继续。
5. collective 提交后不能取消、抢占或重排。
6. scheduler 出错时必须显式失败或按文档化规则 FIFO fallback，不能无限静默挂起。

## 6. 开发里程碑

### M0：PyTorch 行为 harness

构建两 rank standalone 程序。这一步不是训练 workflow，也不修改外部项目。

实验包括：

- 两个 collective 的 FIFO 和固定重排；
- 单 rank 延迟 ready；
- sequence 和 plan hash 不一致；
- task 重复和缺失；
- 先用 Gloo，再用 NCCL/CUDA；
- async `Work.wait()` 和完成时间。

验收：记录 `API call -> CUDA enqueue -> collective execution -> Work completion` 的实际行为，并确认错误场景有界退出。

### M1：核心 schema 与 plan 校验

实现 `TaskKey`、`CommIntent`、生命周期状态、确定性 hash 和校验测试。不新增独立的 `PlannedIntent` 或 `LaunchPlan` 类型。

验收：覆盖重复 key、缺失 key、metadata mismatch、plan hash mismatch 和 per-group sequence validation。

### M2：V0 同步 admission

实现训练线程内的 scheduler facade：

```text
submit intent -> 校验 plan -> admission -> 调用原始 async collective
```

暂不引入 worker thread 或替代 CUDA stream。

验收：两 rank Gloo harness 能重现 FIFO 和安全的固定重排，所有 rank 的 sequence log 相同。

### M3：NCCL/CUDA event 语义

加入 CUDA event 和 profiler instrumentation，明确区分 ready、admission、submission 和 completion。

验收：profiler/Nsight 证明 stream dependency 正确，`wait()` 不会提前返回。

### M4：V1 异步 admission worker

把 deferred launch 移到 scheduler worker 和显式 communication stream。首先验证 ProcessGroupNCCL 是否允许 worker thread 提交 collective。如果不可靠，应记录限制并评估后续 C++ enforcer，不要用未验证的锁机制掩盖问题。

验收：producer 在提交 intent 后可以继续执行，consumer 通过 `ScheduledWork` 等待；没有错误的 stream dependency 或 sequence divergence。

### M5：Megatron DP gradient adapter

适配真实 Megatron 的 gradient synchronization path：

```text
start_grad_sync(bucket) -> CommIntent -> scheduler.submit()
finish_grad_sync(bucket) -> ScheduledWork.wait()
```

验收：FIFO mode 的 loss 和 gradient 与 baseline 一致；关闭 scheduler 可恢复原始路径；真实 training iteration 能产生 telemetry。

### M6：Plan 版本切换

使用 iteration `k` 的 telemetry 为 iteration `k+1` 安装 plan。初始 policy 可以仍为 FIFO，重点是版本化、分发、hash 校验和安全边界切换。

验收：所有 rank 提交相同版本，旧版本 work 已排空后再 commit 新版本。

## 7. 交付物

- standalone Gloo/NCCL mechanism harness；
- 带校验测试的 plan/admission core；
- per-rank sequence 和 timing log；
- profiler/Nsight 实验记录；
- 带 FIFO fallback 的 Megatron DP adapter；
- plan version transition 机制。

## 8. 决策关口

| 关口 | 需要回答的问题 |
|---|---|
| M0 | 当前 API 能否观察和诊断 deferred collective submission？ |
| M3 | 不修改 PyTorch 时能否正确表达 event/stream dependency？ |
| M4 | Python thread 提交 ProcessGroupNCCL 是否足以支撑 prototype？ |
| M5 | framework-level 接入是否保持训练正确性和 baseline 行为？ |
| M6 | plan/admission 机制是否稳定到可以开始 policy 研究？ |

在 M6 通过前，不把 SimAI 的 flow-level bandwidth policy 移植到真实 stack。Phase 1 只暴露 collective-level control。
