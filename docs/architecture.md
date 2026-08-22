# 系统架构

本文档是项目的长期设计参考，记录真实 training stack 的边界、schedule layer 的机制和后续阶段仍需遵守的正确性不变量。

## 1. 问题边界

`SimAI/simai-flow-scheduler` 已经验证了 flow-level priority 和带宽分配策略在模拟训练执行中的效果。但真实 PyTorch/NCCL stack 不会向上层暴露相同的逐 flow 事件和控制接口，因此这些策略不能直接移植。

本项目首先回答：

> schedule layer 能够在不破坏 collective 正确性的前提下观察和控制哪些训练通信行为？

当前答案是 collective-level planning 和 admission。collective 内部以及网络层的更细粒度控制保留为后续扩展。

## 2. Training Stack 与控制边界

```text
Megatron / training framework
  -> torch.distributed API
  -> c10d ProcessGroup
  -> ProcessGroupNCCL
  -> NCCL communicator、channels、kernels
  -> NVLink / PCIe / NIC / network
```

初始 schedule layer 位于训练框架和原始 `torch.distributed` 调用之间，不替换 c10d 或 NCCL：

```text
训练调用点
  -> CommIntent
  -> schedule layer：plan 校验与 admission
  -> 原始 torch.distributed collective
  -> 底层 Work / CUDA completion
  -> ScheduledWork 与 telemetry
```

## 3. 可观察与可控制事件

| 事件或资源                                      | 当前 schedule layer | 说明                                     |
| ----------------------------------------------- | ------------------: | ---------------------------------------- |
| layer、microbatch、DP/TP/PP 角色等训练语义      |              可观察 | 只有 framework adapter 掌握。            |
| tensor/gradient ready                           |              可观察 | 通过 CUDA event 或框架回调获得。         |
| collective intent 创建                          |              可观察 | 本地提交`CommIntent` 时产生。          |
| c10d collective API 调用                        |              可控制 | scheduler 可以延迟或准入。               |
| collective 完成                                 |              可观察 | 通过底层`Work`、future 或 CUDA event。 |
| 同一 process group 内尚未提交 collective 的顺序 |        有条件可控制 | 所有成员 rank 必须保持一致。             |
| 不同 process group 间的发射时机                 |        有条件可控制 | 不能破坏各 group 的诱导顺序和训练依赖。  |
| CUDA stream/event 依赖                          |              可控制 | 异步实现需要显式管理。                   |
| NCCL channel/kernel/chunk 顺序                  |        初期不可控制 | 需要修改 NCCL/UCC/ProcessGroup。         |
| packet 或 flow 的进入/退出                      |        初期不可观察 | c10d/NCCL 未暴露这些事件。               |
| 每条 flow 的带宽分配                            |        初期不可控制 | 需要下沉到 data plane 或网络层。         |
| collective 提交后的抢占/取消                    |            不可控制 | 当前层不能安全撤回已提交 collective。    |

## 4. 各层职责

### 4.1 Training Framework Semantic Adapter

适配器将训练调用转换为带语义的 `CommIntent`。它是唯一能够识别 training DAG 中 producer 和 consumer 的层级。

可传递的元数据包括：

```text
iteration、microbatch、virtual pipeline stage、layer、
parallelism type、process group、collective type、bytes、
producer、consumer、expected consumer deadline
```

第一个接入点是 Megatron 的 DP gradient synchronization；TP、PP 等属于后续适配器。

### 4.2 Per-Rank Schedule Layer

schedule layer 同时包含 plan 路径和 admission 路径：

```text
共享的 plan 概念
  -> 本地 plan 校验
  -> pending CommIntent 表
  -> runtime admission 仲裁
  -> 原始 collective launcher
  -> ScheduledWork 与 telemetry
```

第一阶段不实现 collective 算法，只控制原始 launcher 是否以及何时被调用。

### 4.3 Data Plane

初期 data plane 是未修改的 ProcessGroupNCCL/NCCL，保持生产 collective 算法不变，使实验变量明确为 launch/admission 而不是新的 all-reduce 实现。

未来可以将 enforcement 下沉到 C++ ProcessGroup wrapper、NCCL、UCC 或网络 data plane，但必须由上层机制无法提供所需控制的证据驱动。

## 5. Plan + Dynamic Admission

### 5.1 CommIntent 与 Plan

项目只保留一个 `CommIntent` 数据结构，不为 plan 额外引入 `PlannedIntent`。plan 是概念上的共享执行计划，由确定性的 `CommIntent` 元数据和有序 `TaskKey` 序列构成：

```text
CommIntent 的共享元数据
  - TaskKey
  - collective/op 类型
  - process_group 标识
  - bytes 与训练语义
  - plan version / baseline order

CommIntent 的本地运行时绑定
  - 实际 local tensor 与 output buffer
  - local process-group handle
  - ready CUDA event
  - 原始 launch_fn
```

计划中的元数据必须确定性生成并可用于 hash；tensor、stream handle 和 Python closure 只属于本 rank 的运行时对象，不能作为 plan 内容广播。

### 5.2 Plan 结构

概念上，一个 plan 包含：

```text
Plan {
  version,
  window_id,
  CommIntent 元数据集合,
  每个 process group 的诱导子序列,   // 顺序不变量
  plan_hash                          // 文档身份，非执行顺序
}
```

Plan 是一份**全 rank 共享的规范文档**（同一 version/window/hash），不是每 rank 各自的执行计划。`entries` 是确定性的共享载体；per-group 诱导子序列由它过滤得到，是唯一需要跨 rank 一致的顺序不变量。**每 rank 的实际执行序列是运行时投影**（把该 rank 所属各 group 的子序列按训练 DAG 依赖合并），不存入 plan，由 framework adapter 提交、scheduler 按 per-group 子序列校验。`plan_hash` 的语义是「所有 rank 安装同一份文档」，不等于执行顺序相等。

Plan 由配置文件指定的通信 Task 序列或应用层提交的 CommIntent 序列实例化；当前骨架只保留 `CommIntent` 和 scheduler 边界。初期使用每 iteration 的静态有序 key 序列，后续再由 iteration `k` 的 telemetry 生成 `k+1` 的序列。

### 5.3 Dynamic Admission

Admission 路径根据 plan 未固定的运行时状态做出有限调整：

- producer 是否已经 ready；
- 计划中的 collective 是否需要 delay；
- outstanding collective 数量是否达到上限；
- 不同 process group 中哪个 ready task 先发射；
- 当前运行时测量是否需要反馈给下一版本 plan。

约束如下：

1. 可以 delay task，但不能在 producer ready 前提交。
2. 不能在本地跳过或重排同一个 process group 的计划子序列。
3. 只有在各成员 rank 的诱导序列仍然一致时，才允许进行跨 process group 仲裁。
4. task 提交给 NCCL 后不能取消、抢占或重排。

## 6. 正确性模型

对于任意 process group `G`，所有成员 rank 必须在 `G` 上提交相同且兼容的 collective 序列。实现必须主动暴露 divergence，而不是静默等待 NCCL 超时。

需要检查：

- plan 安装时的文档身份（plan hash）一致性——确保各 rank 安装同一份共享文档；
- `TaskKey` 的确定性构造；
- intent 重复和缺失；
- 提交前的 per-group sequence 校验——执行期的顺序不变量，由 scheduler 完成；
- 有界 timeout 和错误处理；
- FIFO fallback 或显式 fail-stop 模式。

`plan_hash` 一致性与 per-group sequence 校验是两类不同的问题：前者回答「各 rank 是否安装同一份共享文档」，后者回答「每个 group 的执行顺序是否在其成员间一致」。

任务生命周期为：

```text
CREATED -> READY -> WAITING_FOR_ADMISSION -> ADMITTED -> SUBMITTED -> COMPLETED
```

`SUBMITTED` 是不可逆边界。

## 7. Plan 版本切换

初始安全重配置边界是 iteration boundary：

```text
收集 iteration k 的 telemetry
  -> 构建 iteration k+1 的 plan
  -> 分发并校验 plan hash
  -> drain 受旧 plan 影响的 work
  -> commit 新 plan
```

初期不需要 iteration 内替换 plan；后者需要额外的跨 rank ready-state 和 plan agreement 协议。

## 8. 演进路线

| 层级              | 后续可能支持的能力                     | 所需新机制                          |
| ----------------- | -------------------------------------- | ----------------------------------- |
| Framework adapter | DP、TP、PP、FSDP/ZeRO 语义 intent      | 新适配器和 consumer dependency 建模 |
| Schedule layer    | runtime-adaptive policy、跨 group 仲裁 | 版本化 plan 和 admission policy     |
| ProcessGroup      | 脱离具体框架的统一 enforcement         | C++ wrapper 或 custom backend       |
| NCCL/UCC          | channel、algorithm、chunk 控制         | 修改 collective runtime             |
| NIC/network       | flow priority、pacing、带宽分配        | transport 和网络 data plane         |

项目从前两行开始；在实现下层之前，不应宣称已经具备 flow-level control。
