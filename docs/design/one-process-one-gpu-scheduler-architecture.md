# 一进程一 GPU 模型下的通信 Scheduler 架构构想

## 1. 背景与基本模型

本文讨论主流 PyTorch 分布式训练中的“一进程一 GPU”模型：一个训练任务由多个 rank 进程共同组成，每个 rank 进程主要管理一张 GPU，同时参与若干不同的 TP、PP、DP ProcessGroup。

```text
Training Job
├── Rank 0 Process -> GPU 0 -> TP/PP/DP ProcessGroups
├── Rank 1 Process -> GPU 1 -> TP/PP/DP ProcessGroups
├── Rank 2 Process -> GPU 2 -> TP/PP/DP ProcessGroups
└── ...
```

因此，通信 scheduler 面对的核心对象并不是“一台机器上的所有 GPU”，而是分散在各 rank 进程中的通信请求。每个 rank 都能直接看到本进程发往多个 communicator 的请求，但无法天然看到其他 rank 的完整运行状态。

整体设计需要同时解决两个层面的问题：

- 在每个 rank 内截获、保存并最终发射真实通信请求；
- 在多个 rank 之间形成一致的调度意图，避免各自独立决策造成 collective 顺序冲突。

## 2. 核心思想：控制面与执行面分离

从逻辑上看，scheduler 可以分为控制面和执行面。

```text
                  Communication Scheduling Layer
                 /                              \
        Control Plane                       Execution Plane
     观察、决策和跨 rank 协调              保存并发射本地通信请求
```

控制面关注“应该发送什么”：

- 收集通信任务的逻辑信息；
- 观察 pending 任务及可能到来的后续任务；
- 根据依赖、优先级、拓扑或训练关键路径决定顺序；
- 在多个 rank 之间形成兼容的调度结果；
- 将决定表达为某种顺序、许可或执行配置。

执行面关注“如何在本 rank 正确地发送”：

- 保存真实 tensor、ProcessGroup 和 CUDA 上下文；
- 在调度器允许之后调用底层 ProcessGroup/NCCL；
- 保持原有 producer、communication、consumer 之间的 stream 依赖；
- 将底层通信的完成状态重新连接到训练框架原有的异步语义。

这一区分的关键在于：控制面只需要处理任务元数据，而执行面必须位于训练进程内部。tensor、CUDA event、ProcessGroup、NCCL communicator 和内存生命周期都属于原进程，不适合由一个外部 scheduler 进程直接接管。

## 3. Rank 内部的逻辑结构

每个 rank 进程内部可以存在一个本地 scheduler layer，位于训练框架与 ProcessGroup 之间：

```text
Application / Autograd
          |
          | collective request
          v
 Communication Interception
          |
          | communication intent
          v
      Local Scheduler
     /               \
local control      local execution
     \               /
       ProcessGroup / NCCL
               |
              GPU
```

通信截获层将原本立即进入 ProcessGroup 的 collective 表达为一个 communication intent。这个 intent 一方面包含可供控制面理解的逻辑属性，例如通信类型、所属 communicator、训练阶段和优先级；另一方面关联只有本进程才能使用的执行对象，例如 tensor、CUDA 依赖和真实 ProcessGroup。

Local Scheduler 将“请求被训练代码创建”和“请求真正提交给 ProcessGroup”分离。由此获得一个 pending 区域，使调度策略有机会观察多个通信任务并选择它们的提交时机和顺序。

真实通信最终仍由本 rank 进程发起。一个统一的本地 launch executor 可以作为该 rank 上所有被调度 communicator 的共同出口，从而使控制面的顺序决定能够转化为确定的 host launch order。这里统一的是 host submission，而不是要求 GPU 等待每个通信完成后再执行下一个通信。

## 4. 初版可以在部署上合并两层

控制面和执行面的分离首先是一种职责边界，不要求初版就部署独立的全局服务。

初版可以让每个 rank 的 Local Scheduler 同时承担：

- 根据共享规则决定当前可以提交的任务；
- 在本进程中发射被选中的真实通信请求。

```text
Rank Process
┌──────────────────────────────────────┐
│ Local Scheduler                      │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ local policy │->│ local launch │ │
│  └──────────────┘  └──────────────┘ │
└──────────────────────────────────────┘
                 |
          ProcessGroup / NCCL
```

虽然部署在一起，概念上仍保留两者边界：调度决策只依赖可描述的任务信息，launch 部分负责本地 CUDA 和 ProcessGroup 语义。这样既能保持初版结构简单，也不会把长期架构绑定在单一进程内。

## 5. 多 Rank 之间的关系

每个 rank 都有一个 Local Scheduler，但这些 scheduler 不能被视为完全独立的调度器。collective 是多个 rank 共同参与的操作，同一个 communicator 的成员必须提交相互匹配的操作；多个 communicator 相互重叠时，还可能形成跨 communicator 的顺序关系。

因此，更准确的系统图景是：

```text
                 Shared Scheduling Intent
                    /       |       \
                   v        v        v
             Rank 0      Rank 1      Rank 2
             Local       Local       Local
             Scheduler   Scheduler   Scheduler
                |           |           |
              GPU 0       GPU 1       GPU 2
```

“Shared Scheduling Intent”不一定意味着存在一个实时中心服务。它也可以来自一份确定性的全局计划，各 rank 根据自己参与的 ProcessGroup 取得本地投影。重要的是，各 rank 的本地决定来自兼容的共同规则，而不是仅根据自己的 pending queue 任意重排。

如果未来调度需要依赖动态网络状态、全局任务到达情况或跨 rank critical path，则可以把控制面进一步提升为节点级或全局 coordinator：

```text
                Global / Node Control Plane
                    ^                 |
              task metadata     scheduling decisions
                    |                 v
          Local Execution Planes in Rank Processes
```

此时外部控制面负责观察和决策，各 rank 内的执行面仍负责真实 tensor 和 NCCL 操作。两者之间传递的是元数据与调度决定，而不是通信数据本身。

## 6. 一次通信请求的概念路径

从架构角度，一次通信请求经过以下过程：

```text
训练框架产生 collective
          |
          v
截获层将其转换为 communication intent
          |
          v
Local Scheduler 将其纳入 pending 任务视图
          |
          v
控制面根据顺序、依赖和策略决定何时允许提交
          |
          v
本地执行面调用原始 ProcessGroup/NCCL
          |
          v
通信完成语义重新返回训练框架
```

这个结构让 scheduler 可以改变通信的提交时间、相对顺序和执行配置，同时将训练框架原有的 tensor 所有权、CUDA stream 依赖和异步 Work 语义封装在本地执行面内部。

## 7. 架构边界

该设计中的几个关键边界是：

- scheduler 统一管理的是通信请求的 admission 和 host submission，而不是取代 NCCL 的数据传输实现；
- 跨 rank 的控制逻辑可以集中或分布式实现，但实际 collective 仍由各 rank 的本地执行面发起；
- 一个 rank 上的多个 ProcessGroup 共享同一个调度视图，避免各 communicator 分别调度而失去全局顺序信息；
- 调度层改变通信时机和配置，但不应改变训练程序观察到的数据依赖和完成语义；
- 初版可以合并控制面与执行面，未来是否拆分以及如何通信属于具体实现选择。

关于跨 communicator 的一致顺序，参见 [cross-communicator-launch-order.md](cross-communicator-launch-order.md)。关于调度后如何保持 CUDA stream 依赖，参见 [scheduler-stream-dependency.md](scheduler-stream-dependency.md)。
