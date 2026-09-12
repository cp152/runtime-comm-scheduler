# 3D 并行下跨 Communicator 的 Host Launch Order

## 1. 问题是什么

这里的 host launch order 指 CPU 向 ProcessGroup/NCCL 发出通信调用的先后顺序。它只约束操作的提交顺序，并不要求前一个 collective 执行完成后才能提交下一个，因此不会天然消除 GPU 上的通信并发。

同一个 communicator 内，各成员 rank 必须以兼容的顺序提交相同的 collective。但当多个 communicator 在 rank/GPU 上相互重叠时，只满足各 communicator 内部顺序仍可能不够。

例如：

```text
comm A = {rank 0, rank 1}
comm B = {rank 1, rank 2}
comm C = {rank 2, rank 0}
```

如果 rank 0、1、2 分别先发射 C、A、B，就可能形成跨 communicator 的循环等待。此时每个 communicator 内都只有一个操作，所以单独检查各 communicator 的 collective sequence 发现不了问题。

## 2. 不要求所有 Rank 拥有相同任务集合

3D 并行中，每个 rank 参与的 TP、PP、DP process group 通常不同。因此，“一致的 launch order”并不是让所有 rank 执行完全相同的通信列表。

更合适的定义是：

1. 系统先为一个 scheduling window 建立共享的全局逻辑顺序；
2. 每个 rank 删除自己不参与的任务，得到本地投影；
3. 每个 rank 按该投影提交通信任务。

例如：

```text
global plan:       A -> B -> C -> D
rank 0 projection: A ->     C -> D
rank 1 projection: A -> B
rank 2 projection:      B -> C
```

不参与某项通信的 rank 直接跳过该任务，不需要提交占位 collective。不同 rank 的任务集合可以不同，但它们共有的任务应来自同一份逻辑计划，不能各自独立决定互相冲突的提交次序。

全局任务身份不应依赖本地 ProcessGroup 对象地址，可以使用确定性字段：

```text
TaskKey = iteration / microbatch / layer / phase /
          parallelism-type / group-membership / op-type / occurrence
```

## 3. 第一版实现建议

第一版可以采用保守的全序方案：

- 给 scheduling window 中的任务分配确定性的 `launch_seq`；
- 各 rank 根据 group membership 生成本地投影；
- 每张 GPU 只由一个 host launch sequencer 向底层 ProcessGroup 提交操作；
- sequencer 按序快速提交，但不等待前一个任务执行完成。

这样做的目的主要是先避免跨 communicator 的循环顺序，而不是让网络操作串行执行。

后续如果全序产生明显的 head-of-line blocking，可以将其放宽为偏序 DAG。只在以下任务间建立顺序边：

- 使用同一个 communicator；
- 共享同一 rank/GPU，且底层并发发射可能带来进展风险；
- 存在训练计算或数据依赖；
- 调度协议明确要求保持次序。

动态优先级策略只能从 DAG 当前的 ready frontier 中选任务。如果要改变已有顺序约束，应由跨 rank 协议共同生成新的 plan/grant，而不能由某个 rank 单方面越过前驱。

## 4. 需要保持的不变量

- 同一 communicator 的成员提交相同且兼容的 collective sequence。
- 每个 rank 的本地顺序是共享逻辑计划的合法投影。
- 同一 GPU 上只有一个组件最终决定跨 communicator 的 host 提交顺序。
- 已经提交给底层 ProcessGroup/NCCL 的操作不能再取消或重排。
- CUDA Graph 模式初期可以将整个 capture/replay 区域视为不可拆分的调度单元。
