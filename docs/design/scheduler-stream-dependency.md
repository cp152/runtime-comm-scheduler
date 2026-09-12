# 引入通信 Scheduler 后的 CUDA Stream 依赖保持方案

## 1. 目标

调度器需要延迟或重排通信任务的提交时间，但不能改变原有 producer、通信和 consumer 之间的数据依赖。

本文只讨论 stream dependency 的传递，不讨论多个 communicator 之间应采用什么 launch order。

## 2. 原始异步通信的依赖关系

典型通信涉及三类逻辑 stream：

```text
producer stream       ProcessGroup NCCL stream       consumer stream
      P                         N                           C
      |                         |                           |
      +-- record E_ready ------>|                           |
                                |-- NCCL kernel             |
                                +-- record E_done --------->|
```

本质上是两条 CUDA event 依赖：

```text
P --E_ready--> N --E_done--> C
```

调用 collective 时，ProcessGroupNCCL 会根据调用线程的 current CUDA stream 建立 producer 到其内部 NCCL stream 的依赖。之后，`Work.wait()` 会将通信完成依赖接到调用 `wait()` 时的 current stream。通常这些是 GPU-side stream wait，不需要同步整个设备。

## 3. Scheduler 带来的问题

如果应用线程只创建 `CommIntent`，真正的 ProcessGroup 调用由后台 launch thread 执行，那么底层看到的 current stream 已经不再是原 producer stream。

如果调度器直接在后台调用 ProcessGroup，就可能出现：

- NCCL 在 producer 尚未写完输入时读取 tensor；
- consumer 在通信尚未完成时读取输出；
- tensor 在 pending 期间失去引用，底层存储被 allocator 复用。

因此，`CommIntent` 除了描述通信操作，还必须携带原调用点的数据依赖和对象生命周期。

## 4. Producer 到通信的依赖桥接

应用线程创建 intent 时执行：

1. 记录当前 CUDA device 和 producer stream；
2. 在 producer stream 上记录 `ready_event`；
3. 保存输入 tensor、输出 buffer 及相关对象的强引用；
4. 将 intent 放入 pending queue；
5. 返回尚未绑定真实 ProcessGroup Work 的 `ScheduledWork`。

不要在这里调用 `ready_event.synchronize()`。调度器只需要保存 GPU 依赖，不应该让 CPU 等待 producer 完成。

调度器准许任务提交后，每张 GPU 的唯一 launch thread 按 process group 使用独立
bridge stream `S_gate[group]`：

```text
S_producer --ready_event--> S_gate[group] --> ProcessGroup internal NCCL stream
```

具体过程为：

1. 设置正确的 CUDA device guard；
2. 根据 logical process group 选择 `S_gate[group]` 并进入其 stream guard；
3. 让 `S_gate[group]` 等待 intent 中的 `ready_event`；
4. 在该 stream context 中调用原始 ProcessGroup，并以异步模式取得真实 Work；
5. 将真实 Work 绑定到 `ScheduledWork`。

未经修改的 ProcessGroup 会把 `S_gate[group]` 视为 caller current stream，从而继续建立 `S_gate[group] -> NCCL stream` 的依赖。gate stream 只负责把原 producer dependency 传递给 ProcessGroup；NCCL kernel 仍运行在 ProcessGroup 管理的内部 stream 上。

“单 launch thread”不等于“单 gate stream”。如果不同 communicator 共用一条 gate
stream，后一个 intent 会不必要地继承前一个 intent 的 producer-ready 依赖，造成
跨 communicator head-of-line blocking。同一 group 的 collective 本来就必须有序，
因此按 `(device, process_group_id)` 分配 gate stream 可以在保持 host launch 全序的
同时，让不同 group 的 GPU dependency 相互独立。

只有在未来绕过 ProcessGroup、直接调用 NCCL 时，才需要由调度层自己选择 NCCL stream、记录 completion event 并维护 allocator 安全性。

## 5. 通信到 Consumer 的依赖桥接

`ScheduledWork.wait()` 可以实现为：

1. 若任务尚未 launch，等待真实 Work 被绑定；
2. 调用真实 `Work.wait()`；
3. 让真实 Work 将 NCCL completion dependency 插入此时的 consumer current stream。

最终依赖链为：

```text
producer stream
   -> ready_event
   -> scheduler bridge stream
   -> ProcessGroup NCCL stream
   -> NCCL completion event
   -> consumer stream
```

对于原本的同步 collective，调度层内部仍可使用异步 ProcessGroup 调用，但必须在返回应用前执行对应的 `ScheduledWork.wait()`，以保持原 API 的同步语义。

## 6. 实现时需要保持的不变量

- launch thread 必须显式设置正确的 device guard 和 stream guard。
- `ready_event` 必须记录在创建 intent 时实际的 producer current stream 上。
- pending 阶段必须持有 tensor 和 buffer 的强引用。
- `ScheduledWork` 必须保留底层 Work，直到 consumer dependency 已经建立。
- 普通依赖传递不应使用 `cudaDeviceSynchronize()` 或 event 的 CPU synchronize。
- 必须考虑异常传播：launch 失败、真实 Work 失败或调度器停止时，`ScheduledWork.wait()` 不能永久阻塞。
- 任务一旦提交给底层 ProcessGroup，调度器只能观察其状态，不能再把它当作 pending 任务重新排序。

## 7. 推荐的最小落地路径

第一版可以只实现以下链路：

```text
CommIntent creation
  -> record producer ready_event
  -> scheduler admission
  -> bridge stream waits ready_event
  -> async ProcessGroup call
  -> bind real Work
  -> ScheduledWork.wait delegates to real Work.wait
```

先保持原有 FIFO 顺序验证数值正确性、无 race 且没有全设备同步，再打开优先级和重排策略。这样可以将“stream 语义是否正确”和“调度策略是否正确”分开验证。
