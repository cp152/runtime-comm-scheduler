# Work.wait() 与完成信号语义：实测结论与下一步

更新时间：2026-09-08

> 本文档取代原 `docs/todo-revisit-wait-semantics.md`（现仅存于 `check_wait` 分支），把两轮
> wait/完成信号实测（2×3090 与 2×3080 Ti）沉淀为已定论事实，并给出对 scheduler 完成语义与
> M5 方向的明确建议。逐字原始记录见 `check_wait@docs/experiments/wait-semantics-recheck-raw.md`。

## 1. 为什么做这组实验

M3/M4 实现里，`ScheduledWork.wait()` 依赖 `_ensure_gpu_complete`（`torch.cuda.synchronize()`）
每次 wait 做**全设备排空**。这个代价是否可去掉，取决于两个问题：

1. c10d 的 stream-ordered 完成在本环境是否真的生效（M3 发现 `WorkNCCL.wait()` 对在途大
   collective 以 ~0.01ms 返回，暗示可能未生效）；
2. 「发射 collective 后，同一 stream 上的后续计算是否自动排在其后、自动读到正确数据」。

问题 2 的旧答案（2×3090，v1–v10 结论 #6「同行紧跟计算读到正确数据」）曾支撑一个备选架构：
scheduler 退化为 **admission gate**、发射留在训练 stream、常见路径零 device sync。第二轮实验
（2×3080 Ti，v11–v19）正是为了验证该前提是否真实成立。

## 2. 环境与方法

### 2.1 两轮环境

| | 旧盒（v1–v10, 2026-08-24） | 新盒（v11–v19, 2026-09-08） |
|---|---|---|
| GPU 拓扑 | 2× RTX 3090, PXB（PCIe 多桥），无 NVLink | 2× RTX 3080 Ti, PIX，无 NVLink |
| torch / CUDA | 2.12.1+cu130 / CUDA 13.0 | 2.12.1+cu130 / CUDA 13.2（driver 595.71.05） |
| 2GiB all_reduce 稳态墙钟 | ~280ms | ~500ms |
| 工具 | 无 profiler | 无 profiler；nsys（/usr/local/bin/nsys） |

### 2.2 共同约束

两轮都用**非 legacy 显式 stream**（刻意避开 legacy 流的隐式同步污染测量），两 rank 各绑单卡
NCCL，async 提交。**因此所有结论严格限定于「显式 stream 发射」**；默认(current) stream 路径
是否不同是 §5 的核心开放问题。

### 2.3 实验角色（v11–v19）

| 实验 | 角色 |
|---|---|
| v11 | v3 复现 + 逐语句 CPU 时间线：定位 ~500ms 块落在哪条语句 |
| v12 | 归因实验：隔离/复用语句，区分块在 `t.sum()` 内还是 `ev.record()`（含一处脚本 bug，已标注，不作数据） |
| v13 | 对照：证明「进程内第几个 sum 阻塞」的序数是 **sum 自身**、而非「第几个 all_reduce」 |
| v14 | 6 个不 wait 的 rep：确认首 rep 阻塞且正确、后续 rep 快速且 stale |
| v15 | 8 个 nowait/wait 交替 rep：同上的序数规律 + wait 恢复正确 |
| v16 | reader 判别：rank0 用 `t.sum()` 读、rank1 用 `out.copy_(t)` 读同一个首个 un-waited collective |
| v17 | `set_sync_debug_mode("warn")`：500ms 块内是否有 torch 命名同步点 |
| v18 | torch.profiler 探针：能否观测该 stall（方法学） |
| v19 | nsys 干净复现：定位阻塞的确切 CUDA API（根因） |

## 3. 结论（已定论事实）

| # | 结论 | 证据 |
|---|---|---|
| 1 | 裸 `Work.wait()` 在 µs 级返回且 `is_completed()==False`。其语义是 **device-side stream ordering**（让调用方当前 stream block 在 NCCL work 完成之后），**不阻塞 CPU、不是 device sync**。 | 新盒 v11–v16 wait 均 8–40µs；torch 2.12 docstring「wait() is the same as synchronize(): letting the current stream block on the completion of the NCCL work」 |
| 2 | 显式 stream 上 async all_reduce 后，**没有自动的 caller-stream ordering**：后续同流计算不会被自动排到 collective 之后，NCCL 在自己私有的内部 stream 上执行。 | v13（S_C）：后续 sum ~50µs、读 stale |
| 3 | 「首个 sum 读到正确」的机制是 **host 侧 launch stall**：进程内第一个 `t.sum()` 的 reduce kernel 在 `cudaLaunchKernel` 内部 host 阻塞约一个在途 collective 的时长（新盒最大单次 494.7ms），因此读到的是已 reduce 的正确值。**不是 allocator**（cudaMalloc 总量 1.3ms/4 次）、**不是显式同步**。 | v19（nsys cuda_api_sum）；v14/v16 首读 500ms+ |
| 4 | 此后进程内所有 `t.sum()` 纯异步（~50µs）：不阻塞、不追踪在途 NCCL，若 collective 仍在飞则读 stale（未 reduce 的 1N/2N）。 | v14 rep1+、v15 i2+、v16 rep1、v13 S_C |
| 5 | 该「首读保护」与读取的 **kernel 类型**相关，不是「消费输出」本身：`out.copy_(t)` 型读者即使进程内第一次读，也 ~0.2ms 返回并读 stale；只有 reduce 型读取（`t.sum()`）触发 #3 的 stall。 | v16：rank0 sum 读 527ms 正确，rank1 copy_ 读 0.197ms stale |
| 6 | 当前流 `event.record()` 不追踪 collective：event ~ms 级触发（只反映本地 fill/读，而非 NCCL 完成），`ev.record()` 自身不阻塞。 | 新盒复测（v11/v16 evrec µs–0.135ms）；与旧盒一致 |
| 7 | `torch.cuda.synchronize()` 仍是**唯一可靠完成信号**。 | 空闲 µs、在途 ~collective 时长（两盒一致） |
| 8 | `set_sync_debug_mode("warn")` 在 500ms stall 块内 0 warning：该块不在 torch 命名同步点里（是 launch 层的隐式阻塞）；`item()` 等已知同步点会 warn。 | v17 |
| 9 | torch.profiler(kineto) 无法观测该 stall：region 墙钟 ~1152ms 但捕获 CPU 事件仅 ~19ms，且跨周期清事件；方法学上此类 host-stall 只能靠 nsys。 | v18 |

### 3.1 对旧结论 #6 的订正

旧盒 v1–v10 的「同行紧跟计算读到正确」（#6）被新盒证明**不是 c10d 自动排序**，而是首个
reduce 型读取的 launch-stall 副产物（#3/#4/#5）。因此：

- **旧「零 sync admission-gate」前提在新盒不成立**。若发射后不 wait 就读输出，从进程内第二个
  sum 起、或任何 copy 型读取起，都会读到未 reduce 数据；
- **Megatron DP 路径的安全不依赖自动排序**：它在 bucket 边界显式 `wait()`/同步、或梯度不再被
  读的边界才消费。真实模式安全 = 显式完成边界，不是「发射后同行计算自动正确」。

### 3.2 残留的盒子间差异（开放）

- 旧盒 v3/v4/v8/v10 在进程内多次 rep 都读到正确，与新盒「第二个 sum 起即 stale」不一致。
  机制未调和，疑与 driver/CUDA（13.0 vs 13.2）、拓扑（PXB vs PIX）或旧盒未复现的 launch-stall
  规律有关。**结论以新盒为准；M5 必须在目标开发盒上重新验证关键假设，不能跨盒外推。**

## 4. 对 scheduler 架构与 M5 的含义

1. **scheduler 在独立 comm stream 上代发射时，拿不到廉价的 GPU 完成信号**（#1/#2/#6：wait、
   future、当前流 event 都不追踪 NCCL 完成）。现状 `_ensure_gpu_complete`（device sync）在
   M5 前**不能去掉**——这与 2026-08-24 的决策一致，但理由从「wait 不可靠」升级为「完成语义
   在代发射路径上根本没有廉价信号」。
2. **admission-gate 备选架构的可行性被收紧**：它要求「训练 stream 自己发射 + 后续计算自动
   正确」成立。新盒显式 stream 上该前提不成立；它只有在 **默认(current) stream 路径确有
   device-order 保证**时才可能复活——这正是 §5 Exp-1 要判定的问题。
3. 若走 gate 且需要「某 group 已完成 / outstanding 上限」类完成门控策略，没有廉价完成信号，
   可改用**程序序 checkpoint 限流**（纯 CPU，`max_outstanding` 从「GPU 在途数」改为「程序序
   未消费数」），或接受少量定向 sync。

## 5. 下一步（建议与理由）

### Exp-1（优先）：默认 stream 的 auto-order 判别

**为什么必须做**：真实 Megatron 的 gradient all_reduce 在训练线程的**当前（默认）stream** 上
发射，而 v1–v19 全部在显式 stream 上跑，结论 #2/#3 不能直接外推到默认 stream。若默认 stream
确有 device-order 保证，gate 架构零-sync 复活，M5 走「gate + 程序序限流」；若没有，M5 必须走
「代发射 + 定向 sync」或「训练线程 gate、原始发射不变」。

**方法**（无需改 scheduler，在盒子上即可跑）：默认 stream 上 `all_reduce(t, async_op=True)`，
随后同流紧跟：
- (a) `t.sum()`（进程内第 1 个 vs 第 2 个）读数是否正确；
- (b) `out.copy_(t)` 读者读数是否正确；
- (c) 对照：读前不 sync vs `torch.cuda.synchronize()`。

判据：(b) 正确 ⇒ 默认流有 device-order 自动排序；(b) stale 而 (a) 首读正确 ⇒ 仍是
launch-stall 副产物，gate 前提不成立。

### 完成信号决策（与 Exp-1 结果联动）

- 现状保留 `_ensure_gpu_complete`（全同步）直到 Exp-1 出结果；
- 之后三选一：保留同步（接受 M5 DP 重叠损失）/ 流 wait-event / 程序序 checkpoint 限流。

### 非阻塞研究项（记录即可，不拦 M5）

- 首个 sum 的 `cudaLaunchKernel` host-stall 触发条件（为何只第一个、为何只 reduce 型 kernel、
  与盒子/驱动相关性）；
- 旧盒 v10 差异（§3.2）。

### 里程碑

M5（Megatron DP adapter）与 M6（plan 版本切换）**需另行批准**后再立项；本文档不替 M5 定架构，
只提供 Exp-1 与完成信号决策作为其前置输入。

## 6. 关联

- 逐字原始记录：`check_wait@docs/experiments/wait-semantics-recheck-raw.md`（v1–v19，全部 rank
  原始 JSON 与时间戳）
- 旧盒首轮结论与 admission-gate 备选架构全文：`check_wait@docs/todo-revisit-wait-semantics.md`
- `src/runtime_comm_scheduler/work.py::_ensure_gpu_complete`（§4.1 涉及的现状实现）
- `docs/architecture.md` §4.5/§6（ScheduledWork 与正确性模型）、`docs/phase1-plan.md` M3/M4/M5
