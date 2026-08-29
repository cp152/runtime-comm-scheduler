# TODO：重测 wait 语义，决定如何处理全设备同步

状态：**重测完成（2026-08-24 实测，2×3090，torch 2.12.1+cu130）**。结论与决策映射
见文末「重测结果」。前置：盒子恢复后在 2×3090 上跑下面的实验。

## 问题

`ScheduledWork.wait()` 里的 `_ensure_gpu_complete`（`torch.cuda.synchronize()`）
每次 wait 做**全设备排空**，是 M3 遗留的已知代价：

- 多个 collective 本身在 GPU 上照常并行（worker 早已并发发射）；
- 但 consumer 在 `wait()` 时被 sync 拖住——等的是**所有**在途 GPU 工作，而非
  当前这一个 collective，丢掉「A 的 tensor 一到就开始算 A、B 还在跑」的细粒度
  重叠；
- M5（Megatron DP adapter）的 `finish_grad_sync` 是**逐 bucket 调 `work.wait()`**，
  若每处都全同步，会吃掉 DP 通信重叠，是真实的可用性风险。

## 待重测的开放问题

M3 实测暗示本环境（torch 2.12.1+cu130）连 stream-ordered 完成都可能没生效：
`WorkNCCL.wait()` 对 4GB all_reduce 返回 ~0.01ms，naive 记在当前流的 event
不追踪 NCCL 完成。这与 PyTorch 的设计（c10d 让调用方当前流 wait comm-stream
完成 event）矛盾。可能被 profiler 干扰 / legacy 流隐式同步搅浑，需干净重测。

### 实验（盒子恢复后）

在**非 legacy 显式流**上、不开 profiler，发一个大 all_reduce：

1. 发完后在当前流上 `event.record`，立即 `query()`：若等待 NCCL 则为 False
   （stream-ordered 生效），若立即触发则为 True（未生效）。
2. 裸 `WorkNCCL.wait()` 的墙钟：阻塞到 GPU 完成 vs ~0.01ms 即返回。
3. `underlying.get_future().wait()` 的墙钟：只等当前 collective vs 全设备
   （探中间方案是否可靠）。

## 决策选项（按优先级）

| # | 方案 | 前提 | 代价 |
|---|---|---|---|
| 1 | `wait()` 改为「consumer 流 wait 完成 event」 | 实验 1 证明 stream-ordered 生效 | 消除全同步 |
| 2 | 用 `Work.get_future()` 做集体级完成等待 | 实验 3 证明 future 可靠 | 只等当前 collective，非全设备 |
| 3 | 保留同步 + 接受代价 | 1/2 都不可靠 | M5 需记录 DP 重叠损失 |
| 4 | 自定义 ProcessGroup / C++ enforcer | 架构演进路线下一层 | 工作量大，最后手段 |

> 注：下面「架构讨论记录」新增了一个备选架构（scheduler 退化为 admission
> gate、发射留在训练 stream）。若 stream-ordered 完成被证明生效，该架构可让
> 选项 1/2/3 的取舍在常见路径上**失去意义**——scheduler 不再需要 wait 语义。

## 关联

- `src/runtime_comm_scheduler/work.py::_ensure_gpu_complete`
- `docs/experiments/m0-m4.md` M4 章节「关键发现 #2」（M3 遗留全同步未消除）
- M5（Megatron DP gradient adapter）：逐 bucket `wait()` 的可用性风险

---

## 架构讨论记录（2026-08-24）：完成信号流转与 admission-gate 备选架构

> 为检验上文 wait 语义做准备时，讨论了「正常训练中后续计算如何与 comm stream
> 同步」和「为何 scheduler 层拿不到 NCCL 完成的确定时机」，并据此提出一个
> 备选架构。以下为结论，供后续决定 wait 语义处理方式与 M5 方向时参考。

### 正常训练的完成信号流转（两层机制）

在训练流的普通 `dist.all_reduce(t, async_op=True)` 中，c10d 对**调用方当前流 S**
与内部 **NCCL comm stream C** 做了双向 event 编排：

1. **① 输入就绪**：S 上 record 事件，C `wait_event` 它——C 等 producer 写输入
   （M3/M4 `delayed_ready` 实测过的就是这一半，可用）。
2. **② 输出就绪**：C 上 record 完成事件，S `wait_event` 它——后续在 S 上排的
   计算 op 被 GPU 天然排在 collective 之后。

因此**后续计算什么都不用做**：继续在 S 上排 op，stream-ordered 保证数据正确。
关键性质：这个保证是「**流按序执行**」，不是「CPU 拿到一个完成时刻」；且它绑定
在**发射方当前流**上。Megatron 不逐 op 调 `work.wait()` 还能正确，靠的就是它。

CPU 侧（`Work.wait()`/`get_future()`）走的是**另一个机制**：在 C 上注册 CUDA
host function（`cudaLaunchHostFunc`），C 执行过 collective 后触发回调、唤醒
`wait()`。它阻塞 CPU 但不阻塞 GPU、也**不是** device sync——是「定向到那一个
comm stream」的完成信号。M3 发现 #2（4GB all_reduce `wait()` ~0.01ms 即返回）
说明这条 host-callback 路径在本环境（torch 2.12.1+cu130）存疑，正需上文实验验证。

### 为何 scheduler 层拿不到「确定的结束时机」（四点）

机制本身存在，但**从设计上不向发射方之外的层暴露**：

1. **comm stream 是 `ProcessGroupNCCL` 的私有物**：惰性创建、跨 collective 复用，
   从不给句柄（暴露会破坏 NCCL 内部流序假设）。用户唯一能拿的是
   `torch.cuda.current_stream()`——那永远是自己 的流。
2. **完成 event 藏在 C++ `WorkNCCL` 里**：Python `Work` 只暴露
   `wait()`/`is_completed()`/`get_future()`/`synchronize()`，没有可
   `query()`/挂接的 `completion_event`。
3. **stream-ordered 完成是「发射方当前流」的属性，不是 Work/communicator 的**：
   c10d 编排的是调用那一刻的当前流。scheduler 一旦在自己的 comm_stream 上代发射、
   让 consumer 在训练流上跨流等，c10d 的自动排序对 consumer 无效 → scheduler 只能
   自己证明完成，而无代价的证据（ev_out）够不着 → 只剩 `torch.cuda.synchronize()`。
   **这是 device sync 的真正根源：scheduler 不是发射方流的「主人」。**
4. 剩下的廉价查询（发射方流上 collective 之后自己 record event 再 `query()`）恰好
   依赖第 3 点是否成立，是上文实验 1 要测的开放问题。

### 备选架构：scheduler 退化为 admission gate

训练 stream 先 `acquire`（校验 + plan 序 + 计数），得到允许后**自己**调原始
`dist.all_reduce`；scheduler 不再代发射、不再包 Work、不再提供完成语义。

- **解决了什么**：c10d 的流序自动挂到真正消费数据的流上，device sync 的存在理由
  被整体移除（不再是「换一种更小的同步」，而是消灭这个场景）。producer-ready 变
  成训练流自己的 `wait_event`（原生）。worker 线程、锁、`_worker_error`、
  concurrent-submit 隐患（单一发射线程天然成立）全部消失。约束模型（§5.3）本就
  只有 admission/delay 类约束，gate 是其正确实现而非近似。
- **残留问题**：任何**依赖「collective 已完成」的准入策略**（outstanding 上限、
  delay 到某 group 完成）仍需完成信号——用 host callback（M3 存疑）或流上 event
  query（待测）。绕开思路：把上限改成**程序序 checkpoint 限流**（训练代码到
  DAG 检查点时报告，纯 CPU、不需 GPU 完成知识，但 `max_outstanding` 语义从
  「GPU 在途数」变成「程序序未消费数」）。
- **代价**：阻塞式 `acquire`（延迟 = 训练循环等 gate），需 pre-acquire 纪律避免
  GPU 空转；enforcement 边界从「gate 真实 NCCL 提交」移到「gate 训练线程发射
  时机」，architecture.md 中 schedule layer 的表述需相应调整。

### 对后续阶段的含义

- 上文实验 1/2/3 是这个备选架构能否「零 device sync」的**分水岭**，且**无需先改
  scheduler**，在盒子上即可跑。
- 结果映射：
  - stream-ordered 生效 → gate 架构是 M5 正解，常见路径零 sync；
  - 仅 `Work.wait()` 不可靠、训练流 event 可靠 → 用 event.query 做完成门控，
    架构仍成立；
  - 全不可靠 → 完成门控策略退化为少量定向 sync 或暂时剔除，数据正确性不受
    stream 序影响。
- 注意：M5 `finish_grad_sync` 的逐 bucket `work.wait()` 正是「完成门控」的窄点，
  与本讨论的残留问题直接相关。

---

## 重测结果（2026-08-24 实测）

环境：AutoDL 2× RTX 3090（24GB），torch 2.12.1+cu130，CUDA 13.0，两 rank NCCL
（各绑单卡，`env://`），**无 profiler**，非 legacy 显式流。harness：
`examples/wait_recheck_worker_v2..v10.py` + `examples/run_wait_recheck_v2.py`。

### 环境要点

- **无 NVLink**：`nvidia-smi nvlink` 全部 inactive；`nvidia-smi topo -m` 显示
  GPU0↔GPU1 = PXB（PCIe 多桥）。空闲时 PCIe link gen.current=1。
- 2GiB（`1<<29` float32）all_reduce 的稳态设备占用 **~280ms**（v9 三次
  first/steady sync 全在 280–286ms），与 PCIe 传输一致。
- `torch.cuda.synchronize()` 空闲时 **µs 级**（v5 idle=34µs），即 sync 本身不慢；
  大 collective 在途时才 ~280ms。

### 结论（事件无关的可靠测量）

| # | 结论 | 证据 |
|---|---|---|
| 1 | **裸 `Work.wait()` 提前返回**：~10µs，而 GPU 工作还 ~280ms | v2 S2：wait=9µs，sync_after_wait=285ms |
| 2 | **`get_future().wait()` 同样提前返回**：~10µs，~279ms 剩余 | v2 S3：fut=11µs，sync_after_fut=279ms |
| 3 | **`torch.cuda.synchronize()` 是唯一可靠完成信号** | v5（空闲 µs）、v9（在途 280ms） |
| 4 | **当前流 `event.record()` 不追踪 collective**：事件 ~3ms 触发（只反映本地 fill），sync_after_ev=275ms | v2 S1 |
| 5 | all_reduce enqueue 非阻塞（~0.4ms，不 flush 调用方流） | v7 enqueue_wall≈0.4ms |
| 6 | **训练流发射自己的 all_reduce 后，同行紧跟的计算读到正确（已 reduce）数据** | v3/v4/v8/v10：`t.sum()`=3×N 精确正确，两 rank 多次 |

**M3 发现 #2 在干净环境（无 profiler、非 legacy 流）下确认复现**，且同样打在
`get_future()` 路径上：CPU 侧两条完成信号都不反映 GPU 完成。→ 决策表**选项 2
（get_future）不可行**；选项 1（consumer 流 wait 完成 event）也因结论 4 不可行——
本构建中「stream-ordered 输出完成」未体现在调用方流的 event 上。

### 未解之谜：读输出 vs 不读输出

- 发射 2GiB all_reduce 后**不读输出**直接 `torch.cuda.synchronize()`：**286ms**（v10 no_read）。
- 发射后先 `t.sum()`（读输出）再 sync：**35µs**（v10 read），且读到正确数据；
  事件 ~3.2ms 触发、sync 后 device 即空闲。
- 同一个 collective，消费输出让 device 在 ~3ms 就「空闲」，不消费则 sync 等 ~280ms。
  机制未定（疑与本构建的延迟执行 / 完成信号行为有关），**不影响结论 1/2/4/6**：
  - 结论 1/2（wait/future 提前返回）用 sync 墙钟直接测量，无事件参与；
  - 结论 6（读数据正确）在两次读之间独立成立，是 PyTorch 训练正确性的既有保证。
- 对当前 scheduler：`_ensure_gpu_complete`（`Work.wait()` 后补 `torch.cuda.synchronize()`）
  **必须保留**——单独 wait 不可靠（结论 1），且 scheduler 在独立 comm stream 上代发射、
  consumer 跨流等待，没有「同行消费自排序」的便利。

### 对架构讨论的更新映射

原映射三档（stream-ordered 生效 / 仅 wait 不可靠 / 全不可靠）实测落在**中间偏后**：

- **数据正确性（admission-gate 的零 sync 前提）成立**：训练流发射 + 同行消费，
  c10d 保证读到正确数据（结论 6），gate 架构常见路径无需 wait、无需 sync。
- **完成信号全不可靠**：wait（#1）、future（#2）、当前流 event（#4）都不追踪 GPU
  完成。完成门控策略（outstanding 上限、delay 到某 group 完成）**没有廉价信号**，
  选项：转**程序序 checkpoint 限流**（纯 CPU，把 `max_outstanding` 从「GPU 在途数」
  改成「程序序未消费数」），或接受**少量定向 sync**（sync 空闲时 µs，代价与真实
  在途工作成正比）。
- M5 `finish_grad_sync` 的逐 bucket `work.wait()` 若保留当前 scheduler 设计，每处
  全同步的可用性风险**仍成立**；改为 gate 架构 + 程序序限流可绕开。
