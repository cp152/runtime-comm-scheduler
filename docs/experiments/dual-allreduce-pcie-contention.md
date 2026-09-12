# 双 Communicator All-Reduce 的 PCIe 争用实验

日期：2026-09-12

状态：**完成（2× RTX 3080 Ti）**

## 1. 问题与预判

问题是：同一对 GPU 上的两个独立 all-reduce 同时执行时，共享 PCIe 路径是否会
产生超加性退化，即并发 makespan 大于两个任务各自单独执行耗时之和
（`1 + 1 > 2`）。这个结果决定后续 scheduler 是否应该默认避免跨 communicator
并发。

实验前的预判是：大消息会共享 PCIe 带宽，并发 makespan 接近单任务耗时之和，
并可能因两个 NCCL communicator 争用 PCIe copy engine、channel 和 SM 资源而高出
约 5%～25%；小消息可能因固定开销重叠略有收益。

实际结果否定了“大消息具有整体 `1 + 1 > 2`”的预判：本机上两个任务并发的总
makespan 比单独耗时之和低约 13%～15%。但是，每个任务自己的完成延迟增加约
70%～73%。因此吞吐目标与关键路径延迟目标会得到不同的调度决策。

## 2. 环境与实际传输路径

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA GeForce RTX 3080 Ti，12 GiB |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| NCCL | 2.29.7 |
| GPU topology | `NODE`，无 NVLink |
| GPU P2P read/write | `CNS`（chipset not supported） |
| NCCL transport | `SHM/direct/direct` |
| NCCL topology path | `PHB`，模型带宽约 12 GB/s |
| NCCL channels | 每 communicator 2 个 collective channels |

`nvidia-smi topo -p2p r/w` 表明两卡不能直接 CUDA P2P。NCCL INFO 日志明确记录
两个 communicator 的 channel 均使用：

```text
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
```

因此本实验观察的是经 host shared memory 的 PCIe 路径争用，不是 NVLink，也不是
跨节点网络争用。

## 3. 实验结构

新增 harness：

- [`examples/dual_allreduce_contention_worker.py`](../../examples/dual_allreduce_contention_worker.py)
- [`examples/run_dual_allreduce_contention.py`](../../examples/run_dual_allreduce_contention.py)

两个 rank 各绑定一张 GPU。默认 ProcessGroup 使用 Gloo，仅承担轮次间 CPU barrier；
另外创建两个成员完全相同的 NCCL ProcessGroup：

```text
group A = {rank 0, rank 1}
group B = {rank 0, rank 1}
```

每组使用独立 tensor、独立 communicator 和独立 caller CUDA stream。两 rank 始终以
相同 host 顺序调用 A、B，避免 collective sequence divergence。

每个 payload 测量四种模式：

1. `single_a`：只有 group A all-reduce；
2. `single_b`：只有 group B all-reduce；
3. `sequential`：B 的 caller stream 显式等待 A completion，再发射 B；
4. `concurrent`：先快速提交 A、B，再分别安装 completion wait，两者允许在 GPU
   和 PCIe 路径上重叠。

CUDA event 从共同 start event 测到各任务的 completion event。每轮结果取两个 rank
中较大的时间作为全局 makespan。正式复验使用 6 轮预热、50 轮测量；四种模式的
执行顺序逐轮旋转，减少温度、频率和固定先后顺序偏差。

主要比值为：

```text
concurrent_over_isolated_sum =
    concurrent_makespan / (single_a_time + single_b_time)
```

- `> 1`：出现问题所述的整体 `1 + 1 > 2`；
- `≈ 1`：共享资源使两个任务近似完全串行；
- `< 1`：并发提高 aggregate utilization，存在有效重叠收益。

## 4. 结果

### 4.1 50 轮正式复验

下表时间均为两 rank 全局 makespan 的中位数：

| 每个 all-reduce | A 单独 | B 单独 | 单独耗时和 | 强制顺序 | 并发 | 并发/单独和（P10–P90） | 并发/顺序 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 MiB | 0.259 ms | 0.260 ms | 0.521 ms | 0.442 ms | 0.391 ms | 0.748（0.716–0.796） | 0.881 |
| 16 MiB | 2.508 ms | 2.519 ms | 5.027 ms | 4.945 ms | 4.349 ms | 0.865（0.855–0.876） | 0.880 |
| 64 MiB | 9.705 ms | 9.703 ms | 19.405 ms | 19.364 ms | 16.705 ms | 0.861（0.854–0.869） | 0.863 |
| 256 MiB | 38.335 ms | 38.323 ms | 76.682 ms | 76.444 ms | 65.453 ms | 0.854（0.839–0.866） | 0.855 |

所有轮次的数值结果均正确。16–256 MiB 时强制顺序时间与单独耗时和基本一致，说明
baseline 口径合理。并发 makespan 稳定低于二者，未观察到整体超加性退化。

另一份独立 25 轮 run 的 `concurrent / isolated_sum` 中位数分别为：

```text
1 MiB:   0.759
16 MiB:  0.870
64 MiB:  0.869
256 MiB: 0.852
```

与 50 轮复验一致。

### 4.2 单任务延迟膨胀

并发 makespan 改善不代表每个任务没有受 PCIe 争用影响：

| 每个 all-reduce | 单独延迟 | 并发 A 完成 | 并发 B 完成 | A slowdown | B slowdown |
|---:|---:|---:|---:|---:|---:|
| 1 MiB | 约 0.26 ms | 0.327 ms | 0.391 ms | 1.26× | 1.50× |
| 16 MiB | 约 2.51 ms | 4.337 ms | 4.349 ms | 1.73× | 1.73× |
| 64 MiB | 约 9.70 ms | 16.690 ms | 16.705 ms | 1.72× | 1.72× |
| 256 MiB | 约 38.33 ms | 65.238 ms | 65.452 ms | 1.70× | 1.71× |

两个大任务几乎同时完成，说明它们确实在大部分执行区间共享链路，而不是由 NCCL
完全串行化。每个任务大约慢 1.7×，但两者重叠后的总 makespan 只有顺序执行的
约 85%，相当于 aggregate throughput 提升约 17%。

## 5. 对调度策略的含义

本机结果不支持“只要两个 communicator 共享 PCIe，就必须默认互斥”的策略：

- 如果目标是提高一个 window 的 aggregate throughput，两个同等优先级的大
  all-reduce 并发有约 13%～15% makespan 收益；盲目串行化会损失这部分吞吐。
- 如果其中一个 collective 位于关键路径，并发会把它的自身延迟放大约 70%；此时
  推迟非关键任务、让关键任务独占链路可能更好。
- 因此 scheduler 应把 PCIe/SHM path 视为共享容量资源，并同时考虑 task criticality
  和 aggregate utilization，而不是只使用“允许并发/禁止并发”二元规则。

一个保守的后续策略可以是：

```text
两个任务都非关键、优先级接近：允许并发
一个任务关键、另一个有 slack：关键任务运行期间延迟另一个任务
链路已有多个 inflight task：通过实测 slowdown curve 决定 admission 上限
```

当前 M4.5 的单 host launch worker 与 per-group gate stream 已经允许这种策略：host
提交顺序仍保持确定性，但不同 group 的 GPU execution 可以重叠。后续策略层只需决定
何时 admission，而不需要改回多 host worker。

## 6. 适用边界

这不是所有部署环境的通用常数：

- 本实验仅覆盖两个相同成员集合的 communicator、两个并发任务；三个或更多任务
  可能进入更严重的饱和区间，甚至出现 `1 + 1 > 2` 型退化。
- NVLink、支持 PCIe P2P 的机器、多 NUMA/跨 socket、跨节点 InfiniBand 会得到不同
  slowdown curve。
- 相同 communicator 上的 collective 本来就有严格顺序，不能用本实验推断其并发。
- 真实训练还会同时竞争 SM、HBM、PCIe 的参数 offload/checkpoint traffic，需要在
  M5 adapter 接入后用真实 workload 复验。

因此本结果适合作为“不要预先禁止跨 communicator 并发”的证据，但 scheduler 最终
仍应按目标硬件建立 contention profile。

## 7. 复现

在两张可见 GPU 的机器上，从仓库根目录运行：

```bash
python examples/run_dual_allreduce_contention.py \
  --sizes-mib 1,16,64,256 \
  --rounds 50 \
  --warmup-rounds 6 \
  --output artifacts/dual-allreduce-contention.json
```

runner 会自行启动两个 rank，不需要 `torchrun`。它还会显式移除
`TORCH_NCCL_BLOCKING_WAIT`，避免 blocking wait 把两个 host launch 串行化。

本次原始结果保存在：

```text
artifacts/gpu3080-contention/dual-allreduce-contention-run1.json
artifacts/gpu3080-contention/dual-allreduce-contention-run2.json
```

`artifacts/` 被 `.gitignore` 排除，文档中的汇总数据可进入版本控制，原始逐轮 JSON
保留在本地实验产物中。
