# Wait 语义重测：各实验原始输出记录（v1–v10）

> 本文档**逐字记录** 2026-08-24 在 AutoDL 2×3090 盒子上跑 wait 语义重测各版本 worker
> 的原始 stdout（每 rank 一个 JSON，driver 用 `json.dumps(..., indent=2)` 打印）。
> 供逐组核对时间戳。分析结论见 `docs/todo-revisit-wait-semantics.md`「重测结果」。

## 环境（所有版本一致）

| 项               | 值                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 平台             | AutoDL/seetacloud，2× RTX 3090（24GB），NVIDIA driver 595.71.05                                                          |
| Python / PyTorch | `/root/miniconda3/bin/python` / torch 2.12.1+cu130（CUDA 13.2）                                                         |
| NCCL             | 可用；两 rank，每进程`CUDA_VISIBLE_DEVICES=<rank>` 绑单卡                                                               |
| rendezvous       | `env://`（`MASTER_ADDR=127.0.0.1` + 随机端口）                                                                        |
| 拓扑             | **无 NVLink**（`nvidia-smi nvlink` 全 inactive）；`topo -m` GPU0↔GPU1 = PXB（PCIe 多桥）；空闲 PCIe link gen=1 |
| 测量流           | 非 legacy 显式流`torch.cuda.Stream()`；**无 profiler**                                                            |

所有脚本位于 `examples/`（盒子 `.../examples/`）。2GiB = `1<<29` float32。

## 运行方式与约定

- v1 用 `run_wait_recheck.py`（指向 `wait_recheck_worker.py`）；
  v2–v10 用 `run_wait_recheck_v2.py <worker名>`。
- `sync_after_X_s`：信号 X 返回后**立即** `torch.cuda.synchronize()` 的残余墙钟。
  > 0 表示 X 返回/触发时 GPU 仍有工作（X 未反映完成）；≈0 表示当时已空闲。
  >
- `ev_X_s`：事件 X 从 record 到 `query()` 为真的轮询墙钟（sleep 1ms 粒度）。
- reduce 真值：rank0 fill=1.0、rank1 fill=2.0，SUM 后均为 3.0；
  `3*N = 3*(1<<29) = 1610612736`；`3*1024 = 3072`。

---

## v1 — 原始三实验（back-to-back）

脚本 `wait_recheck_worker.py`。在同一流上连发三个 2GiB collective：
exp1（当前流 event + 立即 query）、exp2（裸 `Work.wait()`）、exp3（`get_future().wait()`），
各自配独立 ground-truth event。**注意：三个 collective 背靠背，GPU 排队（v2 起已消除该污染）。**

```json
{
  "rank": 0,
  "world": 2,
  "exp1": {
    "q0_immediate": false,
    "gt1_s": 0.002124,
    "ok1": true
  },
  "exp2": {
    "wait_s": 1e-05,
    "gt2_s": 0.290202,
    "ok2": true
  },
  "exp3": {
    "fut_s": 4.4e-05,
    "gt3_s": 0.28309
  }
}
{
  "rank": 1,
  "world": 2,
  "exp1": {
    "q0_immediate": false,
    "gt1_s": 0.002122,
    "ok1": true
  },
  "exp2": {
    "wait_s": 1e-05,
    "gt2_s": 0.288394,
    "ok2": true
  },
  "exp3": {
    "fut_s": 5.2e-05,
    "gt3_s": 0.284309
  }
}
```

## v2 — 孤立测量（每段 sync 重置）

脚本 `wait_recheck_worker_v2.py`。每个测量前 `torch.cuda.synchronize()` 重置 GPU，
只测一个 fresh collective；`sync_after_*` 表示该信号返回时剩余 GPU 工作。

- `fill_s`：2GiB fill 本地基线。
- S1：当前流 event（q0_immediate 立即 query、ev_s 触发、sync_after_ev_s）。
- S2：裸 `Work.wait()`（wait_s + sync_after_wait_s）。
- S3：`get_future().wait()`（fut_s + sync_after_fut_s）。
- S4：`sync_total_s`（collective 真实时长参照）。

```json
{
  "rank": 0,
  "world": 2,
  "fill_s": 0.003206,
  "S1": {
    "q0_immediate": false,
    "ev_s": 0.003173,
    "sync_after_ev_s": 0.275244
  },
  "S2": {
    "wait_s": 9e-06,
    "sync_after_wait_s": 0.285297,
    "ok": true
  },
  "S3": {
    "fut_s": 1.1e-05,
    "sync_after_fut_s": 0.278593
  },
  "S4": {
    "sync_total_s": 0.2804
  }
}
{
  "rank": 1,
  "world": 2,
  "fill_s": 0.003191,
  "S1": {
    "q0_immediate": false,
    "ev_s": 0.003174,
    "sync_after_ev_s": 0.275333
  },
  "S2": {
    "wait_s": 1e-05,
    "sync_after_wait_s": 0.285347,
    "ok": true
  },
  "S3": {
    "fut_s": 9e-06,
    "sync_after_fut_s": 0.278366
  },
  "S4": {
    "sync_total_s": 0.280359
  }
}
```

## v3 — 竞争测试（真实计算读 tensor）

脚本 `wait_recheck_worker_v3.py`。2GiB all_reduce 后同行立刻 `t.sum()`（读 tensor，
非 `event.record`），sync 后取 `computed`。`seen_reduced`：读到 reduce 后真值
（3*N）；`seen_unreduced_rank_fill`：读到未 reduce 的 fill 值（rank+1）*N。

```json
{
  "rank": 0,
  "world": 2,
  "expected": 1610612736,
  "race": {
    "ev_s": 0.003182,
    "computed": 1610612736.0,
    "seen_reduced": true,
    "seen_unreduced_rank_fill": false
  }
}
{
  "rank": 1,
  "world": 2,
  "expected": 1610612736,
  "race": {
    "ev_s": 0.003192,
    "computed": 1610612736.0,
    "seen_reduced": true,
    "seen_unreduced_rank_fill": false
  }
}
```

## v4 — 双事件 + comm stream 探测

脚本 `wait_recheck_worker_v4.py`。

- A：AR1 后 `ev_a`（AR 发射点后）→ `t.sum()` → `ev_b`（sum 后）；
  `computed`/`correct` = sum 读到的是否为 reduce 真值。
- B：AR1 后立刻发 AR2（1024，排 AR1 后），s 上 `ev` 轮询 + sync；
  `ev_ar2_s` + `sync_after_ar2_s` 反映 comm stream 何时释放。

```json
{
  "rank": 0,
  "world": 2,
  "expected": 1610612736,
  "A": {
    "ev_a_s": 8e-06,
    "ev_b_s": 0.003195,
    "computed": 1610612736.0,
    "correct": true
  },
  "B": {
    "ev_ar2_s": 0.00214,
    "sync_after_ar2_s": 0.282632
  }
}
{
  "rank": 1,
  "world": 2,
  "expected": 1610612736,
  "A": {
    "ev_a_s": 8e-06,
    "ev_b_s": 0.003178,
    "computed": 1610612736.0,
    "correct": true
  },
  "B": {
    "ev_ar2_s": 0.002139,
    "sync_after_ar2_s": 0.282637
  }
}
```

> 注：v4 首跑因 `w2.wait(timeout=10)` 传入 int 触发 c10d 签名错误失败，无输出；
> 删除该行后重跑，上为重跑结果。

## v5 — sync 本身是否慢

脚本 `wait_recheck_worker_v5.py`。对三种状态计时 `torch.cuda.synchronize()`：
完全空闲、极小 op（0-dim 写）、2GiB fill（无 collective）、极小 all_reduce（1024）。

```json
{
  "rank": 0,
  "idle_sync_s": 3.4e-05,
  "tiny_sync_s": 4.9e-05,
  "fill_sync_s": 0.002414,
  "tiny_ar_sync_s": 0.00012
}
{
  "rank": 1,
  "idle_sync_s": 1.9e-05,
  "tiny_sync_s": 5.5e-05,
  "fill_sync_s": 0.002408,
  "tiny_ar_sync_s": 0.000312
}
```

## v6 — event 可靠性控制

脚本 `wait_recheck_worker_v6.py`。C：空流 event（对照）；A：~3ms fill 后 event
（应 ~3ms 触发）；B：all_reduce 前后 `current_stream == s`（cudaid 对比）。

```json
{
  "rank": 0,
  "C_empty_stream_event_s": 1e-05,
  "A_event_after_fill_s": 0.003201,
  "B_cur_is_s_before": true,
  "B_cur_is_s_after": true,
  "B_cur_stream_eq_before_after": true,
  "A2_event_after_ar_s": 4e-06,
  "B_cur_stream_cudaid": 218513808,
  "B_s_cudaid": 218513808
}
{
  "rank": 1,
  "C_empty_stream_event_s": 1e-05,
  "A_event_after_fill_s": 0.003197,
  "B_cur_is_s_before": true,
  "B_cur_is_s_after": true,
  "B_cur_stream_eq_before_after": true,
  "A2_event_after_ar_s": 2e-06,
  "B_cur_stream_cudaid": 224203840,
  "B_s_cudaid": 224203840
}
```

## v7 — all_reduce enqueue 是否 flush 调用方流

脚本 `wait_recheck_worker_v7.py`。`ctrl_event_after_fill_s`（对照，fill→event）；
`enqueue_wall_s`（fill 后调 all_reduce 的返回墙钟）；`ev_after_ar_s`（fill→AR→event）。

```json
{
  "rank": 0,
  "ctrl_event_after_fill_s": 0.003195,
  "enqueue_wall_s": 0.000469,
  "ev_after_ar_s": 0.002121
}
{
  "rank": 1,
  "ctrl_event_after_fill_s": 0.003188,
  "enqueue_wall_s": 0.000347,
  "ev_after_ar_s": 0.002122
}
```

## v8 — comm stream 占用探测（读输出）

脚本 `wait_recheck_worker_v8.py`。AR1（2GiB）后立刻 AR2（1024），立刻读两者
（`p.sum()`、`t.sum()`）；`sync_after_s` 反映读后设备是否空闲。

```json
{
  "rank": 0,
  "expected_p": 3072,
  "expected_t": 1610612736,
  "race2": {
    "ev_s": 0.003184,
    "sync_after_s": 4.1e-05,
    "p_computed": 3072.0,
    "p_ok": true,
    "t_computed": 1610612736.0,
    "t_ok": true
  }
}
{
  "rank": 1,
  "expected_p": 3072,
  "expected_t": 1610612736,
  "race2": {
    "ev_s": 0.003185,
    "sync_after_s": 4.1e-05,
    "p_computed": 3072.0,
    "p_ok": true,
    "t_computed": 1610612736.0,
    "t_ok": true
  }
}
```

## v9 — 稳态吞吐（一次性 init?）

脚本 `wait_recheck_worker_v9.py`。无 throwaway；`first_sync_s`（1024 warmup 后首个
大 collective）、`steady_sync_s`（随后 3 个 2GiB all_reduce 各自 sync 墙钟）。

```json
{
  "rank": 0,
  "first_sync_s": 0.285059,
  "steady_sync_s": [
    0.285297,
    0.283981,
    0.28039
  ]
}
{
  "rank": 1,
  "first_sync_s": 0.285031,
  "steady_sync_s": [
    0.285279,
    0.28399,
    0.28037
  ]
}
```

## v10 — 读输出 vs 不读输出（对照）

脚本 `wait_recheck_worker_v10.py`。`no_read_sync_s`（fill→AR→立即 sync，不读）；
`read_sync_s`（fill→AR→`t.sum()`→sync）；`read_ev_s`、`read_sum_correct`。

```json
{
  "rank": 0,
  "expected": 1610612736,
  "no_read_sync_s": 0.286315,
  "read_sync_s": 3.5e-05,
  "read_ev_s": 0.003182,
  "read_sum_correct": true
}
{
  "rank": 1,
  "expected": 1610612736,
  "no_read_sync_s": 0.287831,
  "read_sync_s": 2.8e-05,
  "read_ev_s": 0.003174,
  "read_sum_correct": true
}
```

---

## 备注

- 全部输出为当次运行的 stdout，未经任何换算；秒为单位。
- v1 的完整输出在会话边界后从 compaction 前 transcript 恢复，逐字一致
  （`ok1`/`ok2` 两 rank 均 `true`；exp3 无 ok 字段）。
- 3090 盒子在 2026-08-24 跑完后已休眠（重测时 `ssh gpu3090` 端口拒绝连接）；
  如需复跑任一组，先在 AutoDL 控制台开机。

---

## 2026-09-08 新盒子（2× RTX 3080 Ti，PIX）：wait 语义再探 v11–v15

> 新开容器，`autodl-tmp` 为空，脚本从本地 scp 过去。跑的是**上一节 3090 盒子问题
> 的延续**：用户修正了对 `Work.wait()` 语义的理解（见下「文档确认」），并想回到
> v3，用**逐语句 CPU 时间线**弄清「`t.sum()` 能读到正确值但 event 却 ~ms 触发」时
> CPU 到底在哪条语句被阻塞。v11–v15 原始输出如下（stdout 逐字，driver 已重跑落盘）。

### 环境（与 3090 盒子不同点）

| 项 | 值 |
|---|---|
| 平台 | AutoDL 新容器，**2× RTX 3080 Ti**（12GB），driver 595.71.05，CUDA 13.2 |
| Python / PyTorch | `/root/miniconda3/bin/python` / torch **2.12.1+cu130**（同 3090 盒子） |
| 拓扑 | `nvidia-smi topo -m`：GPU0↔GPU1 = **PIX**（≤1 个 PCIe 桥，比 3090 盒子的 PXB 干净）；**无 NVLink** |
| NCCL | 两 rank，每进程 `CUDA_VISIBLE_DEVICES=<rank>` 绑单卡，`env://` |
| 测量流 | 非 legacy 显式流 `torch.cuda.Stream()`；无 profiler |

**2GiB（`1<<29` float32）all_reduce 稳态墙钟 ≈ 500ms**（3090 盒子为 ~280ms，本盒
反而更慢；fill 单独 2.45ms）。正确性真值同前：`3*N=1610612736`；未 reduce 时
rank0 读 `1*N=536870912`、rank1 读 `2*N=1073741824`。

### 文档确认（`Work.wait()` 语义）

`torch.distributed.Work.wait` docstring 原文：

> "calling wait() is the same as calling synchronize(): **Letting the current stream
> block on the completion of the NCCL work.** However, if timeout is set, it will
> block the CPU thread until the NCCL work is completed or timed out."

→ **wait() = 让「当前流」阻塞在 NCCL work 完成上（设备侧流序），不阻塞 CPU 线程**；
只有设 `timeout` 才阻塞 CPU。本节实测与此一致：`wait()` 均 ~10–40µs 返回、
`is_completed=False`（GPU 尚未完成）。

### 时间线字段说明（v11–v13）

每条 mark 是 (label, CPU `perf_counter`)；`dt_ms` = 距上一条的 CPU 墙钟，
`cum_ms` = 距场景起点。**dt 大 = 该条语句 CPU 真被阻塞**。导出量：
`sum_cpu_ms` = `t.sum()` 调用自身 CPU 阻塞；`item_block_ms` = `sx.item()` 阻塞；
`sync_after_item_ms` = item 后再 sync 的剩余；`wait_ms` = `w.wait()` 阻塞。

---

### v11 — v3 复刻 + 逐语句 CPU 时间线

脚本 `wait_recheck_worker_v11.py`。场景（每场景前 `synchronize()` 复位）：
T_fill（fill 墙钟）、T_noread（fill→AR→sync 不读）、T_item_nowait（v3 原样：AR
后**不 wait** 直接 sum→item→sync）、T_item_afterwait（AR 后**先 wait()** 再 sum）。

```json
{
  "rank": 0,
  "world": 2,
  "expected": 1610612736,
  "scen": {
    "T_fill": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"pre_sync","cum_ms":0.6122,"dt_ms":0.6122},
      {"st":"post_sync","cum_ms":3.0642,"dt_ms":2.452}], "fill_gpu_ms": 2.452 },
    "T_noread": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.0377,"dt_ms":0.0377},
      {"st":"pre_sync","cum_ms":0.3537,"dt_ms":0.316},
      {"st":"post_sync","cum_ms":503.3699,"dt_ms":503.0162}], "sync_ms": 503.0162 },
    "T_item_nowait": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.0722,"dt_ms":0.0722},
      {"st":"sum_launch","cum_ms":0.2446,"dt_ms":0.1725},
      {"st":"ev_record","cum_ms":497.2283,"dt_ms":496.9837},
      {"st":"item_in","cum_ms":497.3855,"dt_ms":0.1572},
      {"st":"item_out","cum_ms":499.6932,"dt_ms":2.3076},
      {"st":"sync_out","cum_ms":499.7403,"dt_ms":0.0471}],
      "computed": 1610612736.0, "correct": true,
      "item_block_ms": 2.3076, "sync_after_item_ms": 0.0471, "ev_fired_by_end": true },
    "T_item_afterwait": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.3256,"dt_ms":0.3256},
      {"st":"wait_in","cum_ms":0.5528,"dt_ms":0.2272},
      {"st":"wait_out","cum_ms":0.5809,"dt_ms":0.0281},
      {"st":"item_in","cum_ms":0.6523,"dt_ms":0.0714},
      {"st":"item_out","cum_ms":493.4867,"dt_ms":492.8344},
      {"st":"sync_out","cum_ms":493.5553,"dt_ms":0.0687}],
      "computed": 1610612736.0, "correct": true,
      "wait_ms": 0.0281, "is_completed_after_wait": false,
      "item_block_ms": 492.8344, "sync_after_item_ms": 0.0687 }
  }
}
{
  "rank": 1,
  "world": 2,
  "expected": 1610612736,
  "scen": {
    "T_fill": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"pre_sync","cum_ms":0.783,"dt_ms":0.783},
      {"st":"post_sync","cum_ms":3.2371,"dt_ms":2.4541}], "fill_gpu_ms": 2.4541 },
    "T_noread": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.0424,"dt_ms":0.0424},
      {"st":"pre_sync","cum_ms":0.5041,"dt_ms":0.4617},
      {"st":"post_sync","cum_ms":503.1466,"dt_ms":502.6425}], "sync_ms": 502.6425 },
    "T_item_nowait": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.0636,"dt_ms":0.0636},
      {"st":"sum_launch","cum_ms":0.2481,"dt_ms":0.1845},
      {"st":"ev_record","cum_ms":497.0877,"dt_ms":496.8396},
      {"st":"item_in","cum_ms":497.4487,"dt_ms":0.361},
      {"st":"item_out","cum_ms":499.5482,"dt_ms":2.0995},
      {"st":"sync_out","cum_ms":499.6264,"dt_ms":0.0783}],
      "computed": 1610612736.0, "correct": true,
      "item_block_ms": 2.0995, "sync_after_item_ms": 0.0783, "ev_fired_by_end": true },
    "T_item_afterwait": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.3588,"dt_ms":0.3588},
      {"st":"wait_in","cum_ms":0.7348,"dt_ms":0.376},
      {"st":"wait_out","cum_ms":0.7639,"dt_ms":0.0291},
      {"st":"item_in","cum_ms":0.8409,"dt_ms":0.077},
      {"st":"item_out","cum_ms":493.4106,"dt_ms":492.5696},
      {"st":"sync_out","cum_ms":493.4734,"dt_ms":0.0628}],
      "computed": 1610612736.0, "correct": true,
      "wait_ms": 0.0291, "is_completed_after_wait": false,
      "item_block_ms": 492.5696, "sync_after_item_ms": 0.0628 }
  }
}
```

v11 观测：
- AR 墙钟 ≈ 500ms（T_noread sync=503ms）；fill 2.45ms。
- **T_item_nowait**：CPU 挡在 `sx = t.sum()` 内（sum_launch→ev_record dt=497ms，≈ 整个
  collective），item 只再挡 2.3ms → **此 run 里不 wait 时 sum 也「自动挡满等完」并读对**。
- **T_item_afterwait**：wait 28µs 返回、`is_completed=False`；sum 0.07ms 返回；item 挡
  493ms 读对 → wait 只插流序不挡 CPU，CPU 代价在消费点（item）才付。与文档语义一致。

---

### v12 — 两个归属判别

脚本 `wait_recheck_worker_v12.py`。S_ctrl_sum：fill 后**无 collective** 直接 sum→item
（测 sum 是否固有 CPU 阻塞）；S_ev_after_ar：AR 后立刻 `ev.record()`+轮询、不消费输出
（测 event 是否追踪 collective）。

```json
{
  "rank": 0,
  "world": 2,
  "expected": 1610612736,
  "scen": {
    "S_ctrl_sum": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":1.0376,"dt_ms":1.0376},
      {"st":"sum_out","cum_ms":16.2415,"dt_ms":15.2039},
      {"st":"item_out","cum_ms":18.7104,"dt_ms":2.4689},
      {"st":"sync_out","cum_ms":18.7693,"dt_ms":0.0589}],
      "sum_cpu_ms": 15.2039, "item_block_ms": 2.4689, "correct": false },
    "S_ev_after_ar": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.0559,"dt_ms":0.0559},
      {"st":"ev_rec","cum_ms":0.6129,"dt_ms":0.557},
      {"st":"poll_done","cum_ms":2.8677,"dt_ms":2.2548},
      {"st":"sync_out","cum_ms":507.2143,"dt_ms":504.3466}], "ev_s": 2.1436 }
  }
}
{
  "rank": 1,
  "world": 2,
  "expected": 1610612736,
  "scen": {
    "S_ctrl_sum": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":0.9898,"dt_ms":0.9898},
      {"st":"sum_out","cum_ms":16.0515,"dt_ms":15.0617},
      {"st":"item_out","cum_ms":18.5194,"dt_ms":2.468},
      {"st":"sync_out","cum_ms":18.5881,"dt_ms":0.0687}],
      "sum_cpu_ms": 15.0617, "item_block_ms": 2.468, "correct": false },
    "S_ev_after_ar": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"ar_launch","cum_ms":0.0627,"dt_ms":0.0627},
      {"st":"ev_rec","cum_ms":0.4743,"dt_ms":0.4115},
      {"st":"poll_done","cum_ms":2.7192,"dt_ms":2.2449},
      {"st":"sync_out","cum_ms":507.4674,"dt_ms":504.7481}], "ev_s": 2.1563 }
  }
}
```

v12 观测：
- **S_ctrl_sum**：无 collective、GPU 只有 2.4ms fill 在途时，`t.sum()` 也挡 **~15ms**
  CPU（远超在途 fill 时长）→ sum 的 CPU 成本不是固定的、且不是简单「等在途 GPU 工作」。
  ⚠ `correct:false` 是脚本笔误（该场景无 reduce，期望值应为 `(rank+1)*N` 而非 `3*N`），
  数据值本身未记录、无意义，忽略。
- **S_ev_after_ar**：AR 后立刻 record 的 event ~2.1–3.2ms 触发（≈ fill 时间），其后 sync
  仍挡 ~504ms → **event 不追踪 collective**（3090 盒子的「结论 4」在本盒复现，跨盒子成立）。

---

### v13 — 2×2 归因：sum 的 CPU 挡块跟什么走

脚本 `wait_recheck_worker_v13.py`。S_A fill 在途即 sum；S_B 先 sync（GPU 空）再 sum；
S_C AR 在途不 wait 即 sum；S_D AR 在途先 wait 再 sum。

```json
{
  "rank": 0,
  "world": 2,
  "scen": {
    "S_A_fill_inflight": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":1.0844,"dt_ms":1.0844},
      {"st":"sum_out","cum_ms":16.8537,"dt_ms":15.7694},
      {"st":"item_out","cum_ms":19.3206,"dt_ms":2.4669},
      {"st":"sync_out","cum_ms":19.3832,"dt_ms":0.0625}],
      "sum_cpu_ms": 15.7694, "item_block_ms": 2.4669 },
    "S_B_fill_done": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":2.5262,"dt_ms":2.5262},
      {"st":"sum_out","cum_ms":2.5798,"dt_ms":0.0536},
      {"st":"item_out","cum_ms":5.0708,"dt_ms":2.491},
      {"st":"sync_out","cum_ms":5.0864,"dt_ms":0.0156}],
      "sum_cpu_ms": 0.0536, "item_block_ms": 2.491 },
    "S_C_ar_inflight": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":0.6089,"dt_ms":0.6089},
      {"st":"sum_out","cum_ms":0.6574,"dt_ms":0.0485},
      {"st":"item_out","cum_ms":5.1345,"dt_ms":4.4771},
      {"st":"sync_out","cum_ms":491.9297,"dt_ms":486.7951}],
      "sum_cpu_ms": 0.0485, "item_block_ms": 4.4771 },
    "S_D_ar_waited": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":0.7022,"dt_ms":0.7022},
      {"st":"sum_out","cum_ms":0.7672,"dt_ms":0.0651},
      {"st":"item_out","cum_ms":507.5654,"dt_ms":506.7981},
      {"st":"sync_out","cum_ms":507.7345,"dt_ms":0.1692}],
      "sum_cpu_ms": 0.0651, "item_block_ms": 506.7981 }
  }
}
{
  "rank": 1,
  "world": 2,
  "scen": {
    "S_A_fill_inflight": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":1.0478,"dt_ms":1.0478},
      {"st":"sum_out","cum_ms":16.7527,"dt_ms":15.7049},
      {"st":"item_out","cum_ms":19.2282,"dt_ms":2.4755},
      {"st":"sync_out","cum_ms":19.2874,"dt_ms":0.0592}],
      "sum_cpu_ms": 15.7049, "item_block_ms": 2.4755 },
    "S_B_fill_done": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":2.5287,"dt_ms":2.5287},
      {"st":"sum_out","cum_ms":2.5907,"dt_ms":0.062},
      {"st":"item_out","cum_ms":5.0799,"dt_ms":2.4891},
      {"st":"sync_out","cum_ms":5.1019,"dt_ms":0.0221}],
      "sum_cpu_ms": 0.062, "item_block_ms": 2.4891 },
    "S_C_ar_inflight": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":0.5894,"dt_ms":0.5894},
      {"st":"sum_out","cum_ms":0.6412,"dt_ms":0.0518},
      {"st":"item_out","cum_ms":5.0151,"dt_ms":4.3739},
      {"st":"sync_out","cum_ms":491.9195,"dt_ms":486.9044}],
      "sum_cpu_ms": 0.0518, "item_block_ms": 4.3739 },
    "S_D_ar_waited": { "timeline": [{"st":"t0","cum_ms":0.0,"dt_ms":0.0},
      {"st":"sum_in","cum_ms":0.6791,"dt_ms":0.6791},
      {"st":"sum_out","cum_ms":0.742,"dt_ms":0.0629},
      {"st":"item_out","cum_ms":507.6721,"dt_ms":506.9301},
      {"st":"sync_out","cum_ms":508.0864,"dt_ms":0.4143}],
      "sum_cpu_ms": 0.0629, "item_block_ms": 506.9301 }
  }
}
```

v13 观测（关键矛盾）：
- S_A 15.8ms vs S_B 0.05ms → sum 的 CPU 挡块与「GPU 是否有在途工作」有关，但量级远超
  在途 fill（15ms ≫ 2.4ms），具体机制未明。
- **S_C**（AR 在途、不 wait）：sum 只挡 0.048ms 就返回、item 挡 4.5ms、collective 到
  sync 还在跑（486ms）→ **与 v11 T_item_nowait（sum 挡满 497ms 读对）同构却相反结果**。
  → 「不 wait 时 sum 会不会等 collective」不是稳定的，引出 v14/v15。
- S_D：wait 后 sum 0.065ms 返回、item 挡 507ms → 与 v11 T_item_afterwait 一致。

---

### v14 — 重复性实验（决定性）

脚本 `wait_recheck_worker_v14.py`。同构场景连跑 6 遍（每遍 sync 复位：
fill→AR(async)→sum→item→sync，**不 wait**）。只重跑记录（本次落盘 log）：

```json
{
  "rank": 0,
  "world": 2,
  "expected": 1610612736,
  "reps": [
    { "rep": 0, "sum_cpu_ms": 500.6292, "item_block_ms": 2.4658, "sync_after_item_ms": 0.0728, "computed": 1610612736.0, "correct": true },
    { "rep": 1, "sum_cpu_ms": 0.0608, "item_block_ms": 5.1865, "sync_after_item_ms": 493.1607, "computed": 536870912.0, "correct": false },
    { "rep": 2, "sum_cpu_ms": 0.0568, "item_block_ms": 4.7731, "sync_after_item_ms": 488.6646, "computed": 536870912.0, "correct": false },
    { "rep": 3, "sum_cpu_ms": 0.0488, "item_block_ms": 4.7785, "sync_after_item_ms": 498.8556, "computed": 536870912.0, "correct": false },
    { "rep": 4, "sum_cpu_ms": 0.0577, "item_block_ms": 4.9762, "sync_after_item_ms": 484.894, "computed": 536870912.0, "correct": false },
    { "rep": 5, "sum_cpu_ms": 0.0465, "item_block_ms": 5.1516, "sync_after_item_ms": 503.4942, "computed": 536870912.0, "correct": false }
  ]
}
{
  "rank": 1,
  "world": 2,
  "expected": 1610612736,
  "reps": [
    { "rep": 0, "sum_cpu_ms": 500.9344, "item_block_ms": 2.4619, "sync_after_item_ms": 0.071, "computed": 1610612736.0, "correct": true },
    { "rep": 1, "sum_cpu_ms": 0.0624, "item_block_ms": 5.2998, "sync_after_item_ms": 493.0899, "computed": 1073741824.0, "correct": false },
    { "rep": 2, "sum_cpu_ms": 0.06, "item_block_ms": 4.8434, "sync_after_item_ms": 488.8007, "computed": 1073741824.0, "correct": false },
    { "rep": 3, "sum_cpu_ms": 0.0428, "item_block_ms": 4.9423, "sync_after_item_ms": 498.9095, "computed": 1073741824.0, "correct": false },
    { "rep": 4, "sum_cpu_ms": 0.0471, "item_block_ms": 5.1227, "sync_after_item_ms": 484.7574, "computed": 1073741824.0, "correct": false },
    { "rep": 5, "sum_cpu_ms": 0.0442, "item_block_ms": 5.2119, "sync_after_item_ms": 503.5744, "computed": 1073741824.0, "correct": false }
  ]
}
```

v14 结论（首轮 + 本次重跑均一致）：
- **rep0**：sum 挡 ~500ms（自动等完）、读到正确 3N。
- **rep1–5**：sum 0.05ms 就返回、item ~5ms 读到**未 reduce 数据**（rank0=1N、rank1=2N），
  collective 到 sync 还在跑（~490ms 剩余）。
→ v3/v8/v10 单发测的「sum 读对」只在特定历史位置成立；**一旦重复，不 wait 就读脏**。

---

### v15 — 交替 nowait/wait（决定性）

脚本 `wait_recheck_worker_v15.py`。8 遍交替：i 偶 = nowait、i 奇 = wait。

```json
{
  "rank": 0,
  "world": 2,
  "expected": 1610612736,
  "reps": [
    { "i": 0, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 488.4649, "item_block_ms": 2.4653, "computed": 1610612736.0, "correct": true },
    { "i": 1, "mode": "wait", "wait_ms": 0.0392, "sum_cpu_ms": 0.1013, "item_block_ms": 506.6503, "computed": 1610612736.0, "correct": true },
    { "i": 2, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 0.0621, "item_block_ms": 4.8267, "computed": 536870912.0, "correct": false },
    { "i": 3, "mode": "wait", "wait_ms": 0.0202, "sum_cpu_ms": 0.0583, "item_block_ms": 504.4591, "computed": 1610612736.0, "correct": true },
    { "i": 4, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 0.0578, "item_block_ms": 4.752, "computed": 536870912.0, "correct": false },
    { "i": 5, "mode": "wait", "wait_ms": 0.008, "sum_cpu_ms": 0.0618, "item_block_ms": 494.0125, "computed": 1610612736.0, "correct": true },
    { "i": 6, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 0.0566, "item_block_ms": 4.8733, "computed": 536870912.0, "correct": false },
    { "i": 7, "mode": "wait", "wait_ms": 0.0078, "sum_cpu_ms": 0.0622, "item_block_ms": 491.0619, "computed": 1610612736.0, "correct": true }
  ]
}
{
  "rank": 1,
  "world": 2,
  "expected": 1610612736,
  "reps": [
    { "i": 0, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 488.4817, "item_block_ms": 2.4622, "computed": 1610612736.0, "correct": true },
    { "i": 1, "mode": "wait", "wait_ms": 0.0165, "sum_cpu_ms": 0.0651, "item_block_ms": 506.5784, "computed": 1610612736.0, "correct": true },
    { "i": 2, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 0.0613, "item_block_ms": 4.7078, "computed": 1073741824.0, "correct": false },
    { "i": 3, "mode": "wait", "wait_ms": 0.0228, "sum_cpu_ms": 0.0598, "item_block_ms": 504.4545, "computed": 1610612736.0, "correct": true },
    { "i": 4, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 0.0581, "item_block_ms": 4.9055, "computed": 1073741824.0, "correct": false },
    { "i": 5, "mode": "wait", "wait_ms": 0.0077, "sum_cpu_ms": 0.0546, "item_block_ms": 493.9941, "computed": 1610612736.0, "correct": true },
    { "i": 6, "mode": "nowait", "wait_ms": 0.0, "sum_cpu_ms": 0.0539, "item_block_ms": 4.7687, "computed": 1073741824.0, "correct": false },
    { "i": 7, "mode": "wait", "wait_ms": 0.0082, "sum_cpu_ms": 0.0666, "item_block_ms": 491.0781, "computed": 1610612736.0, "correct": true }
  ]
}
```

v15 结论（决定性，两 rank 完全一致）：
- **从 i=1 起正确性严格跟随 `wait()`**：nowait（i=2,4,6）→ sum 50µs 返回、item ~5ms
  读脏（1N/2N）；wait（i=1,3,5,7）→ 全对，CPU 代价落在 item（≈500ms）。
- **i=0 nowait 是特例**：sum 自动挡满 ~488ms 读对（同 v14 rep0）。其触发条件≠简单的
  「首个 collective」——见下方未解。
- `wait_ms` 8–40µs、之后 item 才挡满 → **wait() = 插流序不挡 CPU**（文档语义实证）。

---

### 综合结论与未解（新盒子）

| # | 结论 | 证据 |
|---|---|---|
| A | `Work.wait()` 让**当前流**阻塞于 NCCL work，不阻塞 CPU 线程；代价在之后真正消费处（item/sync）才付 | v11/v13 S_D/v15 wait 列：wait 8–40µs + item 挡 ~500ms + `is_completed=False` |
| B | **`all_reduce(async_op=True)` 不会自动把调用方流排到 collective 之后**（显式非默认流上）；不 wait 就同流消费 → 确定性脏读 | v14 rep1–5 / v15 i=2,4,6：sum 50µs 返回，读到 1N/2N，collective 到 sync 还在跑 |
| C | 想正确必须显式 `w.wait()`（流序）；「自动挡满等完」只在特定历史位置出现 | v15 i=0/v14 rep0 特例 vs i≥1 nowait 全脏 |
| D | 当前流裸 event 不追踪 collective（~2–3ms 触发、sync 剩 ~500ms） | v12 S_ev_after_ar（跨 3090/3080Ti 两盒复现） |
| E | `t.sum()` 自身的 CPU 挡块是可变量：GPU 空时 ~0.05ms，fill 在途时 ~15ms，特定历史下 ~500ms → **不是干净探针** | v13 S_A/S_B、v11/v14 |

**对旧会话结论的修正**：旧「结论 6」——训练流自发射 all_reduce、同流紧跟计算读到正确
数据、admission-gate 零 sync 前提成立——**不成立/不稳健**：v3/v8/v10 都是单发测量，
命中「自动挡满」特例；重复即脏读（B）。Megatron 真实用法安全是因为它 async 后**不读**
tensor、在 bucket 边界显式 `work.wait()` 后才消费（靠 wait 而非自动流序）。

**未解（需下一步实验）**：
1. 「自动挡满读对」的精确触发条件。v11 T_item_nowait 是进程内第 3 个大 collective 仍自动
   挡满读对；v14 rep1（第 3 个）/v15 i=2（第 4 个）却脏读——历史依赖，触发边界未定
   （可能与前一个 collective 是否被消费、comm stream 上的 event 记录状态有关）。
2. `t.sum()` 在 GPU 有在途工作时挡 ~15ms 的机制（> 在途 fill 的 2.4ms 数倍）。
3. 以上全部在**显式非默认流**测得；**默认流上 c10d 是否自动插排序**未测——这直接决定
   「训练流自发射+自消费」在真实（默认流）场景是否安全，是下一轮第一个该做的对照。
4. 老 3090 盒 v10（throwaway 后第 2 个 collective）读也对，与新盒脏读不一致——可能盒子
   相关，3090 盒休眠中暂无法复验。
