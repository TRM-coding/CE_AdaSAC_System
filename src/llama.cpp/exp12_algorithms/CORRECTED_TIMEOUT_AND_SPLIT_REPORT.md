# Exp12 修正版 Local Split 与 Timeout 实验说明

日期：2026-04-28

## 结论摘要

前面几轮实验里有两类结果需要明确区分：

1. 早期 `alternate_0.75` 下 PPL 飙到 `69` 或更高的结果不代表调度设计本身有问题。原因是旧实现把 `rate` 当成“直接丢弃 tail rank”，导致 tail 在 local split 前就被截掉了。
2. 修复后，local split 的语义已经改回设计目标：major 和 minor 同时计算，major 计算 head/main rank，minor 计算 tail rank；只有 timeout 触发时，minor tail 才会被丢弃。
3. 修复后，在不触发 timeout 的情况下，SVD local split 的 PPL 回到 no-SVD baseline：`15.1424`。
4. 最近 `major=2 / minor=6 / minor 50% load` 这组里，timeout 不同但 no-load PPL 都一样，是合理的：那张 PPL 表是在无背景负载下测的，只能说明无负载时没有丢 tail。
5. 对同一 `major=2 / minor=6 / minor 50% load` decode 运行加 runtime 计数后，发现大多数慢点并不是 timeout 等待造成的。很多低 tok/s 样本中 `drop=0`，major 等 minor 的时间也接近 0ms，因此之前“timeout 越大 tok/s 越低”的均值表不能作为稳定结论。

## 当前实现语义

在 local split 模式下：

```text
major group: 计算 head/main rank slice
minor group: 计算 tail rank slice
major/minor: 同时开始计算
major 完成后: 最多等待 minor timeout_ms
minor 按时完成: 保留完整 rank 结果
minor 超时: 丢弃 minor tail 贡献
```

这和错误实现不同。错误实现的问题是：

```text
先按照 rate 截掉 tail rank
再把剩余 rank 分给 major/minor
```

这样会导致即使 no-timeout，tail 也已经被丢弃，所以 PPL 会异常变差。

修复后的关键代码在：

- `src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml_mul_mat_svd_use_local_tail_split(...)`
- `ggml_compute_forward_mul_mat_svd_vec(...)`
- `ggml_graph_plan(...)`

修复后的行为：

- local split active 时，`rate` 表示 minor/tail rank 比例。
- runtime 工作 rank 为 full rank。
- `k_group_a = k_keep`，group A 计算 head rank。
- group B 计算剩余 tail rank。
- `major_group_id = 0`，`minor_group_id = 1`，避免因为 rank 数量大小自动交换 major/minor 语义。
- 只有 timeout decision 为 drop 时，minor group 的 tmp slice 才会被清零。

## Runtime 诊断计数

为了避免继续靠 PPL 或 tok/s 反推 timeout 行为，我加了 runtime 计数输出：

```text
[svd-local-timeout-profile]
```

字段含义：

| 字段 | 含义 |
|---|---|
| `up_keep/gate_keep/down_keep` | 对应 SVD op 中 minor tail 在 timeout 前完成，被保留 |
| `up_drop/gate_drop/down_drop` | 对应 SVD op 中 minor tail 超时，被丢弃 |
| `up_wait/gate_wait/down_wait` | major leader 在 major 完成后额外等待 minor 的总时间 |

这行 profile 在 `ggml_svd_local_profile_print_and_reset()` 中打印，`decode_svd_test` 和 `perplexity_svd_test` 都能看到。

## 有效实验结果

### 6+2, 20% Load, rate=0.25

报告：

`results/rerun_isolated_20260428_195933/fixed_local_split_6p2_load20_rate025_20260428/REPORT.md`

设置：

- major group：`60-65`，6 核
- minor group：`66-67`，2 核
- load：20%
- rate：`alternate_0.25`
- 语义：奇数层 minor 计算 25% tail rank

| policy | total timeout | per active layer timeout | PPL | decode tok/s | speedup vs baseline |
|---|---:|---:|---:|---:|---:|
| `baseline_no_svd` | - | - | 15.1424 | 30.1652 | 1.000x |
| `svd_local_split_6p2_rate025` | 0 ms | 0 ms | 15.1424 | 26.6076 | 0.882x |
| `svd_local_split_6p2_rate025` | 20 ms | 2 ms | 15.2896 | 27.5909 | 0.915x |
| `svd_local_split_6p2_rate025` | 40 ms | 3 ms | 15.1529 | 26.4152 | 0.876x |
| `svd_local_split_6p2_rate025` | 60 ms | 5 ms | 15.0807 | 27.7179 | 0.919x |
| `svd_local_split_6p2_rate025` | 80 ms | 6 ms | 15.1424 | 27.5437 | 0.913x |

解释：

- 修复后 no-timeout split 的 PPL 和 no-SVD baseline 一样，说明 tail 没有默认丢弃。
- timeout 20/40ms 有轻微 PPL 波动，可能存在少量 tail drop 或短样本 PPL 噪声。
- 这组速度仍低于 baseline，说明 local split 本身的同步/分组开销还没抵消收益。

### 2+6, Minor 50% Load, rate=0.25

报告：

`results/rerun_isolated_20260428_195933/fixed_local_split_major2_minor6_minor_load50_rate025_decode_20260428/REPORT.md`

设置：

- major group：`60-61`，2 核，无背景负载
- minor group：`62-67`，6 核
- minor load：`stress-ng --cpu 6 --cpu-load 50`
- rate：`alternate_0.25`

原始 decode 均值表：

| policy | total timeout | tok/s mean | stdev | speedup vs baseline |
|---|---:|---:|---:|---:|
| `baseline_no_svd` | - | 11.9502 | 18.4270 | 1.000x |
| `svd_local_split_major2_minor6_rate025` | 0 ms | 27.3206 | 1.2995 | 2.286x |
| `svd_local_split_major2_minor6_rate025` | 20 ms | 7.7918 | 12.5011 | 0.652x |
| `svd_local_split_major2_minor6_rate025` | 40 ms | 17.8540 | 15.1314 | 1.494x |
| `svd_local_split_major2_minor6_rate025` | 60 ms | 8.9258 | 11.8344 | 0.747x |
| `svd_local_split_major2_minor6_rate025` | 80 ms | 1.5635 | 2.4265 | 0.131x |

这张表现在只能作为 raw phenomenon，不能解释为“timeout 越大越慢”。原因是 stdev 很大，raw runs 出现了大量极慢点。

无负载 PPL 参考：

| timeout | PPL |
|---:|---:|
| baseline | 15.1424 |
| 0 ms | 15.1424 |
| 20 ms | 15.1424 |
| 40 ms | 15.1424 |
| 60 ms | 15.1424 |
| 80 ms | 15.1424 |

这张 PPL 表只说明无负载时没有 tail 被丢弃；它不能说明 minor 50% load 下的质量损失。

### Timeout Profile 诊断

报告：

`results/rerun_isolated_20260428_195933/timeout_profile_diagnosis_major2_minor6_load50_20260428/DIAGNOSIS.md`

同样使用：

- major group：`60-61`
- minor group：`62-67`
- minor load：50%
- rate：`alternate_0.25`

诊断结果：

| timeout | repeat | tok/s | drop 情况 | major 等 minor |
|---:|---:|---:|---|---:|
| 20 ms | 0 | 21.9162 | drop=0 | 约 0.006 ms |
| 20 ms | 1 | 3.7342 | drop=0 | 约 0.652 ms |
| 20 ms | 2 | 26.3347 | drop=0 | 约 0.007 ms |
| 60 ms | 0 | 0.9737 | gate drop=1 | 约 25.859 ms |
| 60 ms | 1 | 9.0924 | drop=0 | 约 0.005 ms |
| 60 ms | 2 | 2.5900 | drop=0 | 约 0.003 ms |
| 80 ms | 0 | 25.9108 | drop=0 | 约 0.003 ms |
| 80 ms | 1 | 8.6383 | drop=0 | 约 0.003 ms |
| 80 ms | 2 | 26.7240 | drop=0 | 约 0.002 ms |

结论：

- 在 `major=2/minor=6/rate=0.25` 下，minor tail 大多数时候没有超时。
- 这是合理的：major 只有 2 核，却要算约 75% rank；minor 有 6 核，即使每核 50% 背景负载，也只算 25% tail。
- 很多低 tok/s 样本中 `drop=0`，`wait` 接近 0ms，所以慢点不是 timeout 等 minor 导致的，而是 CPU 调度 / cgroup / stress-ng 干扰下的计算抖动。
- 因此，原先的“timeout 增大导致 tok/s 下降”结论不成立。

## 需要作废或降级解释的结果

以下结果不能作为最终结论：

| 结果 | 问题 |
|---|---|
| PPL `287` / `2280` 的短样本表 | `ctx-size=32` 太短，PPL 数值无效 |
| `alternate_0.75` PPL `69` 的旧表 | 旧实现提前丢 tail rank，不是设计语义 |
| 早期 timeout policy speedup 表 | 基于旧 local split / truncation 语义，只能作为历史记录 |
| `major=2/minor=6` 均值表里 timeout 越大越慢 | 被高方差极慢点污染，runtime profile 不支持该因果解释 |
| no-load PPL 表解释 loaded decode 行为 | 条件不一致，只能说明无负载时 split 不改变 PPL |

## 后续实验建议

若目标是验证“timeout 丢弃 minor tail 能否换取速度，并衡量 PPL 损失”，应该使用更容易让 minor 迟到的设置：

| 设置 | 建议 |
|---|---|
| major/minor 核心数 | major 6 核，minor 2 核 |
| minor 工作量 | rate 0.5 或 0.75 |
| 背景负载 | 只打 minor 核，比如 50% / 80% |
| PPL | 必须在同一 loaded 条件下跑，不能用 no-load PPL 代替 |
| 报告字段 | 同时报告 PPL、tok/s、`keep/drop`、`wait` |
| repeats | 至少 5 次，报告 median 和 raw runs |

更推荐的下一组实验矩阵：

```text
major: 60-65
minor: 66-67
minor load: 50%, 80%
rate: alternate_0.5, alternate_0.75
timeout budget: 0, 20, 40, 60, 80 ms
PPL ctx: 128 或 256，chunks=1
decode repeats: >= 5
```

判断标准：

- 如果 `drop > 0` 且 tok/s 提升，同时 PPL 只小幅变差，说明 timeout 策略有效。
- 如果 `drop = 0`，则不应该把速度变化归因于 timeout。
- 如果 `wait` 很大但 `drop = 0`，说明 timeout 过宽，major 在等 minor，可能拖慢。
- 如果 tok/s 慢但 `drop = 0` 且 `wait ~= 0`，说明慢点来自系统调度噪声。

## 当前可信结论

截至当前修复和诊断：

1. local split 的计算语义已经修正，no-timeout 下不会默认丢 tail。
2. timeout 丢弃路径可以被触发，runtime 现在能直接计数。
3. `major=2/minor=6/rate=0.25` 不是验证 timeout 收益的合适拓扑，因为 minor 通常不晚。
4. 之前看起来矛盾的现象，主要来自把 no-load PPL 和 loaded decode 混在一起解释，以及高方差 decode 均值被极端慢点污染。
5. 下一步需要用 `major=6/minor=2` 或更高 minor tail rate，让 timeout 机制真正进入可观测区间，再做 PPL/速度 tradeoff 表。

