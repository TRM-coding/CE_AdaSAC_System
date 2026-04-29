# Exp12 Core Additivity, Profile Builder, and Scheduler Update

日期：2026-04-29

## 目的

Algorithmv2 5.2 中的 DP 需要真实的 `Tmain` / `Ttail` 表。如果把核心利用率也直接放进 profile 维度，组合会爆炸：

```text
layer × q × core split × utilization_vector
```

因此本轮先验证一个降维假设：

```text
T(S) ~= 1 / sum_i(1 / T_i)
```

其中 `T_i` 是核心 `i` 在某个背景利用率下单独完成同一 decode workload 的时间。

## 新增实现

### 1. 核心可加性验证

脚本：

```text
src/llama.cpp/exp12_algorithms/validate_core_additivity.py
```

功能：

- 支持核心利用率 `0%-100%`，默认步长 `10%`。
- 对每个核心测单核曲线 `T_i(u)`。
- 对若干多核组合测 `T_meas(S,u)`。
- 用 harmonic throughput sum 预测 `T_pred(S,u)`。
- 输出误差：

```text
relative_error = abs(T_meas - T_pred) / T_meas
```

主要输出：

```text
single_core.csv
multi_core.csv
additivity_error.csv
ADDITIVITY_REPORT.md
summary.json
```

本轮运行：

```bash
python3 src/llama.cpp/exp12_algorithms/validate_core_additivity.py \
  --cpus 60-67 \
  --loads 0,10,20,30,40,50,60,70,80,90,100 \
  --repeats 1 \
  --timeout-s 90 \
  --out-dir src/llama.cpp/exp12_algorithms/results/additivity_full_20260429_r1
```

结果：

```text
count  = 66
mean   = 0.1941
median = 0.1372
p90    = 0.4012
max    = 0.7127
```

结论：

```text
单纯可加性不能直接认为成立。
```

中位误差约 `13.7%`，说明这个模型可作为低维估计的基础；但 `p90` 达到 `40.1%`，尤其 6-8 核组合误差较大，说明必须加入 group-size/load 校准，关键 split 还需要直接测。

误差最大的点集中在 8 核组合：

| cpus | load | measured ms | predicted ms | relative error |
|---|---:|---:|---:|---:|
| 60-67 | 30% | 69.6333 | 20.0044 | 0.7127 |
| 60-67 | 20% | 40.4118 | 20.2801 | 0.4982 |
| 60-67 | 40% | 38.8865 | 20.0717 | 0.4838 |
| 60-67 | 100% | 37.0624 | 20.1547 | 0.4562 |
| 60-67 | 10% | 35.3874 | 19.8055 | 0.4403 |

这说明多核同步、内存带宽、调度抖动会显著破坏理想线性加速。

### 2. Additive + Calibration Profile Builder

脚本：

```text
src/llama.cpp/exp12_algorithms/build_additive_profile.py
```

输入：

- `single_core.csv`
- `additivity_error.csv`

输出：

- 每个 uniform utilization 场景一个 profile：

```text
profile_load_0.json
profile_load_10.json
...
profile_load_100.json
```

每个 profile 包含：

```text
core_splits[p]
  major_cpus
  minor_cpus
  layers[l]
    candidates[q]
      main_ms
      tail_ms
      loss
      weight
```

构造模型：

```text
T = work / sum(core_speed(util)) * calibration(n_cpus, avg_util)
```

其中：

- `core_speed(util)` 来自单核实测曲线。
- `calibration(n_cpus, avg_util)` 来自多核实测 / additive 预测的比值。
- 如果某个 `n_cpus` 未直接测到，使用最近的已测 group-size/load 校准因子，避免未测的 5/7 核组合被错误当成完美线性。

本轮生成：

```bash
python3 src/llama.cpp/exp12_algorithms/build_additive_profile.py \
  --single-core-csv src/llama.cpp/exp12_algorithms/results/additivity_full_20260429_r1/single_core.csv \
  --additivity-error-csv src/llama.cpp/exp12_algorithms/results/additivity_full_20260429_r1/additivity_error.csv \
  --cpus 60-67 \
  --uniform-loads 0,10,20,30,40,50,60,70,80,90,100 \
  --out-dir src/llama.cpp/exp12_algorithms/results/additive_profiles_20260429_r1_calibrated
```

注意：

这是一版 calibrated profile，不是最终物理真值 profile。由于可加性验证没有完全通过，后续要对关键 split 直接 profile。

### 3. Scheduler 多核心划分枚举

文件：

```text
src/llama.cpp/exp12_algorithms/scheduler.py
```

新增能力：

- 读取 `core_splits` profile。
- 外层枚举 `p`，即 major/minor 核心划分。
- 每个 `p` 内部执行原有 DP。
- 本地可行时，不进入 offload。
- 只有所有本地 split 都无解时，才搜索 suffix offload。

选择规则：

```text
1. 本地可行优先于 offload
2. 本地方案中先最小 loss
3. loss 相同，选择更少裁剪层
4. 再相同，偏向更大的 p
5. 最后选择更低 total_main_ms
```

因此，在所有核心空闲且完整本地执行可行时，`p=M`、`rate=0`、`offloaded_layers=[]` 一定会被选中。

如果本地全部无解，offload 搜索规则是：

```text
1. 先最少 offloaded layers
2. 再最小 loss
3. 再最小 total time
```

这对应“只有大量核心高占用、本地无法满足时限时，才卸载”。

## Scheduler Sanity

用 calibrated profile 跑 `load=0,10,...,100` 的求解结果摘要：

| load | mode | p | offloaded layers | clipped layers | max rate | loss | total ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | local | 8 | 0 | 0 | 0.0 | 0.0000 | 32.0358 |
| 10 | local | 7 | 0 | 0 | 0.0 | 0.0000 | 31.2980 |
| 20 | local | 7 | 0 | 0 | 0.0 | 0.0000 | 31.0668 |
| 30 | local | 7 | 0 | 0 | 0.0 | 0.0000 | 31.1907 |
| 40 | local | 7 | 0 | 0 | 0.0 | 0.0000 | 32.1840 |
| 50 | local | 8 | 0 | 0 | 0.0 | 0.0000 | 31.8804 |
| 60 | local | 7 | 0 | 0 | 0.0 | 0.0000 | 31.4846 |
| 70 | local | 8 | 0 | 0 | 0.0 | 0.0000 | 31.7830 |
| 80 | local | 7 | 0 | 0 | 0.0 | 0.0000 | 31.6699 |
| 90 | local | 8 | 0 | 0 | 0.0 | 0.0000 | 32.3058 |
| 100 | local | 7 | 0 | 14 | 0.3 | 0.8179 | 33.0091 |

这个 sanity 满足最重要的逻辑约束：

- `load=0` 时选择完整本地执行：`p=8`、不裁剪、不卸载。
- 低/中负载时仍然不卸载。
- 高负载时先尝试本地裁剪；只有本地裁剪无解时，才会进入 offload。

## 当前可信结论

1. 核心吞吐可加性不是严格成立。中位误差可接受，但高分位误差太大。
2. 因此不能用纯 harmonic model 作为最终 profile。
3. 已实现 calibrated additive profile，作为 DP 的低维输入模型。
4. 已完善 scheduler：它现在能枚举 core split，并保证空闲时优先选择 no-SVD/no-offload。
5. 要达到“物理真实最优”，下一步必须对关键 core split 做 direct profile，用直接测得的 `Tmain/Ttail` 覆盖 additive 估计。

## 后续建议

下一步应该补 direct split profile，优先测以下组合：

```text
p=8: major=60-67, minor=empty
p=7: major=top7, minor=last1
p=6: major=top6, minor=last2
p=4: major=top4, minor=last4
```

并按利用率 `0,10,...,100`、rate `0,0.1,...,0.9` 直接测 `Tmain/Ttail`。这样可以把当前 calibrated profile 中误差最大的高核数组合替换成真实测量值。

## 异构核心负载补充验证

上一节 `additivity_full_20260429_r1` 验证的是 uniform load：

```text
S = {60,61,62,63}, load=50%
=> 60/61/62/63 都是 50% 背景负载
```

Algorithmv2 实际面对的是 heterogeneous load vector，例如：

```text
60:0%, 61:20%, 62:60%, 63:100%
```

因此补充了异构负载验证脚本：

```text
src/llama.cpp/exp12_algorithms/validate_heterogeneous_additivity.py
```

输出目录：

```text
src/llama.cpp/exp12_algorithms/results/heterogeneous_additivity_20260429_r2
```

主要 CSV：

```text
heterogeneous_raw.csv
heterogeneous_additivity_error.csv
```

运行命令：

```bash
python3 src/llama.cpp/exp12_algorithms/validate_heterogeneous_additivity.py \
  --single-core-csv src/llama.cpp/exp12_algorithms/results/additivity_full_20260429_r1/single_core.csv \
  --repeats 2 \
  --timeout-s 90 \
  --out-dir src/llama.cpp/exp12_algorithms/results/heterogeneous_additivity_20260429_r2
```

所有 raw run 均成功：

```text
heterogeneous_raw.csv: 20 rows, status=ok
```

汇总：

```text
count  = 10
mean   = 0.4767
median = 0.3864
max    = 0.9967
```

结果表：

| name | cpus | load vector | measured ms | predicted ms | relative error |
|---|---|---|---:|---:|---:|
| `2core_0_50` | `60,61` | `60:0,61:50` | 87.8016 | 78.6581 | 0.1041 |
| `2core_20_80` | `60,61` | `60:20,61:80` | 85.7729 | 80.9497 | 0.0562 |
| `2core_0_100` | `60,61` | `60:0,61:100` | 86.2750 | 78.3588 | 0.0918 |
| `4core_0_0_50_50` | `60,61,62,63` | `60:0,61:0,62:50,63:50` | 48.9686 | 39.3762 | 0.1959 |
| `4core_0_20_60_100` | `60,61,62,63` | `60:0,61:20,62:60,63:100` | 1319.1255 | 39.8240 | 0.9698 |
| `4core_10_30_70_90` | `60,61,62,63` | `60:10,61:30,62:70,63:90` | 54.5482 | 40.3251 | 0.2607 |
| `6core_mixed` | `60-65` | `60:0,61:0,62:20,63:50,64:80,65:100` | 4832.7450 | 26.4376 | 0.9945 |
| `8core_gradient` | `60-67` | `60:0,61:10,62:20,63:30,64:60,65:70,66:90,67:100` | 40.8787 | 19.9499 | 0.5120 |
| `8core_major_idle_minor_busy` | `60-67` | `60:0,61:0,62:0,63:0,64:80,65:80,66:100,67:100` | 5973.5100 | 19.7901 | 0.9967 |
| `8core_alternating` | `60-67` | `60:0,61:100,62:10,63:90,64:20,65:80,66:30,67:70` | 48.0120 | 19.9318 | 0.5849 |

分组结论：

| n_cpus | count | median error | mean error | max error |
|---:|---:|---:|---:|---:|
| 2 | 3 | 0.0918 | 0.0840 | 0.1041 |
| 4 | 3 | 0.2607 | 0.4755 | 0.9698 |
| 6 | 1 | 0.9945 | 0.9945 | 0.9945 |
| 8 | 3 | 0.5849 | 0.6978 | 0.9967 |

异构负载下，纯可加模型比 uniform load 更不可靠。尤其当某些参与计算的核心接近 `100%` 背景负载时，多线程计算会被慢核心和 barrier 拖住，实际时间可能达到秒级，而 harmonic-sum 仍会给出几十毫秒的乐观预测。

因此，Algorithmv2 的真实 profile 不能只依赖：

```text
core × utilization 单核表
```

还必须加入：

```text
典型 heterogeneous load vector 的 direct split profile
```

尤其是：

```text
major 低负载 / minor 高负载
混合高低负载
包含 90%-100% 背景负载核心的组合
```
