# Exp12 Results 数据说明

本文档说明 `results/` 下各批实验数据的用途、文件含义和主要字段。当前目录的实验主题是负载感知的边端协同调度、SVD local split、per-layer timeout，以及用于 Algorithmv2 5.2 的 latency/profile 建模。

## 先读哪些结果

当前较可信、建议优先使用的数据：

- `additivity_full_20260429_r1/`: 8 个 CPU 在 uniform load 下的单核/多核可加性验证。
- `heterogeneous_additivity_20260429_r2/`: 非均匀 load vector 下的异构可加性验证。
- `additive_profiles_20260429_r1_calibrated/`: 基于单核实测和多核校准生成的 scheduler profile。
- `lhs_profile_dataset_20260429_r1/`: 后续拟合 `f(q_cpu60,...,q_cpu67,Q)` 的测量计划数据集。
- `latency_model_20260429_r1/`: 已测 latency 数据拟合出来的监督学习数据集、模型指标和预测结果。
- `rerun_isolated_20260428_195933/`: 修复 local split 语义后的重跑实验和诊断，其中 `DETAILED_DATA.md`、`RERUN_REPORT.md`、`TIMEOUT_VS_NO_SVD_BASELINE.md` 是汇总入口。

需要降级为历史记录或只作现象参考的数据：

- 早期 `timeout_policy_*`、`effectiveness_*` 中部分结果基于旧 local split 语义。旧语义会在 local split 前提前丢 tail rank，不能作为最终 timeout 策略结论。
- `effectiveness_pilot_20260428/` 和 `share_sweep_20260428/` 主要是早期 pilot，没有完整 summary 表。
- 某些 PPL 小样本实验使用很短 ctx，只适合 sanity check，不适合作为质量结论。

## 通用文件类型

`results/` 中大量文件是同一模式的重复产物，可按扩展名和命名理解：

| 文件模式 | 含义 |
|---|---|
| `*.md` | 人类可读实验报告或摘要，通常是最好的入口。 |
| `raw.json` | 每次运行的原始记录，包含命令、return code、吞吐、生成文本、日志尾部等。 |
| `summary.csv` / `summary.json` | 对 raw runs 按 policy/load/budget 聚合后的均值、方差、speedup、匹配率等。CSV 适合画图，JSON 适合程序读取。 |
| `*.rates.txt` | 每层 SVD rate 列表，逗号分隔，可直接作为 `decode_svd_test` 的 rate 文件。 |
| `timeout_budget_*.txt` / `timeouts.txt` | 每层 timeout 表，逗号分隔；某层为 `0` 表示该层不启用 per-layer timeout。 |
| `*.log` | 单次 benchmark、decode、PPL 或 stress-ng 输出日志。文件名通常编码了 load、budget、core split、repeat。 |
| `stress.log` / `stress_repeat.log` | 背景负载生成器日志，多数来自 `stress-ng`。 |
| `profile.json` | scheduler 输入 profile，描述每层候选 rate、时延、deadline、传输/尾端估计等。 |
| `schedule.json` | scheduler 输出结果，包含模式、可行性、rates、offload split、估计总时延等。 |
| `experiment.json` | 单次集成实验配置和输出路径索引。 |

## 主要字段词典

| 字段 | 含义 |
|---|---|
| `cpu_load` / `load_pct` / `cpu_load_percent` | 背景 CPU 负载百分比，通常来自 `stress-ng --cpu-load`。 |
| `load_workers` | 负载生成进程/线程数量。 |
| `cpus` | 本次测量使用的 CPU 集合，例如 `60,61,62,63`。 |
| `cpu` | 单核测量时的 CPU id。 |
| `load_vector` | 每个 CPU 对应的负载向量，例如 `60:0,61:50`。 |
| `Q_pct` | 模型接口中的 workload/quality 维度。本轮 latency model 数据里该值存在但固定。 |
| `q_cpu60` 等 | LHS 测量计划中 CPU 60 对应的 utilization/queue/load 维度取值。 |
| `policy` | 实验策略名，例如 `baseline`、`alternate_0.75`、`timeout_budget_8`、`baseline_no_svd`。 |
| `repeat` / `runs` | 重复编号或聚合后的运行次数。 |
| `status` / `succeed` / `returncode` | 运行状态。`returncode=0` 通常表示命令成功。 |
| `decode_tok_s` | decode 吞吐，单位 token/s。越高越快。 |
| `generation_decode_ms` | 生成阶段 decode 用时，单位 ms。越低越快。 |
| `prefill_decode_ms` | prefill + decode 的总耗时或解析出的相关耗时，视 benchmark 输出而定。 |
| `elapsed_s` | 外层脚本观测到的 wall-clock 耗时，单位秒。 |
| `speedup` / `speedup_vs_*` | 相对 baseline 的速度比。`1.0` 表示无提升，大于 `1.0` 表示更快。 |
| `generated_text` | 单 token/短文本 decode 输出。主要用于轻量一致性检查。 |
| `top1_id` / `top1_piece` / `top1_logit` | 下一 token top1 的 id、文本片段和 logit。 |
| `generated_text_match_rate` / `top1_match_rate` | 与参考策略输出一致的比例。不是完整质量指标。 |
| `ppl` / `ppl_stderr` | perplexity 及其标准误。越低通常越好。 |
| `delta_ppl_vs_*` / `ppl_ratio_vs_*` | 相对参考 PPL 的差值或比例。 |
| `eval_tok_s` / `ppl_eval_tok_s` | PPL 评估吞吐。 |
| `timeout_budget_ms` | 全局 timeout budget，调度器会分配到 active truncated layers。 |
| `active_layer_timeout_ms` | 被实际裁剪/启用 timeout 的层所分到的 per-layer timeout。 |
| `main_ms` / `tail_ms` | profile 中 major/head rank 和 minor/tail rank 的估计耗时。 |
| `loss` / `weight` | scheduler DP 使用的裁剪损失和 timeout budget 分配权重。 |
| `measured_ms` / `predicted_ms` | 实测时延和模型预测时延。 |
| `relative_error` | 相对误差，通常为 `abs(measured - predicted) / measured`。 |
| `ape` | absolute percentage error，预测绝对百分比误差。 |
| `log` | 对应单次运行日志文件路径。 |

## 目录说明

### `additivity_full_20260429_r1/`

完整核心可加性验证结果。脚本入口是 `validate_core_additivity.py`，实验覆盖 CPU `60-67`、uniform loads `0,10,...,100`、多种 core combo。

| 文件 | 含义 |
|---|---|
| `ADDITIVITY_REPORT.md` | 可读报告，给出误差统计和结论。当前结论是 `direct_split_profile_required`，即关键 split 仍需直接 profile。 |
| `single_core.csv` | 单核曲线。每行是某个 CPU 在某个 uniform load 下跑同一 decode workload 的结果。 |
| `multi_core.csv` | 多核组合实测结果。 |
| `additivity_error.csv` | 用 harmonic throughput sum 预测多核耗时后的误差表。 |
| `summary.json` | 可加性实验的配置、统计摘要和 verdict。 |
| `single_cpu{cpu}_load{load}_rep{n}.log` | 单核单次运行日志。 |
| `multi_{cpu...}_load{load}_rep{n}.log` | 多核组合单次运行日志。 |
| `stress.log` | 背景负载日志。 |

主要 CSV 字段：

| 文件 | 字段说明 |
|---|---|
| `single_core.csv` | `kind` 固定为 single；`cpu` 是 CPU id；`load_pct` 是背景负载；`repeat` 是重复编号；`status` 是运行状态；`generation_decode_ms` 是 decode 耗时；`elapsed_s` 是外层耗时；`log` 指向日志。 |
| `multi_core.csv` | `kind` 固定为 multi；`cpus` 是 CPU 组合；`n_cpus` 是核心数；其余字段同上。 |
| `additivity_error.csv` | `cpus/n_cpus/load_pct` 描述组合；`measured_ms` 为多核实测；`predicted_ms` 为单核吞吐调和相加预测；`relative_error` 为相对误差；`missing_single_cpus` 表示缺失的单核基准。 |

### `additivity_sanity_20260429/`

小规模 sanity 版可加性验证，覆盖 CPU `60,61` 和 loads `0,50,100`。文件结构与 `additivity_full_20260429_r1/` 相同，用于快速检查脚本和环境，不建议作为最终模型依据。

### `heterogeneous_additivity_20260429_r2/`

异构 load vector 可加性验证。这里每个 CPU 可以有不同负载，例如 `60:0,61:100`，用来检验 uniform load 校准是否能推广到异构场景。

| 文件 | 含义 |
|---|---|
| `HETEROGENEOUS_ADDITIVITY_REPORT.md` | 可读报告，列出每个异构场景的实测、预测和误差。 |
| `heterogeneous_raw.csv` | 每个异构场景每次 repeat 的原始运行结果。 |
| `heterogeneous_additivity_error.csv` | 按场景聚合后的异构可加性误差。 |
| `summary.json` | 误差摘要，例如 count、mean、median、max。 |
| `{scenario}_rep{n}.log` | 单个异构场景的日志。 |
| `stress.log` | 背景负载日志。 |

`heterogeneous_raw.csv` 字段：`name` 是场景名；`cpus` 是 CPU 集合；`load_vector` 是每核负载；`repeat/status/generation_decode_ms/elapsed_s/log` 同上。

`heterogeneous_additivity_error.csv` 字段：`measured_ms_median` 是多次 repeat 的中位实测耗时；`predicted_ms` 是单核模型预测；`relative_error` 是误差；`repeats` 是参与聚合的次数。

### `sampling_speed_probe_20260429/`

异构可加性/采样速度的探针实验。文件结构接近 `heterogeneous_additivity_20260429_r2/`，但场景较少，主要用于确认某些坏点和微基准速度现象。

| 文件 | 含义 |
|---|---|
| `CORRECTED_MICROBENCH_SPEED.md` | 对修正后 microbenchmark 速度现象的说明。 |
| `HETEROGENEOUS_ADDITIVITY_REPORT.md` | 探针场景的异构可加性报告。 |
| `heterogeneous_raw.csv` / `heterogeneous_additivity_error.csv` | 原始运行和误差表，字段同异构可加性实验。 |
| `probe_*_rep0.log` | 各探针场景单次日志，例如 easy、balanced、bad、gradient。 |
| `summary.json` / `stress.log` | 摘要和负载日志。 |

### `additive_profiles_20260429_r1/`

第一版 additive scheduler profiles。由 `build_additive_profile.py` 根据单核曲线和可加性误差构造，用于 `scheduler.py`。这是模型化 profile，不是完整 direct profile。

| 文件 | 含义 |
|---|---|
| `PROFILE_REPORT.md` | profile 生成报告，列出每个 load 场景的 full local ms 和 deadline。 |
| `index.json` | profile 索引，通常列出可用场景、CPU、rate、路径。 |
| `profile_load_{0..100}.json` | 每个 uniform load 场景的 scheduler profile。 |
| `scheduler_sanity.jsonl` | 用这些 profile 跑 scheduler sanity 的逐行 JSON 输出。 |

`profile_load_*.json` 主要结构：

| 字段 | 含义 |
|---|---|
| `model` / `source` | profile 类型和来源。 |
| `cpus` | 可用 CPU 集合。 |
| `utilization` | 当前 uniform load 场景。 |
| `n_layers` | 层数。 |
| `deadline_base_ms` / `local_deadline_ms` / `request_deadline_ms` / `timeout_budget_ms` | scheduler 约束参数。 |
| `full_local_ms` | 不裁剪、不卸载的本地完整执行估计耗时。 |
| `sorted_cpus_by_effective_speed` | 按有效速度排序后的 CPU。 |
| `core_splits` | major/minor 核心划分候选。 |
| `tx_ms_by_split_m` / `end_ms_by_split_m` | 后缀 offload 时传输和远端尾部耗时估计。 |

`core_splits[*]` 中：`p` 是 major 核数；`major_cpus` / `minor_cpus` 是两组 CPU；`layers[*].candidates[*]` 给出某层某个 rate 下的 `main_ms`、`tail_ms`、`loss`、`weight`。

### `additive_profiles_20260429_r1_calibrated/`

校准版 additive profiles。它和 `additive_profiles_20260429_r1/` 结构相同，但构造时引入了 group-size/load calibration factor，更适合作为当前 scheduler 输入。由于可加性误差仍偏大，报告中也明确提示关键 split 后续需要 direct profile。

### `lhs_profile_dataset_20260429_r1/`

用于后续建立高维 latency/profile 函数的数据采样计划。注意：这个目录目前是“要测哪些配置”的 dataset，不是已测 latency。

| 文件 | 含义 |
|---|---|
| `LHS_PROFILE_DATASET_REPORT.md` | 数据集报告，给出样本数、source 分布、Q 分布。 |
| `metadata.json` | 生成参数，包括 CPU、load/q 取值、LHS 样本数、seed、总样本数。 |
| `profile_lhs_dataset.csv` | 6375 条待测配置，混合结构化边界点和 Latin hypercube samples。 |

`profile_lhs_dataset.csv` 字段：`sample_id` 是样本 id；`source` 表示来源，例如 `lhs`、`all_equal`、`single_core_*`、`major_minor_grid`；`Q_pct` 是全局 workload/quality 维度；`q_cpu60` 到 `q_cpu67` 是各 CPU 维度取值；`load_vector` 是可读的每核负载向量。

### `latency_model_20260429_r1/`

把已有测量结果整理成监督学习数据，并拟合 sklearn tabular regressor 的结果目录。

| 文件 | 含义 |
|---|---|
| `LATENCY_MODEL_REPORT.md` | 模型报告，列出数据量、候选模型指标、最佳模型和最差预测样本。 |
| `supervised_dataset.csv` | 标准化后的监督学习数据集，含 train/test split。 |
| `model_metrics.csv` | 每个候选模型的 train/test 指标。 |
| `test_predictions.csv` | 最佳模型在 held-out test set 上的预测明细。 |
| `best_model.pkl` | pickle 保存的 sklearn 模型和 feature metadata。 |
| `summary.json` | 模型训练摘要。 |

`supervised_dataset.csv` 字段：`row_id` 是样本 id；`split` 是 train/test；`source` 是数据来源，例如 single、uniform_multi、heterogeneous；`name` 是原实验名；`cpus/load_vector/Q_pct` 是输入特征；`latency_ms` 是目标时延。

`model_metrics.csv` 字段：`train_*` 和 `test_*` 分别是训练集/测试集上的 MAE、median AE、MAPE、median APE、p90 APE、max APE 和 R2。

`test_predictions.csv` 字段：`actual_ms` 是真实值；`predicted_ms` 是模型预测；`abs_error_ms` 是绝对误差；`ape` 是绝对百分比误差。

### `cgroup_20260428/`

早期 cgroup 集成烟测/调度样例。用于确认 profile、scheduler、rates 文件和实验 JSON 能串起来。

| 文件 | 含义 |
|---|---|
| `profile.json` | scheduler 输入 profile。 |
| `schedule.json` | scheduler 输出，包含 `mode`、`feasible`、`rates`、`split_m`、`offloaded_layers`、`total_ms`。 |
| `rates.txt` | scheduler 生成的每层 SVD rates。 |
| `experiment.json` | 本次实验的路径、CPU、profile、schedule、decode 结果索引。 |

### `timeout_integration_smoke/`

per-layer timeout 接口的集成烟测。文件结构类似 `cgroup_20260428/`，额外包含：

| 文件 | 含义 |
|---|---|
| `timeouts.txt` | scheduler 生成的每层 timeout 文件，可作为 `decode_svd_test` 的第 12 个参数。 |

### `effectiveness_20260428/`

早期 SVD rate 调度有效性测试。对比 baseline 和若干 rate policy，在固定负载设置下看吞吐提升。因时间较早，结论应结合后续修正版报告解释。

| 文件 | 含义 |
|---|---|
| `SUMMARY.md` | 实验摘要。 |
| `raw.json` | 每次 decode 原始记录。 |
| `summary.csv` / `summary.json` | 按 `load_workers,policy` 聚合的吞吐和 speedup。 |
| `alternate_0.5.rates.txt` / `alternate_0.75.rates.txt` / `uniform_0.5.rates.txt` | 测试用 rate policy。 |

`summary.csv` 字段：`decode_tok_s_mean/stdev` 是吞吐均值和标准差；`generation_decode_ms_mean` 是 decode 耗时均值；`speedup_vs_same_load_baseline` 是相同负载下相对 baseline 的 speedup。

### `effectiveness_load_sweep_20260428/`

按 CPU load sweep 的有效性测试，包含 `load0/`、`load20/`、`load50/`、`load80/`、`load100/` 五个子目录。

| 文件 | 含义 |
|---|---|
| `LOAD_SWEEP_SUMMARY.md` | 跨 load 汇总报告。 |
| `load_sweep_summary.csv` | 跨 load 汇总表。 |
| `load*/SUMMARY.md` | 单个 load 的摘要。 |
| `load*/raw.json` | 单个 load 下每次运行原始记录。 |
| `load*/summary.csv` / `load*/summary.json` | 单个 load 下按 policy 聚合后的表。 |
| `load*/alternate_0.75.rates.txt` | 该批实验使用的 rate 文件。 |

`load_sweep_summary.csv` 字段：`cpu_load_percent` 是负载；`baseline_tok_s` / `scheduled_tok_s` 是两种策略吞吐；`baseline_ms` / `scheduled_ms` 是 decode 耗时；`speedup` 是 scheduled 相对 baseline 的速度比；`*_stdev` 是标准差。

### `effectiveness_stress20_20260428/` 和 `effectiveness_stress20_repeat_20260428/`

20% stress-ng 负载下的有效性测试，repeat 版增加重复次数。文件结构与 `effectiveness_20260428/` 相同：`SUMMARY.md`、`raw.json`、`summary.csv/json`、rate 文件。用于观察固定中低负载下 rate 调度是否带来吞吐变化。

### `effectiveness_pilot_20260428/`

早期 pilot，只保存 rate 文件：

| 文件 | 含义 |
|---|---|
| `alternate_0.5.rates.txt` / `alternate_0.75.rates.txt` / `uniform_0.5.rates.txt` | 不同层级 SVD rate 策略。 |

没有 raw/summary，因此只能作为 rate 配置来源。

### `share_sweep_20260428/`

早期 rank share sweep。主要用于探索 `alternate_0.75` 策略和 raw/summary 记录格式。

| 文件 | 含义 |
|---|---|
| `alternate_0.75.rates.txt` | 本批使用的 rate file。 |
| `raw.json` | 原始运行记录。 |
| `summary.json` | 聚合摘要。 |

### `timeout_policy_20260428/`

早期 per-layer timeout 策略测试。保留作历史记录，因旧 local split 语义问题，不能直接作为最终 timeout 策略结论。

| 文件 | 含义 |
|---|---|
| `SUMMARY.md` | 实验摘要。 |
| `raw.json` | 每次 timeout budget 运行原始记录。 |
| `summary.csv` / `summary.json` | 按 `cpu_load,policy` 聚合后的吞吐、speedup、文本匹配率。 |
| `alternate_0.75.rates.txt` | 使用的 SVD rate file。 |
| `timeout_budget_0.txt` / `timeout_budget_2.txt` / `timeout_budget_8.txt` / `timeout_budget_16.txt` | 各 budget 的 per-layer timeout 文件。 |

`summary.csv` 字段：`speedup_vs_no_timeout` 是相同 load 下相对 `timeout_budget_0` 的速度比；`generated_text_match_rate/top1_match_rate` 是轻量输出一致性指标。

### `timeout_policy_load80_20260428/`

只针对 80% CPU load 的 timeout policy 测试。文件结构与 `timeout_policy_20260428/` 相同，但 budget 集合是 `0,2,8`。同样属于早期语义数据，作历史/现象参考。

### `timeout_policy_combined_20260428/`

把多个 timeout policy 结果合并后的汇总目录。

| 文件 | 含义 |
|---|---|
| `SUMMARY.md` | 合并摘要。 |
| `summary.csv` / `summary.json` | 合并后的 `cpu_load,policy` 指标。 |

`summary.csv` 中 `tok_mean/tok_median` 是吞吐均值/中位数，`ms_mean` 是耗时均值，`speedup_mean/speedup_median` 是相对 no-timeout 的速度比，`text_match/top1_match` 是输出一致性。

## `rerun_isolated_20260428_195933/`

这是修复 local split 语义后的隔离重跑大目录，建议作为 2026-04-28 这批 timeout/local split 结果的主要入口。

### 顶层文件

| 文件 | 含义 |
|---|---|
| `RERUN_REPORT.md` | 重跑总报告。 |
| `DETAILED_DATA.md` | 详细重跑数据表，包含 scheduling summary/raw runs 和 timeout summary/raw runs。 |
| `TIMEOUT_VS_NO_SVD_BASELINE.md` | scheduled SVD + timeout 与 no-SVD baseline 的对比报告。 |
| `effectiveness_combined.csv` / `effectiveness_combined.json` | 各 load 的有效性结果合并表。 |
| `effectiveness_load_{0,20,50,80,100}.csv` | 单 load 有效性 summary 的顶层拷贝/汇总。 |
| `timeout_vs_no_svd_baseline.csv` | no-SVD baseline 与 SVD timeout 的直接对比。 |

`timeout_vs_no_svd_baseline.csv` 字段：`cpu_load` 是负载；`timeout_budget_ms` 是 budget；`baseline_no_svd_tok_s` 是 no-SVD baseline 吞吐；`scheduled_svd_timeout_tok_s` 是调度策略吞吐；`throughput_delta_tok_s` 是吞吐差；`speedup_vs_no_svd_baseline` 是相对 no-SVD 的速度比；`text_match_vs_no_timeout_svd/top1_match_vs_no_timeout_svd` 是相对 no-timeout SVD 的一致性。

### `effectiveness_load_{0,20,50,80,100}/`

修复语义后的有效性 load sweep 子目录。每个子目录包含：

| 文件 | 含义 |
|---|---|
| `SUMMARY.md` | 单 load 摘要。 |
| `raw.json` | 单次运行原始记录。 |
| `summary.csv` / `summary.json` | 按 policy 聚合后的吞吐和 speedup。 |
| `alternate_0.75.rates.txt` | 使用的 rate policy。 |

字段与前面的 effectiveness summary 相同。

### `timeout_policy/`

修复语义后的 timeout policy 重跑目录。

| 文件 | 含义 |
|---|---|
| `SUMMARY.md` | timeout policy 摘要。 |
| `raw.json` | 每次运行原始记录。 |
| `summary.csv` / `summary.json` | 按 `cpu_load,policy` 聚合。 |
| `alternate_0.75.rates.txt` | rate policy。 |
| `timeout_budget_{0,2,4,8,16}.txt` | per-layer timeout 文件。 |

字段与早期 `timeout_policy_20260428/` 相同，但这批数据更接近当前 runtime 语义。

### `fixed_local_split_6p2_load20_20260428/`

修复前后过渡期的一组 6+2 local split、20% load 实验。`6p2` 表示 major/minor 核心划分为 6 核 + 2 核。主要文件：

| 文件 | 含义 |
|---|---|
| `REPORT.md` | 实验报告。 |
| `summary.csv` | policy、timeout budget、PPL、decode tok/s、speedup、生成文本等汇总。 |
| `alternate_0.25.rates.txt` | 奇数层使用 0.25 tail rank 的 rate file。 |
| `timeout_budget_*.txt` | per-layer timeout files。 |
| `baseline_no_svd.*.log` / `budget_*.decode.log` / `budget_*.ppl.log` | baseline、decode、PPL 日志。 |

### `fixed_local_split_6p2_load20_rate025_20260428/`

6+2、20% load、`rate=0.25` 的修正版实验。`summary.csv` 字段包含：`policy`、`timeout_budget_ms`、`active_layer_timeout_ms`、`ppl`、`ppl_stderr`、`delta_ppl_vs_baseline`、`ppl_eval_tok_s`、`decode_tok_s`、`decode_speedup_vs_baseline`、`generation_decode_ms`。报告结论是修复后 no-timeout split 的 PPL 回到 no-SVD baseline，但这组速度仍低于 baseline。

### `fixed_local_split_6p2_minor_load50_rate025_20260428/`

6+2 local split，minor 侧 50% load，`rate=0.25` 的 decode 重复实验。

| 文件 | 含义 |
|---|---|
| `REPORT.md` | 报告。 |
| `raw.json` | 每次 decode 原始记录。 |
| `summary.csv` | 按 timeout budget 聚合后的吞吐、标准差、speedup、输出集合。 |
| `baseline_no_svd_rep*.decode.log` / `budget_*_rep*.decode.log` | 每次 decode 日志。 |
| `timeout_budget_*.txt` / `alternate_0.25.rates.txt` | timeout/rate 配置。 |

`summary.csv` 中 `generated_text_values` 和 `top1_values` 是各 repeat 的输出集合，用于粗略看输出是否变化。

### `fixed_local_split_major2_minor6_minor_load50_rate025_20260428/`

major=2、minor=6、minor 50% load、`rate=0.25` 的早期配置目录，主要保存配置文件和 timeout 文件；完整 decode/PPL 报告在下一个 `_decode_` 目录中。

### `fixed_local_split_major2_minor6_minor_load50_rate025_ctx64_20260428/`

major=2、minor=6、minor 50% load、ctx=64 的配置/timeout 文件目录。主要用于保留当时的 rate 和 timeout files。

### `fixed_local_split_major2_minor6_minor_load50_rate025_decode_20260428/`

major=2、minor=6、minor 50% load、`rate=0.25` 的 decode + no-load PPL 诊断目录。

| 文件 | 含义 |
|---|---|
| `REPORT.md` | 报告，强调均值表不能解释为“timeout 越大越慢”。 |
| `raw.json` | decode 原始记录。 |
| `summary.csv` | decode 按 budget 聚合。 |
| `ppl_noload_summary.csv` | 无背景负载下的 PPL 参考表。 |
| `baseline_no_svd_rep*.decode.log` / `budget_*_rep*.decode.log` | decode 日志。 |
| `baseline_no_svd.ppl_noload.log` / `budget_*.ppl_noload.log` | 无负载 PPL 日志。 |

`ppl_noload_summary.csv` 字段：`timeout_budget_ms`、`ppl_noload`、`ppl_stderr`、`ppl_eval_tok_s`、`elapsed_s`、`returncode`。注意它只说明无负载时是否丢 tail，不能替代 loaded decode 质量结论。

### `timeout_profile_diagnosis_major2_minor6_load50_20260428/`

runtime timeout profile 诊断目录，专门用于解释 major=2/minor=6/load50 下慢点是否来自 timeout。

| 文件 | 含义 |
|---|---|
| `DIAGNOSIS.md` | 诊断报告，解析 `[svd-local-timeout-profile]` 中 keep/drop/wait 计数。 |
| `budget_*.log` / `budget_*_extra_rep*.log` | decode 日志，包含 timeout profile 行。 |
| `stress.log` / `stress_repeat.log` | 背景负载日志。 |

profile 行字段：`up_keep/gate_keep/down_keep` 是 minor tail 按时完成并保留的次数；`up_drop/gate_drop/down_drop` 是 timeout 后丢弃 tail 的次数；`up_wait/gate_wait/down_wait` 是 major 等 minor 的总时间。

### `ppl_timeout_small_20260428/`

较小 PPL sanity 测试，ctx 约 128。用于检查 timeout/SVD split 对 PPL 是否出现明显异常，不适合作最终质量评估。

| 文件 | 含义 |
|---|---|
| `PPL_SANITY_CTX128_REPORT.md` | PPL sanity 报告。 |
| `sanity_summary.csv` | load、policy、timeout budget、PPL、PPL stderr、评估吞吐。 |
| `load*_baseline_no_svd.log` / `load*_scheduled_timeout_budget*.log` | PPL 运行日志。 |

### `ppl_timeout_tiny_20260428/`

更短上下文的 PPL timeout 测试，主要用于快速 smoke，不适合严肃质量结论。

| 文件 | 含义 |
|---|---|
| `PPL_TIMEOUT_TINY_REPORT.md` | 报告。 |
| `summary.csv` | load、policy、timeout budget、no-SVD PPL、当前 PPL、ratio、评估吞吐、是否 timeout。 |
| `load80_rows_2_16.json` | 80% load 下 budget 2 和 16 的部分行记录。 |
| `baseline_no_svd_load0.log` / `load*_budget*.log` | PPL 日志。 |

### `ppl_timeout_20load_large_budget_20260428/`

20% load 下更大 timeout budget 的 PPL 实验。

| 文件 | 含义 |
|---|---|
| `REPORT.md` | 报告。 |
| `summary.csv` | `cpu_load`、`timeout_budget_ms`、`active_layer_timeout_ms`、`ppl`、`delta/ratio`、`eval_tok_s`、耗时和 return code。 |
| `load20_timeout_budget_*.log` | PPL 日志。 |
| `timeout_budget_*.txt` | timeout 配置。 |

### `rate_ppl_diagnosis_20260428/`

不同 SVD rate 对 PPL 的诊断实验，用于分离 rate 本身对质量的影响。

| 文件 | 含义 |
|---|---|
| `REPORT.md` | 诊断报告。 |
| `summary.csv` | `rate_on_odd_layers`、`kept_rank_fraction_on_odd_layers`、PPL、评估吞吐、return code。 |
| `alternate_*.rates.txt` | 不同 rate policy。 |
| `alternate_*.log` / `no_svd.log` | PPL 日志。 |

## 如何使用这些数据

1. 做 scheduler/profile 相关分析，优先用 `additivity_full_20260429_r1/`、`heterogeneous_additivity_20260429_r2/`、`additive_profiles_20260429_r1_calibrated/` 和 `latency_model_20260429_r1/`。
2. 做 timeout/local split 语义分析，优先用 `CORRECTED_TIMEOUT_AND_SPLIT_REPORT.md` 和 `rerun_isolated_20260428_195933/` 下的修正版数据。
3. 做吞吐图，优先读取 `summary.csv`；需要查异常点时再回到 `raw.json` 和对应 `*.log`。
4. 做质量图，优先读取包含 `ppl` 的 `summary.csv`，不要用 `generated_text_match_rate` 代替 PPL。
5. 看到很高方差或极慢点时，需要检查日志里的 runtime profile、stress 日志和 return code，不能只根据均值归因。

