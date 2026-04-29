# Exp12: 负载感知边端协同调度算法

## 当前可信报告

修复 local split 语义后，当前应优先阅读：

- `CORRECTED_TIMEOUT_AND_SPLIT_REPORT.md`
- `ADDITIVE_PROFILE_AND_SCHEDULER_REPORT.md`
- `results/rerun_isolated_20260428_195933/timeout_profile_diagnosis_major2_minor6_load50_20260428/DIAGNOSIS.md`

早期 `TIMEOUT_POLICY_REPORT.md` 和部分 `results/*timeout*` 表格保留作历史记录，但其中一些结果基于旧语义：`rate` 在 local split 前会提前截掉 tail rank。因此旧表不能作为最终 timeout 策略结论。

当前 runtime 语义是：major/minor 同时计算，major 计算 head/main rank，minor 计算 tail rank；只有 timeout 触发时，minor tail 才会被丢弃。

Algorithmv2 5.2 的真实 profile 输入仍在完善中。当前已经新增核心可加性验证脚本和 calibrated additive profile builder；验证结果显示单纯可加性不够可靠，需要对关键 core split 做 direct profile。

本目录实现 `algorithm.pdf` 第五章的两个调度算法：

- `scheduler.py`: 本地动态规划调度。输入各层候选 SVD 截断率、主路径耗时、损失和补偿权重，满足“相邻两层不能同时裁剪”的约束，并按裁剪权重分配超时预算。
- `scheduler.py`: 联合调度。先尝试边侧本地 DP；本地无解时，从后向前搜索最小后层卸载集合，选择满足总时延约束的最大分割点 `m`。
- `run_exp12_local.py`: 本机小实验驱动。不使用 adb；复用现有 `decode_svd_test`，默认从 `60-79` 中取 `60-67` 作为电脑端、`68-75` 作为手机端模拟核心。

## 运行算法

```bash
python3 src/llama.cpp/exp12_algorithms/scheduler.py \
  --profile src/llama.cpp/exp12_algorithms/results/latest/profile.json \
  --local-deadline-ms 35 \
  --request-deadline-ms 56 \
  --timeout-budget-ms 8 \
  --out-json src/llama.cpp/exp12_algorithms/results/latest/schedule.json \
  --out-rates src/llama.cpp/exp12_algorithms/results/latest/rates.txt
```

`rates.txt` 是逗号分隔的每层 SVD rate，可直接作为现有程序第 6 个参数：

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off src/llama.cpp/exp12_algorithms/results/latest/rates.txt \
  60-63 64-67 0.50 0
```

`scheduler.py` 也可以输出论文中的超时预算分配结果：

```bash
python3 src/llama.cpp/exp12_algorithms/scheduler.py \
  --profile src/llama.cpp/exp12_algorithms/results/cgroup_20260428/profile.json \
  --local-deadline-ms 46.383624 \
  --request-deadline-ms 75.15865 \
  --timeout-budget-ms 8 \
  --out-rates /tmp/exp12_rates.txt \
  --out-timeouts /tmp/exp12_timeouts.txt
```

`timeouts.txt` 是逗号分隔的每层 `timeout_ms`。接入 `decode_svd_test` 时作为第 12 个参数，位于全局 `svd_offload_timeout_ms` 之后：

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off /tmp/exp12_rates.txt \
  60-63 64-67 0.25 0 2 /tmp/exp12_timeouts.txt
```

参数含义：

- `0.25`: A 组 rank 份额。
- `0`: 旧的全局 `minor_timeout_ms`，这里置 0，表示不使用全局值。
- `2`: 远端 offload 等待超时，保留旧参数位。
- `/tmp/exp12_timeouts.txt`: 新增的每层本地 minor timeout 表；某层为 `0` 表示该层不启用超时丢弃。

## 本机实验

```bash
python3 src/llama.cpp/exp12_algorithms/run_exp12_local.py
```

可选使用 cgroup v2 cpuset：

```bash
python3 src/llama.cpp/exp12_algorithms/run_exp12_local.py --use-cgroup
```

如果当前 sudo 不能免密，脚本会退回 `taskset`。本实验不启动 adb，也不连接手机；手机端时延通过 profile 中的 `tx_ms_by_split_m` 和 `end_ms_by_split_m` 建模，便于先验证第五章调度策略和现有 SVD rate 文件接口。

## 有负载有效性测试

验证本地 SVD rate 调度在 CPU 加压时是否真的提速：

```bash
python3 src/llama.cpp/exp12_algorithms/benchmark_effectiveness.py \
  --use-cgroup \
  --load-mode stress-ng \
  --cpu-load 20 \
  --load-workers-list 4 \
  --repeats 3 \
  --policies baseline,alternate_0.75 \
  --out-dir src/llama.cpp/exp12_algorithms/results/effectiveness_stress20_repeat_20260428
```

输出：

- `raw.json`: 每次运行的原始 `decode_svd_test` 结果。
- `summary.csv` / `summary.json`: 同负载 baseline 对比的平均吞吐和 speedup。
- `SUMMARY.md`: 可直接阅读的实验摘要。

## 每层 Timeout 策略测试

验证超时预算分配是否能在 minor rank slice 被丢弃时带来速度收益，以及输出是否偏离 no-timeout 参考：

```bash
python3 src/llama.cpp/exp12_algorithms/benchmark_timeout_policy.py \
  --use-cgroup \
  --cpu-loads 20,50,80 \
  --timeout-budgets-ms 0,2,4,8,16 \
  --repeats 2 \
  --out-dir src/llama.cpp/exp12_algorithms/results/timeout_policy_20260428
```

输出：

- `raw.json`: 每次运行的吞吐、生成文本和 top1 token。
- `summary.csv` / `summary.json`: 各 timeout budget 相对 no-timeout 的速度和输出匹配率。
- `SUMMARY.md`: 实验摘要。
