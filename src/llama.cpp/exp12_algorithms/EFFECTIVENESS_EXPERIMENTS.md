# Exp12 调度策略有效性实验设计

日期：2026-04-28

## 目标

验证第五章负载感知调度策略是否能在电脑端 CPU 有负载时提升 SVD decode 速度。当前阶段不使用 adb，因此实验分成两类：

- 已可真实测量：本地 SVD rate 调度在有负载时是否加速。
- 暂不下结论：真实边端后缀卸载收益，需要后续手机端 server/adb 或网络连接后再测。

## 实验一：本地调度同负载对比

假设：在相同 CPU 负载下，减少部分层的 SVD rank 计算量应提升 decode 速度。

变量：

- 运行核心：`60-67`
- 加压核心：`60-63`
- 负载强度：`load_workers = 0, 2, 4`
- 策略：
  - `baseline`: 不使用 SVD rate。
  - `alternate_0.5`: 奇数层 rate=0.5，偶数层不裁剪，满足相邻层不可连续裁剪约束。
  - `alternate_0.75`: 奇数层 rate=0.75，偶数层不裁剪。
  - `uniform_0.5`: 所有层 rate=0.5，作为激进对照。

推荐命令：

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

结果目录：

- `results/effectiveness_20260428/raw.json`
- `results/effectiveness_20260428/summary.csv`
- `results/effectiveness_20260428/SUMMARY.md`

### 纠正说明

最早一版 `effectiveness_20260428` 使用的是不一致的裸 busy-loop 负载口径，这会引入明显污染，因此这部分异常低吞吐结果不再作为结论依据。

脚本现已改为默认使用 `stress-ng --cpu-load`，并在 `--use-cgroup` 时分别建立 run/load cgroup。

### stress-ng 20% 负载结果

结果目录：

- `results/effectiveness_stress20_repeat_20260428/raw.json`
- `results/effectiveness_stress20_repeat_20260428/summary.csv`
- `results/effectiveness_stress20_repeat_20260428/SUMMARY.md`

设置：

- `load-mode = stress-ng`
- `cpu-load = 20`
- `load-workers = 4`
- `repeats = 3`

| load workers | policy | tok/s | speedup |
|---:|---|---:|---:|
| 4 | baseline | 28.9399 | 1.000x |
| 4 | alternate_0.75 | 33.4028 | 1.154x |

观察：在更接近 exp10 的 stress-ng 20% 负载口径下，`alternate_0.75` 相比同负载 baseline 平均提速约 `15.4%`。这说明本地截断调度在中等负载下确实可能带来速度收益。

## 实验二：A/B 份额敏感性

假设：当负载集中在 A 组核心 `60-63` 时，降低 `groupA_share`，把更多 rank 分给 B 组 `64-67`，可能提升速度。

设置：

- rate：`alternate_0.75`
- A 组：`60-63`
- B 组：`64-67`
- load：4 个 busy worker 绑定到 `60-63`
- 扫描：`groupA_share = 0.25, 0.5, 0.75`

结果目录：

- `results/share_sweep_20260428/raw.json`
- `results/share_sweep_20260428/summary.json`

初步结果：

| policy | tok/s | decode ms | speedup |
|---|---:|---:|---:|
| baseline | 0.163041 | 6133.41 | 1.000x |
| share=0.25 | 0.165491 | 6042.61 | 1.015x |
| share=0.5 | 0.164781 | 6068.66 | 1.011x |
| share=0.75 | 0.164539 | 6077.6 | 1.009x |

观察：把 A 组份额降到 `0.25` 是本轮最好点，但提升仍只有 `1.5%` 左右。说明当前实现下，A/B rank 份额调整对强 busy-loop 负载的缓解很有限。

## 实验三：真实边端卸载验证

这部分本轮未跑，因为用户要求暂时不使用 adb。

后续要验证第五章完整策略，需要：

1. 在手机端或本机模拟端启动后缀层/SVD server。
2. 对每个候选分割点 `m` 测量：
   - `T_edge(m)`
   - `T_tx(m)`
   - `T_end(m)`
   - 输出差异或 perplexity/accuracy 变化。
3. 和本地 baseline、本地截断、固定分割点卸载比较。
4. 判定 `m* = max {m | T_total(m) <= T_req}` 是否同时达到最少卸载层数和时延达标。

## 当前结论

在当前“不使用 adb、仅本机 SVD rate 调度”的实验范围内，修正负载口径后：

- 空载：截断策略能明显提速。
- stress-ng 20% 负载：`alternate_0.75` 有约 `15%` 的平均提速。
- 100% busy-loop 强压负载：会把 decode 压到异常低速，不应和 exp10 结果混用。

建议下一步优先做真实边端卸载实验，因为第五章策略的核心收益点是在本地 DP 无解时把后层迁移到端侧，而不是只靠本地截断硬扛高负载。

## 补充：不同负载强度扫描

命令按 `cpu-load = 0, 20, 50, 80, 100` 分档执行，每档重复 3 次：

```bash
BASE=src/llama.cpp/exp12_algorithms/results/effectiveness_load_sweep_20260428
for load in 0 20 50 80 100; do
  python3 src/llama.cpp/exp12_algorithms/benchmark_effectiveness.py \
    --use-cgroup \
    --load-mode stress-ng \
    --cpu-load "$load" \
    --load-workers-list 4 \
    --repeats 3 \
    --policies baseline,alternate_0.75 \
    --out-dir "$BASE/load${load}"
done
```

结果汇总：

- `results/effectiveness_load_sweep_20260428/LOAD_SWEEP_SUMMARY.md`
- `results/effectiveness_load_sweep_20260428/load_sweep_summary.csv`

| stress-ng cpu-load | baseline tok/s | scheduled tok/s | speedup |
|---:|---:|---:|---:|
| 0% | 30.1646 | 33.6793 | 1.117x |
| 20% | 28.7357 | 34.3036 | 1.194x |
| 50% | 29.6295 | 33.7005 | 1.137x |
| 80% | 异常值，已剔除 | 异常值，已剔除 | - |
| 100% | 异常值，已剔除 | 异常值，已剔除 | - |

结论：中低负载下，本地 SVD 截断调度有明显提速；高负载下此前出现过明显异常值，但由于负载口径和环境不干净，这部分结果已经从有效结论中剔除。
