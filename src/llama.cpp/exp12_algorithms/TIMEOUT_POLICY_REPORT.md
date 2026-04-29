# Exp12 每层超时策略有效性实验报告

日期：2026-04-28

## 修正说明

这份报告是早期 timeout 接入实验记录，保留作历史参考。后续检查发现早期 local split 语义存在问题：`rate` 会在 local split 前提前截掉 tail rank，导致 no-timeout 下也可能丢失 tail 贡献。因此本文中的速度和输出代理结果不能作为最终 timeout 策略结论。

修复后的语义、PPL 结果和 runtime timeout/drop 计数见：

`CORRECTED_TIMEOUT_AND_SPLIT_REPORT.md`

当前可信结论以修正版报告为准。

## 目的

验证论文中的超时量分配策略接入 runtime 后：

1. 是否能通过丢弃超时的 minor rank slice 提升速度。
2. 丢弃后是否在轻量输出代理指标上造成明显损失。

本实验不使用 adb，不跑完整 perplexity。损失代理指标包括：

- `generated_text` 是否与 no-timeout 参考一致。
- top-1 next token id 是否与 no-timeout 参考一致。

## 实验设置

- 模型：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 程序：`build-release-current/decode_svd_test`
- run cgroup：`60-67`
- load cgroup：`60-63`
- load：`stress-ng --cpu 4 --cpu-method matrixprod`
- SVD rate：奇数层 `0.75`，偶数层 `0`
- A/B split：A=`60-63`，B=`64-67`，`groupA_share=0.25`
- timeout budget：`0/2/8/16 ms`，按论文公式分配到被裁剪层

其中 `timeout_budget_0` 是 no-timeout 参考：同样的 rate 和 split，但不允许 minor slice 超时丢弃。

## 命令

```bash
python3 src/llama.cpp/exp12_algorithms/benchmark_timeout_policy.py \
  --use-cgroup \
  --cpu-loads 20,50 \
  --timeout-budgets-ms 0,2,8,16 \
  --repeats 3 \
  --out-dir src/llama.cpp/exp12_algorithms/results/timeout_policy_20260428
```

强负载补充：

```bash
python3 src/llama.cpp/exp12_algorithms/benchmark_timeout_policy.py \
  --use-cgroup \
  --cpu-loads 80 \
  --timeout-budgets-ms 0,2,8 \
  --repeats 2 \
  --out-dir src/llama.cpp/exp12_algorithms/results/timeout_policy_load80_20260428
```

## 汇总

汇总文件：

- `results/timeout_policy_combined_20260428/SUMMARY.md`
- `results/timeout_policy_combined_20260428/summary.csv`

| cpu-load | policy | runs | tok/s mean | tok/s median | speedup mean | speedup median | text match | top1 match |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | `timeout_budget_0` | 3 | 31.7458 | 31.7275 | 1.000x | 1.000x | 1.000 | 1.000 |
| 20% | `timeout_budget_2` | 3 | 32.3662 | 32.4727 | 1.020x | 1.023x | 1.000 | 1.000 |
| 20% | `timeout_budget_8` | 3 | 31.9154 | 32.3083 | 1.005x | 1.018x | 1.000 | 1.000 |
| 20% | `timeout_budget_16` | 3 | 23.0352 | 31.8556 | 0.726x | 1.004x | 1.000 | 1.000 |
| 50% | `timeout_budget_0` | 3 | 31.8236 | 32.1916 | 1.000x | 1.000x | 1.000 | 1.000 |
| 50% | `timeout_budget_2` | 3 | 31.7082 | 31.6183 | 0.996x | 0.982x | 1.000 | 1.000 |
| 50% | `timeout_budget_8` | 3 | 31.8354 | 31.7372 | 1.000x | 0.986x | 1.000 | 1.000 |
| 50% | `timeout_budget_16` | 3 | 21.9296 | 30.4109 | 0.689x | 0.945x | 1.000 | 1.000 |
| 80% | `timeout_budget_0` | 2 | 0.15728 | 0.15728 | 1.000x | 1.000x | 1.000 | 1.000 |
| 80% | `timeout_budget_2` | 2 | 0.153362 | 0.153362 | 0.975x | 0.975x | 1.000 | 1.000 |
| 80% | `timeout_budget_8` | 2 | 0.152323 | 0.152323 | 0.968x | 0.968x | 1.000 | 1.000 |

## 观察

1. 输出代理指标上，本轮所有 timeout budget 的 `generated_text` 和 top-1 token 都与 no-timeout 参考一致。因此在这个单 prompt、单 token 口径下，未观察到可见输出损失。
2. `20%` 负载下，小预算 `2 ms` 最稳，平均约 `1.02x`，中位数约 `1.023x`。
3. `50%` 负载下，timeout 策略基本没有稳定速度收益。
4. `80%` 强负载下，timeout 策略反而略慢。此时系统已进入强饱和，等待/同步成本会吞掉收益。
5. `16 ms` budget 出现过明显慢速 outlier，说明超时等待预算过大可能引入尾延迟风险。

## 结论

当前 per-layer timeout 策略已经能实际接入 runtime，并且在轻量输出代理指标上没有观察到损失。但速度收益只在较温和负载下出现，且幅度较小；在中高负载或预算过大时并不稳定。

建议后续将 `2 ms` 级别的小预算作为候选默认值，并增加更多 prompt/perplexity 实验来验证真实精度损失。
