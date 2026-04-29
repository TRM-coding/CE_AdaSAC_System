# Exp12 Per-Layer Timeout Policy Benchmark

日期：2026-04-28 19:45:48

## 设置

- 模型：`/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 程序：`/home/tianruiming/CE_ADA_LLAMA/build-release-current/decode_svd_test`
- 运行核心：`60-67`
- 加压核心：`60-63`
- stress-ng cpu-load：`80`
- repeats：`2`
- groupA share：`0.25`
- SVD rate policy：奇数层 `0.75`，偶数层 `0`

## 汇总

| cpu-load | policy | tok/s | speedup | text match | top1 match |
|---:|---|---:|---:|---:|---:|
| 80% | `timeout_budget_0` | 0.15728 | 1.000x | 1.000 | 1.000 |
| 80% | `timeout_budget_2` | 0.153362 | 0.975x | 1.000 | 1.000 |
| 80% | `timeout_budget_8` | 0.152323 | 0.968x | 1.000 | 1.000 |

## 解释

- `timeout_budget_0` 是 no-timeout 参考：使用同样的 SVD rate 和 A/B split，但不允许 minor slice 超时丢弃。
- 其他策略按论文公式把总 timeout budget 分配到被裁剪层。
- `text match` 和 `top1 match` 是轻量损失代理指标，不等价于完整 perplexity/accuracy。
