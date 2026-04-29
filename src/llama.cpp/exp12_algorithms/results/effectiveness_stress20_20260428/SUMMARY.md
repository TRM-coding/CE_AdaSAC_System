# Exp12 调度有效性实验

日期：2026-04-28 18:09:22

## 实验目的

验证在电脑端 CPU 存在额外负载时，SVD 截断率调度是否能提升 decode 速度。
本实验不使用 adb，不验证真实手机后缀卸载；这里只验证当前可落地的本地 SVD rate 调度。

## 设置

- 模型：`/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 程序：`/home/tianruiming/CE_ADA_LLAMA/build-release-current/decode_svd_test`
- 运行核心：`60-67`
- 加压核心：`60-63`
- 负载模式：`stress-ng`
- stress-ng cpu-load：`20`
- tokens：`1`
- repeats：`1`
- policies：`baseline,alternate_0.75,uniform_0.5`
- cgroup：`enabled`

## 汇总

| load workers | policy | runs | tok/s mean | decode ms mean | speedup |
|---:|---|---:|---:|---:|---:|
| 0 | `alternate_0.75` | 1 | 32.1293 | 31.1242 | 1.175x |
| 0 | `baseline` | 1 | 27.3555 | 36.5558 | 1.000x |
| 0 | `uniform_0.5` | 1 | 34.2193 | 29.2233 | 1.251x |
| 4 | `alternate_0.75` | 1 | 33.4303 | 29.913 | 1.049x |
| 4 | `baseline` | 1 | 31.8607 | 31.3866 | 1.000x |
| 4 | `uniform_0.5` | 1 | 5.22983 | 191.211 | 0.164x |

## 判定标准

- 同一 `load_workers` 下，`speedup > 1.0x` 表示该策略相对同负载 baseline 提速。
- `alternate_*` 策略模拟第五章相邻层不可同时裁剪约束。
- `uniform_*` 策略作为激进截断对照，速度可能更快，但精度风险也更高。
