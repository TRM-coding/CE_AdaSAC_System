# Transformer SVD `mul_mat_svd` 算子性能对比

Date: 2026-04-28

## 目标

验证 Exp10 后实现的融合 SVD 矩阵乘算子：

```cpp
ggml_mul_mat_svd(ctx, w_shape, v, u, input, 0)
```

与非融合 baseline 的性能差异：

```cpp
tmp = ggml_mul_mat(ctx, v, input);
out = ggml_mul_mat(ctx, u, tmp);
```

本实验是纯算子实验，不讨论 SVD 截断秩策略。因此实验变量改为不同矩阵规模。CSV 中的 `mid_dim` 表示两段乘法的中间维度 K，不表示截断秩。

## 工具

Source:

- `bench_svd_mul_mat.cpp`

CMake target:

- `bench_svd_mul_mat_transformer`

Build:

```bash
cmake --build /home/tianruiming/CE_ADA_LLAMA/build-release-current -j 16 --target bench_svd_mul_mat_transformer
```

Run:

```bash
/home/tianruiming/CE_ADA_LLAMA/build-release-current/bench_svd_mul_mat_transformer \
  --threads 8 --warmup 10 --repeat 100 --n-cols 1 --type f32

/home/tianruiming/CE_ADA_LLAMA/build-release-current/bench_svd_mul_mat_transformer \
  --threads 8 --warmup 10 --repeat 100 --n-cols 1 --type f16

/home/tianruiming/CE_ADA_LLAMA/build-release-current/bench_svd_mul_mat_transformer \
  --threads 8 --warmup 10 --repeat 100 --n-cols 1 --type q4_0
```

## 环境

- CPU: Intel Xeon Gold 5218R @ 2.10 GHz
- CPU topology: 2 sockets, 20 cores/socket, 2 threads/core, 80 logical CPUs
- ISA: AVX2, AVX512F, AVX512_VNNI
- Threads: 8
- Decode shape: single token, `n_cols = 1`

## 结果

所有 case 的融合路径与非融合路径输出均一致：`max_abs_diff = 0.000000`。

| 类型 | 矩阵规模 in->out | K | 融合 median ms | 两次 `mul_mat` median ms | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 | 512->512 | 512 | 0.039577 | 0.177924 | 4.50x |
| F32 | 1024->1024 | 1024 | 0.062369 | 0.158444 | 2.54x |
| F32 | 1536->1536 | 1536 | 0.132781 | 0.135812 | 1.02x |
| F32 | 1536->8960 | 1536 | 0.916568 | 1.074027 | 1.17x |
| F32 | 8960->1536 | 1536 | 0.947725 | 0.963664 | 1.02x |
| F16 | 512->512 | 512 | 0.035041 | 0.032792 | 0.94x |
| F16 | 1024->1024 | 1024 | 0.057107 | 0.082978 | 1.45x |
| F16 | 1536->1536 | 1536 | 0.095483 | 0.204986 | 2.15x |
| F16 | 1536->8960 | 1536 | 0.366040 | 0.494398 | 1.35x |
| F16 | 8960->1536 | 1536 | 0.352014 | 0.419523 | 1.19x |
| Q4_0 | 512->512 | 512 | 0.041617 | 0.027007 | 0.65x |
| Q4_0 | 1024->1024 | 1024 | 0.055103 | 0.066606 | 1.21x |
| Q4_0 | 1536->1536 | 1536 | 0.075644 | 0.121518 | 1.61x |
| Q4_0 | 1536->8960 | 1536 | 0.236258 | 0.266596 | 1.13x |
| Q4_0 | 8960->1536 | 1536 | 0.238564 | 0.307354 | 1.29x |

Raw CSV files:

- `results/scale_f32_threads8_ncols1_repeat100.csv`
- `results/scale_f16_threads8_ncols1_repeat100.csv`
- `results/scale_q4_0_threads8_ncols1_repeat100.csv`

中文论文图：

- `figures/svd_mul_mat_scale_speedup_zh.pdf`
- `figures/svd_mul_mat_scale_speedup_zh.svg`
- `figures/svd_mul_mat_scale_speedup_zh.png`
- `figures/svd_mul_mat_scale_latency_zh.pdf`
- `figures/svd_mul_mat_scale_latency_zh.svg`
- `figures/svd_mul_mat_scale_latency_zh.png`

绘图脚本：

- `plot_svd_mul_mat_results.py`

## 结论

在 single-token 矩阵乘算子口径下，融合 `ggml_mul_mat_svd` 与两次官方 `ggml_mul_mat` 的输出一致。除少数很小规模 case 外，融合算子整体更快。

较小规模下结果容易受调度开销和计时波动影响，例如 F16 `512->512` 和 Q4_0 `512->512` 出现回退。中等及较大规模下，融合算子通常具有稳定收益，尤其 F16 `1536->1536` 达到约 `2.15x`，Q4_0 `1536->1536` 达到约 `1.61x`。

旧版 `rank=192/384/768` 图和 CSV 是早期草稿口径，容易被误解为截断秩实验，不建议放入毕设正文。论文中应使用带 `scale_` 前缀的数据和图。
