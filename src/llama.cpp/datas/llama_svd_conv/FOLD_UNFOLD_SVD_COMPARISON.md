# Fold-Unfold SVD 卷积算子对比

日期：2026-04-29

## 实验目标

本实验只比较 `llama.cpp` 目录下的卷积 SVD 实现，不引入 PyTorch 或其他官方算子作为对照。目标是观察在不同卷积核尺寸下，使用 fold-unfold SVD 路径相对不使用 fold-unfold 的 im2col SVD 路径能带来多少速度提升。

对比口径：

- 不使用 fold-unfold：先把输入特征图展开为 im2col 矩阵，再调用 `ggml_mul_mat_svd` 计算 SVD 卷积。
- 使用 fold-unfold：把 SVD 的 `V` 因子 fold 回卷积核形式，先执行 rank 通道卷积，再用 `U` 因子执行 `1x1` 卷积恢复输出通道。

## 复现方式

编译：

```bash
cmake --build /home/tianruiming/CE_ADA_LLAMA/build-release-current --target bench_conv_ops -j4
```

运行：

```bash
taskset -c 40-59 ./build-release-current/bench_conv_ops
```

当前容器可用 CPU 范围为 `0-59`，因此本轮使用 `40-59` 绑定；早先文档中的 `60-79` 在当前运行环境不可用。

画图：

```bash
python /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/generate_perf_charts.py
```

输出图：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/fold_unfold_svd_speedup_by_kernel.png`

数据文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/fold_unfold_svd_results.json`

## 实验设置

- 输入特征图：`N=1, C=128, H=28, W=28`
- 输出通道：`256`
- 卷积核：`1x1`、`3x3`、`5x5`、`7x7`
- stride：`1`
- padding：`kernel_size // 2`
- 线程数：`8`
- warmup：`5`
- repeat：`20`
- 统计值：中位数耗时
- rank 策略：`rank = min(output_channels, input_channels * kernel_h * kernel_w) / 2`

## 实测结果

| 卷积核 | Rank | 不使用 fold-unfold: im2col SVD | 使用 fold-unfold: fold SVD | 加速比 |
|---|---:|---:|---:|---:|
| 1x1 | 64 | 1.979688 ms | 0.107094 ms | 18.49x |
| 3x3 | 128 | 6.782098 ms | 0.708539 ms | 9.57x |
| 5x5 | 128 | 16.966614 ms | 1.829269 ms | 9.28x |
| 7x7 | 128 | 34.799834 ms | 3.774640 ms | 9.22x |

## 结论

在这组 llama.cpp 内部算子级测试中，fold-unfold SVD 路径在所有卷积核尺寸下都明显快于不使用 fold-unfold 的 im2col SVD 路径。

现象上，`1x1` 的加速比最高，达到约 `18.49x`；`3x3`、`5x5`、`7x7` 则稳定在约 `9.2x` 到 `9.6x`。随着卷积核变大，不使用 fold-unfold 的 im2col SVD 路径需要显式构造更大的输入展开矩阵，耗时增长更快；fold-unfold 路径把第一阶段保留为卷积形式，再接一个 `1x1` 通道投影，因此更能控制中间数据和计算调度开销。
