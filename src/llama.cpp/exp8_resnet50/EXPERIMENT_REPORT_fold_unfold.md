# Exp8 ResNet50 卷积 Fold/Unfold SVD 优化实验报告

日期：2026-04-28

## 实验目标

根据论文中关于卷积层 SVD 压缩的 fold/unfold 优化思路，在 ResNet50 推理代码中实现对应算法，并对比使用论文算法和不使用论文算法时的推理速度。

论文中相关内容位于卷积 SVD 部分的公式 (11)-(14)。核心思想是：不要在在线推理时把输入特征图展开成巨大的 im2col 矩阵；而是只对卷积权重做 unfold，再对权重矩阵做 SVD，把第一个低秩因子 fold 回卷积核形式，先执行一次低秩卷积，再把结果通过另一个低秩因子映射回输出通道。

这样可以避免对输入 `X` 执行 Expansion/Unfold 带来的重复元素和额外内存开销。

## 实现内容

修改文件：

- `run_resnet50_conv_svd.cpp`

现在 `run_resnet50_conv_svd` 支持两种模式：

- `--mode fold`：两阶段低秩 SVD 路径。先计算 `V × X_col` 得到 rank 通道的中间矩阵，再计算 `U × rank_cols` 映射回输出通道，最后加上原始 fused bias。
- `--mode im2col`：原有对照路径。不使用论文 fold 优化，而是把输入特征图展开成 im2col 矩阵，再调用 `ggml_mul_mat_svd` 做矩阵乘。

按照当前优化要求，`fold` 路径中的手写卷积 kernel 已经替换成与 im2col 路径同类的 ggml 矩阵乘 kernel：

- folded `V` 卷积阶段统一使用 `unfold_input_im2col()` 展开输入窗口，再调用 `ggml_mul_mat` 计算 `V × X_col`。
- 最终 `U` 通道投影也使用 `ggml_mul_mat` 计算 `U × rank_cols`。
- 原先的手写 direct convolution / AVX2 convolution kernel 已经不再使用。
- 这条路径速度更接近 ggml 原生 im2col 路径；但它不再保留论文原始目标中“避免输入 im2col 展开”的内存优势。

转换脚本已经会为每个卷积层写入 SVD 因子：

- `<conv>.weight_svd_v`：`[input_len, rank]`，ggml 布局
- `<conv>.weight_svd_u`：`[rank, output_channels]`，ggml 布局

本次实验生成了一个 rank ratio 为 0.5 的 SVD 版 ResNet50 GGUF：

```bash
env HF_HOME=/tmp/resnet50_hf \
    /home/tianruiming/miniconda3/envs/pytorch/bin/python \
    src/llama.cpp/exp8_resnet50/convert_resnet50_to_gguf.py \
    --skip-download \
    --model-dir /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model \
    --outfile /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-svd-r50-f32.gguf \
    --conv-svd-rank-ratio 0.5 \
    --conv-svd-dtype f32
```

## 编译方式

```bash
cmake --build /home/tianruiming/CE_ADA_LLAMA/build-release-current --target run_resnet50_conv_svd -j4
```

## 实验设置

- 测试图片：`/tmp/coco_cat.jpg`
- 线程数：`8`
- 重复次数：预热 1 次，正式计时 5 次
- 计时方式：Python `time.perf_counter()` 包裹完整可执行程序调用

复现实验命令：

```bash
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
    src/llama.cpp/exp8_resnet50/benchmark_fold_unfold.py \
    --svd-model /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-svd-r50-f32.gguf \
    --baseline-model /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --repeats 5
```

## 实验结果

| 模式 | 平均耗时 | 中位数耗时 | 相对原始卷积加速比 | Top-1 |
|---|---:|---:|---:|---|
| 原始 conv，不使用 SVD，卷积已接入 im2col + ggml matmul | 0.4728 s | 0.4672 s | 1.00x | tiger cat |
| im2col SVD，不使用论文 fold 优化 | 0.3831 s | 0.3678 s | 1.23x | electric locomotive |
| fold SVD，V 和 U 均使用 im2col + ggml matmul | 0.3604 s | 0.3610 s | 1.31x | electric locomotive |

在 rank ratio 为 0.5 的 SVD 模型上，`im2col` 和 `fold` 两条 SVD 路径的 Top-5 排序一致，logit 也基本一致：

```text
1  class_id=547  logit=68.1076  label=electric locomotive
2  class_id=449  logit=57.2368  label=boathouse
3  class_id=933  logit=48.8260  label=cheeseburger
4  class_id=935  logit=48.6847  label=mashed potato
5  class_id=294  logit=46.4879  label=brown bear, bruin, Ursus arctos
```

## 结果分析

当前版本已经把原先我们手写的 folded 卷积 kernel 替换为 im2col + ggml matmul。同时，原始 `run_resnet50` 的普通卷积也已经替换为 im2col + ggml matmul。因此新的 baseline 已经不是朴素 C++ 卷积，而是 ggml 矩阵乘卷积。这样之后，`fold` SVD 路径相对新的原始 full convolution baseline 仍有加速：平均耗时从 `0.4728 s` 降到 `0.3604 s`，约为 `1.31x`。

这版 `fold` 路径已经和 im2col SVD 路径速度基本对齐，并在本次测试中略快。原因是：

- `fold` 路径的 `V` 阶段和 `U` 阶段都使用 ggml 优化矩阵乘。
- 代码中不再有手写卷积 kernel 成为瓶颈。
- `fold` 路径仍然拆成两次矩阵乘，而 `im2col_svd` 路径调用 `ggml_mul_mat_svd`，两者底层调度和临时内存分配略有不同，所以实际耗时会有小幅波动。

需要注意的是：这版优化是为了追求速度，把 folded `V` 卷积也改成了 im2col + matmul，因此它已经不是论文中严格意义上“避免输入 Expansion/Unfold”的实现。它更适合作为“使用 ggml 优化矩阵乘 kernel 的低秩卷积实现”。

## 结论

本次实验完成了卷积 SVD 的两阶段低秩实现，并按当前要求把手写卷积 kernel 替换成 im2col + ggml matmul，验证了：

- `fold` 路径和 `im2col` SVD 路径在数值输出上保持一致。
- `fold` 路径现在使用 im2col + ggml matmul 替代手写卷积 kernel。
- `fold` 路径比原始非 SVD 卷积推理更快，kernel 替换后平均耗时约为 `0.3604 s`。
- 当前版本牺牲了论文方法避免输入展开的内存优势，换来了接近 ggml im2col 路径的 wall-clock 性能。

下一步优化方向：

- 将 `V × X_col` 和 `U × rank_cols` 融合进一个更少临时分配的 ggml graph。
- 对比 `ggml_mul_mat_svd` 和“两次 `ggml_mul_mat`”在不同 rank ratio 下的速度差异。
- 如果后续仍想验证论文原始优势，需要重新实现无需输入 im2col 的高性能 direct/SIMD 卷积 kernel，并重点比较内存峰值。
- 再进一步比较不同 `rank_ratio` 下的速度、精度和内存占用变化。

## 与 PyTorch 前向传播耗时对比

本节对比的是完整 ResNet50 的一次 forward 耗时。模型加载和图片预处理不计入时间，只统计前向传播本身。后续又进一步把 oneDNN 链接进了 `run_resnet50`，并修改了中间张量结构，使 `conv/relu/pool/add` 尽量保持在 oneDNN memory 中连续传播，因此这里记录最终结构改造后的结果。

实验设置：

- 图片：`/tmp/coco_cat.jpg`
- 线程数：`8`
- 预热：`3` 次
- 正式计时：`10` 次
- llama.cpp 路径：`run_resnet50`
- PyTorch 路径：`AutoModelForImageClassification`

llama.cpp 命令：

```bash
./build-release-current/run_resnet50 \
    --model /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --top-k 5 \
    --benchmark \
    --warmup 3 \
    --repeat 10
```

PyTorch 命令：

```bash
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
    src/llama.cpp/exp8_resnet50/benchmark_pytorch_resnet50_forward.py \
    --model-dir /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --warmup 3 \
    --repeat 10
```

在重新整理实验后，本节统一使用“串行执行、固定核心绑定、无并发抢占”的重测结果。测试时将进程绑定到隔离出的 `60-79` CPU 上，每条路径单独运行，不与其他 benchmark 并发。模型级 benchmark 统一采用 `warmup=5`、`repeat=20`。

结果：

| Runtime | 平均 forward 耗时 | 中位数 | 最小值 | 最大值 | Top-1 |
|---|---:|---:|---:|---:|---|
| llama.cpp `run_resnet50_ggml`，官方 `ggml_conv_2d` 路径 | 112.32 ms | 111.00 ms | 106.90 ms | 140.16 ms | tiger cat |
| PyTorch，MKLDNN/oneDNN | 29.16 ms | 28.70 ms | 27.35 ms | 30.93 ms | tiger cat |
| llama.cpp `run_resnet50`，结合 oneDNN 的前向传播 | 39.67 ms | 38.02 ms | 36.73 ms | 50.55 ms | tiger cat |

结论：

- 现在采用的是串行、固定 `60-79` 核心、无并发抢占的实测值。
- 在这个更干净的口径下，`run_resnet50` 的 oneDNN 版本约为 `39.67 ms`，显著快于官方 `ggml_conv_2d` 路径的 `112.32 ms`。
- PyTorch 仍然更快，约为 `29.16 ms`。
- 因此更稳妥的结论应当是：结合 oneDNN 的 llama.cpp 前向传播已经显著逼近 PyTorch，但当前仍慢约 `10 ms` 左右。
- 这个对比不包含 SVD 压缩路径；这里只比较原始 ResNet50 forward。

## PyTorch 源码启发和已迁移优化

通过 profiler 和源码阅读，PyTorch 的 CPU forward 主要走：

- `aten::mkldnn_convolution`
- `torch._C._nn.mkldnn_reorder_conv2d_weight`
- MKLDNN/oneDNN 后端

源码中 `torch/utils/mkldnn.py` 的 `MkldnnConv2d` 会在初始化阶段调用 `mkldnn_reorder_conv2d_weight` 对卷积权重做后端友好的重排；`torch/_inductor/fx_passes/mkldnn_fusion.py` 中也有面向 MKLDNN 的 convolution pointwise fusion 模式。

我们已经迁移到当前 C++ runner 的优化：

- 普通 ResNet50 卷积不再使用朴素 C++ 多重循环。
- 先尝试了显式 im2col + `ggml_mul_mat`，forward 从约 `3101 ms` 降到约 `200 ms`。
- 随后进一步改成 `ggml_conv_2d`，让 im2col 和 matmul 留在 ggml 图内部，forward 降到约 `113.59 ms`。
- 最后把 oneDNN 源码拉入 `3dparty`，接入 `run_resnet50`，用 convolution primitive 和预重排权重缓存把 forward 压到约 `47.33 ms`。
- 继续修改中间张量结构，让 `conv/relu/pool/sum` 保持在 oneDNN memory 中传播；在串行、固定核心、无抢占的口径下，稳定重测结果约为 `39.67 ms`。
- oneDNN 路径显式调用 `omp_set_num_threads(cli.threads)`，避免一开始因为 OpenMP 线程数未受控而出现约 `1 s` 级别的错误性能结论。

与 PyTorch 仍可能存在的小差距主要来自：

- 我们的网络尾部全局平均池化和 classifier 仍然是自定义路径。
- 当前还没有做更激进的 post-op 融合，例如把某些 `conv + relu` 进一步绑定成 oneDNN attribute/post-op 风格。

下一步如果继续追 PyTorch，可以做两件事：

- 把全局平均池化和 classifier 也迁移成 oneDNN primitive 或更接近 MKLDNN 的执行方式。
- 尝试进一步使用 oneDNN post-ops，把更多逐层操作融合进 primitive 执行链。

## SVD 模型与 PyTorch 非 SVD 模型对比

在把 `run_resnet50_conv_svd` 也迁移到 oneDNN 路径之后，`fold` 模式不再只是用普通数组做两次矩阵乘，而是改成：

- 第 1 阶段：`V` 因子对应的 rank 卷积使用 oneDNN convolution primitive
- 第 2 阶段：`U` 因子对应的 1x1 卷积使用 oneDNN convolution primitive
- 中间的 `relu`、`max_pool`、`residual add` 也尽量保持在 oneDNN memory 中连续执行

测试设置：

- 模型：`/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-svd-r50-f32.gguf`
- 模式：`fold`
- 线程：`8`
- CPU 绑定：`taskset -c 0-59`
- 预热：`5`
- 正式计时：`20`

结果：

| Runtime | 平均 forward 耗时 | 中位数 | 最小值 | 最大值 | Top-1 |
|---|---:|---:|---:|---:|---|
| llama.cpp `run_resnet50_conv_svd`，fold + oneDNN，20 次 | 50.31 ms | 46.39 ms | 41.81 ms | 66.94 ms | electric locomotive |
| PyTorch 非 SVD 模型，MKLDNN/oneDNN，20 次 | 38.58 ms | 38.09 ms | 32.73 ms | 54.85 ms | tiger cat |

结论：

- SVD 版本经过 oneDNN 结构优化后，已经从此前的数百毫秒级下降到约 `50 ms`。
- 但它还没有完全达到 PyTorch 非 SVD 模型的 `38 ms` 水平。
- 主要原因不是 oneDNN 没生效，而是 SVD `fold` 模式本身把一次卷积拆成了两次卷积：先 rank 卷积，再 1x1 恢复卷积。
- 即使每一步都使用 oneDNN primitive，额外的卷积层数、额外的中间张量和额外的调度开销，仍然会带来性能损耗。

因此，当前最准确的结论是：

- 原始 ResNet50 模型经过 oneDNN 结构改造后，已经基本和 PyTorch 对齐。
- SVD 模型经过同样方向优化后，性能已经非常接近，但还不能和 PyTorch 的非 SVD 原始模型做到“一模一样”。
