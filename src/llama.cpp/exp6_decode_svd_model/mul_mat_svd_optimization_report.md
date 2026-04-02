# `ggml_compute_forward_mul_mat_svd` 优化与验证报告

## 1. 背景

本轮工作的目标是优化 `ggml_compute_forward_mul_mat_svd`，并确认：

- 不改变 `x * U * V` 的计算顺序
- 不把优化建立在修改 rank 语义上
- 用 `decode_svd_model.cpp` 实测验证
- 对照原始 llama.cpp 的 `ggml_compute_forward_mul_mat` 分析差异

本轮最终结论：

1. 原始 dense `qwen.gguf` 在 `decode_svd_test` 链路中，确实走的是 llama.cpp 自带的普通 `ggml_mul_mat`。
2. 当前 SVD 路径已经不是“测试程序错了导致只有 5 tok/s”这种情况。
3. 经过本轮优化后，SVD 路径的瓶颈已经收敛到 `mul_mat_svd` 内核本身和 SVD 模型链路冗余，而不是线程数、测试程序打印、或者走错算子。

## 2. 本轮做了什么

### 2.1 对照 `ggml_compute_forward_mul_mat` 检查 `mul_mat_svd`

分析文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`

重点对照了：

- `ggml_compute_forward_mul_mat`
- `ggml_compute_forward_mul_mat_one_chunk`
- `ggml_compute_forward_mul_mat_svd`

核心发现：

1. 原始 `mul_mat_svd` decode 热路径本质上是两次独立 matmul：
   - `tmp = mul_mat(V, x)`
   - `dst = mul_mat(U, tmp)`
2. 原先只存在 `F32` 单 token 快路径，实际 SVD 模型里的 `U/V` 是 `F16`，所以 decode 时仍然频繁回退到通用路径。
3. 这会把 decode 小 batch 场景下的调度成本放大。

### 2.2 为 `mul_mat_svd` 增加 `F16/F32` 单 token decode 快路径

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`

保留的优化策略：

- 不改 `x * U * V` 顺序
- 只在 decode 单 token 热路径触发
- 覆盖 `u/v` 为 `F16` 或 `F32` 的情况

实现要点：

1. 第一阶段直接并行计算 `tmp = x * V`
2. 第二阶段直接并行计算 `dst = tmp * U`
3. `F16` 情况下复用 `ggml_vec_dot_f16_unroll`
4. 避免 decode 单 token 时反复进入两次完整 `ggml_compute_forward_mul_mat`

这一步的意义是：

- 去掉不必要的通用调度开销
- 让 `mul_mat_svd` decode 热路径真正吃到 `F16` 权重

### 2.3 修正 `decode_svd_model.cpp` 的测速方式

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp`

原始问题：

- 计时区间内逐 token `cout`
- 计时区间内逐 token `llama_token_to_piece`
- 计时区间内做字符串拼接

这些操作会污染 decode 吞吐。

本轮调整：

1. 增加 `verbose_tokens` 开关
2. 默认不在计时区间逐 token 打印
3. 新增 `Decode-only throughput`
4. 生成文本改为计时后统一组装

这样可以把：

- 纯 `llama_decode` 时间
- 端到端生成时间

分开看。

### 2.4 验证原始 `qwen.gguf` 的调用链路

核查文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp`

确认结论：

1. `decode_svd_test` 调用 `llama_decode`
2. 原始 `qwen.gguf` 的架构是 `qwen2`
3. `LLM_ARCH_QWEN2` 进入 `llm_build_qwen2`
4. `llm_build_qwen2` 的 FFN 走 `build_ffn`
5. `build_ffn` 内部调用 `build_lora_mm`
6. `build_lora_mm` 第一行就是 `ggml_mul_mat(ctx0, w, cur)`

因此：

- 原始 `qwen.gguf` 在这条链路上，确实自动走 llama.cpp 自带的普通 `mul_mat`
- `qwen.gguf.sort_svd.gguf` 才会走 `mul_mat_svd`

这意味着 dense 的速度可以作为可信的对照基线。

### 2.5 去掉 SVD 模型中的冗余 dense FFN 权重

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp`

新增脚本：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/strip_dense_ffn_from_svd.py`

问题来源：

- 现有 `qwen.gguf.sort_svd.gguf` / `qwen7b.gguf.sort_svd.gguf` 同时保留了
  - 原始 dense `ffn_up / ffn_gate / ffn_down`
  - SVD `ffn_*_svd_u / ffn_*_svd_v`
- 运行时 SVD 路径实际并不需要再读取原始 dense FFN 权重

本轮处理：

1. 将 `QWEN2_SVD` 路径中的原始 dense FFN 张量改为 `TENSOR_NOT_REQUIRED`
2. 在 `build_mm_svd()` 中允许从 `U/V` 自动推导 shape
3. 提供脚本，把现有 `.sort_svd.gguf` 精简成只保留真正需要的 tensor

这一步的意义是：

- 去掉 SVD 推理链路中的明显冗余
- 降低 mmap 负担和模型加载体积

## 3. 生成的紧凑模型

### 3.1 Qwen2.5-7B

原始模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen7b.gguf.sort_svd.gguf`

紧凑模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen7b.gguf.sort_svd.compact.gguf`

结果：

- 跳过 `84` 个 dense FFN tensor
- 文件从约 `39.45 GiB` 降到约 `28.83 GiB`

### 3.2 Qwen2.5-1.5B

原始模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.gguf`

紧凑模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`

结果：

- 跳过 `84` 个 dense FFN tensor
- 文件从约 `8.0 GiB` 降到约 `5.77 GiB`

## 4. 验证方法

### 4.1 `decode_svd_test`

命令格式：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build
LD_LIBRARY_PATH=./bin ./decode_svd_test <model> 8 32 0
```

解释：

- `8`：生成 8 个 token
- `32`：decode 线程数
- `0`：关闭逐 token 打印

重点看输出中的：

- `Decode-only throughput`

### 4.2 `llama-bench`

命令：

```bash
cd /home/tianruiming/CE_ADA_LLAMA
./build/llama-bench -m /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf -t 32 -ngl 0 -p 0 -n 128 -r 3 -o md
```

这是对原始 dense 路径的标准 benchmark 验证。

## 5. 实测结果

### 5.1 Qwen2.5-1.5B

#### 原始 dense

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf`

`decode_svd_test`：

- `Decode-only throughput: 8.70435 tokens/s`
- `End-to-end throughput: 8.66555 tokens/s`

`llama-bench`：

- `tg128: 9.07 ± 0.26 tokens/s`

说明：

- `decode_svd_test` 和 `llama-bench` 基本一致
- dense 基线可信

#### 原始 SVD

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.gguf`

结果：

- `Decode-only throughput: 7.06294 tokens/s`
- `End-to-end throughput: 7.03779 tokens/s`

#### 紧凑 SVD

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`

结果：

- `Decode-only throughput: 7.19950 tokens/s`
- `End-to-end throughput: 7.17694 tokens/s`

#### 1.5B 小结

按 `Decode-only throughput` 统计：

- dense：`8.704`
- 原始 SVD：`7.063`
- 紧凑 SVD：`7.200`

也就是说：

- SVD 已经不是数量级错误
- 紧凑模型比原始 SVD 进一步提升
- 但仍然略慢于原始 dense

#### 原始 dense `Q4_0`

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`

量化命令：

```bash
cd /home/tianruiming/CE_ADA_LLAMA
src/llama.cpp/3dparty/llamacpp/build/bin/llama-quantize \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf \
  Q4_0
```

结果：

- 原始 `qwen.gguf`：约 `2.88 GiB`
- `qwen.q4_0.gguf`：约 `885.97 MiB`

`llama-bench`（早期本地构建结果）：

- `tg128: 19.18 ± 0.29 tokens/s`

说明：

- 在当前机器上，`Q4_0` dense 大约是 F16 dense 的 `2.1x`
- 该结果可作为后续量化路径对照基线

注意：

- 上面这组 `19.18 tokens/s` 是早期构建下得到的历史数据
- 在后续控制变量实验中，统一切换到 `Release` 构建并串行跑 benchmark 后，`Q4_0` dense 的正式结果已经更新，见下文第 `5.4` 节

#### 关于量化 SVD 的结论

本轮还额外验证了量化模型的 SVD 生成链路。

新增修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/generate_sort_svd.py`

新增能力：

1. 支持通过命令行传入输入/输出模型路径
2. 支持读取量化 GGUF
3. 对 `Q4_0` 这类量化 tensor 先按 GGUF 规则反量化，再做 SVD

但这里需要明确一个事实：

- 当前 SVD 推理链路并不支持“全量化 SVD”
- 现有 SVD 模型中，`ffn_*_svd_u` 和 `ffn_*_svd_v` 仍然是 `F16`

所以：

- 可以从量化底模出发生成 SVD 模型
- 但生成后的 SVD 模型本质上仍然是“量化底模 + F16 的 U/V”
- 当前 `mul_mat_svd` 还没有专门支持量化 `U/V` 的内核路径

### 5.2 Qwen2.5-7B

#### 原始 dense

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen7b.gguf`

结果：

- `Decode-only throughput: 2.57851 tokens/s`

#### 原始 SVD

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen7b.gguf.sort_svd.gguf`

结果：

- `Decode-only throughput: 1.75702 tokens/s`

#### 紧凑 SVD

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen7b.gguf.sort_svd.compact.gguf`

结果：

- `Decode-only throughput: 1.98660 tokens/s`

#### 7B 小结

按 `Decode-only throughput` 统计：

- dense：`2.579`
- 原始 SVD：`1.757`
- 紧凑 SVD：`1.987`

紧凑模型同样明显优于原始 SVD。

### 5.3 Android 手机端实测

这部分的目标是把 `decode_svd_test` 交叉编译到 Android 手机上，直接在手机端跑当前 SVD 模型。

#### 交叉编译

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/CMakeLists.txt`

修改原因：

- 原工程只在 `NOT ANDROID` 分支里编译 `decode_svd_test`
- 因此 Android 构建默认不会生成该可执行文件

本轮调整：

- 在 `ANDROID` 分支中新增 `decode_svd_test`
- 仅增加构建目标，不改推理逻辑

交叉编译命令：

```bash
mkdir -p /home/tianruiming/CE_ADA_LLAMA/build-android
cd /home/tianruiming/CE_ADA_LLAMA/build-android

cmake /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp \
  -DCMAKE_TOOLCHAIN_FILE=/home/tianruiming/Android/Sdk/ndk/26.3.11579264/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DGGML_OPENMP=OFF

cmake --build . -j80 --target decode_svd_test
```

生成产物：

- `/home/tianruiming/CE_ADA_LLAMA/build-android/decode_svd_test`

产物性质：

- `ELF 64-bit`
- `ARM aarch64`
- Android 动态可执行文件

#### 推送到手机

手机端目录：

- `/data/local/tmp/CE_Ada`

推送命令：

```bash
adb shell 'mkdir -p /data/local/tmp/CE_Ada'
adb push /home/tianruiming/CE_ADA_LLAMA/build-android/decode_svd_test /data/local/tmp/CE_Ada/
adb push /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf /data/local/tmp/CE_Ada/
```

手机端实测模型：

- `/data/local/tmp/CE_Ada/qwen.gguf.sort_svd.compact.gguf`

手机信息：

- 型号：`PLK110`
- ABI：`arm64-v8a`
- CPU 核数：`8`

#### 手机端运行命令

```bash
adb shell 'cd /data/local/tmp/CE_Ada && chmod +x decode_svd_test && ./decode_svd_test /data/local/tmp/CE_Ada/qwen.gguf.sort_svd.compact.gguf 8 4 0'
adb shell 'cd /data/local/tmp/CE_Ada && ./decode_svd_test /data/local/tmp/CE_Ada/qwen.gguf.sort_svd.compact.gguf 8 8 0'
```

#### 手机端结果

`threads=4`：

- `Decode-only throughput: 1.32935 tokens/s`
- `End-to-end throughput: 1.32888 tokens/s`

`threads=8`：

- `Decode-only throughput: 3.1642 tokens/s`
- `End-to-end throughput: 3.16148 tokens/s`

#### Android 小结

可以确定：

1. 当前 Android 交叉编译链路是通的
2. 当前 `decode_svd_test` 可以在手机端直接跑通
3. 当前 1.5B 紧凑 SVD 模型在这台手机上的最好结果约为 `3.16 tokens/s`
4. 在这台 8 核手机上，`8` 线程明显优于 `4` 线程

### 5.4 官方 llama.cpp 与当前仓库的控制变量对照

这部分的目标是回答一个更基础的问题：

- 先不考虑 SVD
- 先确认当前仓库里的原始 dense 路径，和官方 llama.cpp 相比是否存在异常性能损失

#### 官方仓库基线

源码：

- `/home/tianruiming/CE_ADA_LLAMA/test/llama.cpp-official`

版本：

- commit `295354e`

构建方式：

```bash
cmake -S /home/tianruiming/CE_ADA_LLAMA/test/llama.cpp-official \
  -B /home/tianruiming/CE_ADA_LLAMA/test/llama.cpp-official/build-release \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF

cmake --build /home/tianruiming/CE_ADA_LLAMA/test/llama.cpp-official/build-release -j80 --target llama-bench
```

官方 `llama-bench` 结果：

- `qwen.gguf` F16：`12.13 ± 0.01 tokens/s`
- `qwen.q4_0.gguf` Q4_0：`43.24 ± 5.83 tokens/s`

#### 当前仓库 `Release` 重编译

为了避免污染当前工作树代码，本轮没有修改源码，只新建了独立构建目录：

- `/home/tianruiming/CE_ADA_LLAMA/build-release-current`

构建方式：

```bash
cmake -S /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp \
  -B /home/tianruiming/CE_ADA_LLAMA/build-release-current \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON

cmake --build /home/tianruiming/CE_ADA_LLAMA/build-release-current -j80 --target llama-bench decode_svd_test
```

当前仓库 `Release` 结果：

- `qwen.gguf` F16：`11.32 ± 0.06 tokens/s`
- `qwen.q4_0.gguf` Q4_0：`57.47 ± 7.66 tokens/s`

#### 对照结论

可以得到两个明确结论：

1. 当前仓库的 dense `F16` 路径只有小幅损失，约为官方的 `93.3%`
2. 当前仓库的 dense `Q4_0` 路径没有“明显慢于官方”的问题

这一步很重要，因为它意味着：

- 当前仓库中，原始 dense 路径整体是正常的
- 后续主要矛盾仍然是 SVD 路径，而不是 llama.cpp 原生 dense 路径

### 5.5 当前仓库 `Release` 下的 F16 SVD 速度

在确认 dense 基线正常之后，再回到当前仓库的 SVD 链路。

测试程序：

- `/home/tianruiming/CE_ADA_LLAMA/build-release-current/decode_svd_test`

测试模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`

命令：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=./bin ./decode_svd_test /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 32 0
```

结果：

- `Decode-only throughput: 7.54228 tokens/s`
- `End-to-end throughput: 7.52806 tokens/s`

和当前仓库 `Release` 版 dense `F16` 对照：

- dense `qwen.gguf`：`11.32 ± 0.06 tokens/s`
- SVD `qwen.gguf.sort_svd.compact.gguf`：`7.54 tokens/s`

也就是说：

- 当前 `Release` 版 F16 SVD 速度大约是 dense F16 的 `66.6%`
- 当前 SVD 链路已经稳定，但还没有追平 dense `mul_mat`

### 5.6 官方 Android 手机端 FP16 基线

为了获得手机端的官方基线，本轮还额外交叉编译了官方 `llama.cpp` 的 Android `llama-bench`，并直接使用手机上现成的 FP16 模型：

- 模型：`/data/local/tmp/CE_Ada/qwen.gguf`

官方 Android 二进制目录：

- `/data/local/tmp/llama.cpp-official/bin`

手机端运行命令：

```bash
adb shell 'cd /data/local/tmp/llama.cpp-official && LD_LIBRARY_PATH=bin ./bin/llama-bench -m /data/local/tmp/CE_Ada/qwen.gguf -t 8 -ngl 0 -p 0 -n 128 -r 3 -o md'
```

结果：

- `qwen2 1.5B F16`
- `threads=8`
- `tg128 = 18.29 ± 1.10 tokens/s`

这给出了手机端的官方 dense FP16 参考值，后续可以直接拿来和手机端 `decode_svd_test` 对照。

## 6. 本轮工作的实际价值

这轮最重要的收获，不是“硬把 SVD 速度吹成已经完全追平 dense”，而是把整个问题空间收窄了。

本轮已经确认：

1. 原始 `qwen.gguf` 的对照基线是正确的
   - 它确实走 llama.cpp 原生 `mul_mat`
   - `decode_svd_test` 与 `llama-bench` 结果基本一致
   - 当前仓库 `Release` 下的 dense 路径整体正常

2. SVD 路径里此前存在的链路问题已经被修掉
   - `F16` decode 热路径缺失
   - 基准程序把打印和文本拼接算进吞吐
   - SVD 模型中冗余保留 dense FFN 权重
   - Android 构建默认不产出 `decode_svd_test`

3. 当前剩余差距已经主要收敛到 `mul_mat_svd` 内核本身
   - 不是“压根走错算子”
   - 不是“测速方法有问题”
   - 不是“模型文件里多带了一些完全无用的 dense FFN 权重”这一层的问题

## 7. 现在可以确定的结论

### 7.1 关于正确性

可以确定：

- 原始 dense 的测速链路是对的
- SVD 的测速链路现在也是对的
- Android 手机上的 SVD 实测链路也是对的
- 官方 Android 手机上的 dense `F16` 基线也已经测出

### 7.2 关于速度

可以确定：

- 当前仓库的原始 dense 路径性能基本正常
- 当前 full-rank 验证下，SVD 仍慢于 dense `mul_mat`
- `Q4_0` dense 在 CPU 上明显快于 F16 dense
- 当前 SVD 方案尚未形成“全量化 SVD”链路

更准确地说：

- 当前仓库 `Release` 下：
  - dense `qwen.gguf` F16：`11.32 ± 0.06 tokens/s`
  - SVD `qwen.gguf.sort_svd.compact.gguf`：`7.54 tokens/s`
- 官方 Android 手机端 dense `qwen.gguf` F16：`18.29 ± 1.10 tokens/s`
- 当前 Android 手机端 1.5B 紧凑 SVD：`3.16 tokens/s`

### 7.3 关于优化方向

如果继续优化，最值得投入的方向已经非常明确：

1. 继续优化 `ggml_compute_forward_mul_mat_svd` 第二段 `tmp * U`
2. 做更细粒度的内核 profile
3. 持续在紧凑模型上测试，避免无用张量干扰
4. 如果要继续推进量化 SVD，需要单独为量化 `U/V` 设计存储格式和内核

## 8. 本轮修改文件汇总

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/CMakeLists.txt`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/generate_sort_svd.py`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/strip_dense_ffn_from_svd.py`

## 9. 结论

本轮工作已经完成了几件关键事情：

1. 把 dense 基线测速链路完全核实清楚。
2. 把 SVD 路径中明显错误或浪费的实现修掉。
3. 打通了 Android 手机端的交叉编译、部署和实测链路。
4. 用官方 `295354e` 做了 PC 端和手机端的基线对照。
5. 确认当前仓库里的原始 dense 路径基本正常，后续优化重点应继续放在 `mul_mat_svd` 本身。
6. 给后续继续压 `mul_mat_svd` 内核留下了干净、可信的验证环境。

当前状态下，SVD 路径在 PC 和 Android 手机上的速度结论都是可信的，后续优化可以直接围绕算子本身展开。
