# 2026-04-18 Windows Q4 SVD 快路径修复与 Windows + Android 协同路径检查

## 1. 背景

重新核对 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf` 后，之前提到的 Windows `30+ tok/s` 不能再默认归因于 SVD Q4 快路径。

实际问题是：

- 模型架构是 `qwen2_svd`
- 共有 `168` 个 SVD `U/V` tensor
- 这 `168` 个 SVD tensor 全是 `Q4_0`
- 但旧版 `ggml_compute_forward_mul_mat_svd_vec()` 只接受 `F32/F16` 的 `U/V`
- 因此 Windows Q4 SVD decode 会直接回落到通用 `mul_mat_svd_sub` 两段路径

这就是此前会掉到低速档的直接原因。

## 2. 本次修复

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`

修复方式：

- 不再把 SVD vec 快路径限制为 `F32/F16`
- 改为放行任何存在 `vec_dot` 的 `U/V` 类型
- 当 `b` 或中间 `tmp` 不等于对应 `vec_dot_type` 时，复用现有 `from_float` 框架做一次转换
- 对 `Q4_0` 来说：
  - `V` / `U` 的 `vec_dot_type` 是 `Q8_0`
  - 输入 `x` 先从 `F32` 转成 `Q8_0`
  - 中间 `tmp` 也从 `F32` 转成 `Q8_0`
  - 然后走 `ggml_vec_dot_q4_0_q8_0`

本次没有改动：

- `F16/F32` 原有快路径
- `ggml_vec_dot_f16_unroll` 分支
- 远端 offload 协议与 barrier 语义
- 之前已修好的 Windows Release `-O3 -DNDEBUG` 防护

## 3. 模型核对

核对模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`

确认结果：

- `general.architecture = qwen2_svd`
- `svd_uv_total = 168`
- `svd_uv_tensor_types = { Q4_0: 168 }`

## 4. 构建与产物

Linux release：

```bash
cd /home/tianruiming/CE_ADA_LLAMA
cmake -S src/llama.cpp -B build-release-current -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON
cmake --build build-release-current -j80 --target decode_svd_test
```

Windows cross build：

```bash
cd /home/tianruiming/CE_ADA_LLAMA
cmake -S src/llama.cpp -B build-windows-coop-llvm-q4fix -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Windows \
  -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_C_COMPILER=/home/tianruiming/CE_ADA_LLAMA/test/toolchains/llvm-mingw-20260407-msvcrt-ubuntu-22.04-x86_64/bin/x86_64-w64-mingw32-clang \
  -DCMAKE_CXX_COMPILER=/home/tianruiming/CE_ADA_LLAMA/test/toolchains/llvm-mingw-20260407-msvcrt-ubuntu-22.04-x86_64/bin/x86_64-w64-mingw32-clang++ \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON \
  -DGGML_AVX2=ON \
  -DGGML_FMA=ON \
  -DGGML_F16C=ON \
  -DGGML_BMI2=ON \
  -DGGML_OPENMP=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF
cmake --build build-windows-coop-llvm-q4fix --target decode_svd_test svd_mobile_server -j16
```

产物已同步到：

- `/home/tianruiming/CE_ADA_LLAMA/build-rel/windows/decode_svd_test.exe`
- `/home/tianruiming/CE_ADA_LLAMA/build-rel/windows/svd_mobile_server.exe`
- `/home/tianruiming/CE_ADA_LLAMA/build-rel/windows-x86_64.zip`

## 5. 实测结果

### 5.1 Linux 本地

命令：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
./decode_svd_test \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  24 8 0
```

结果：

- Decode-only throughput: `24.7649 tok/s`
- End-to-end throughput: `24.5214 tok/s`
- `[svd-local-op-profile] up_ops=700 gate_ops=700 down_ops=700`

对照：

- 同机 `F16 compact`：`8.27558 tok/s`

### 5.2 Windows 真机

机器：

- CPU: `AMD Ryzen 7 6800H with Radeon Graphics`
- Cores / Threads: `8 / 16`

命令：

```powershell
cd J:\build-rel\windows
.\decode_svd_test.exe J:\build-rel\models\qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf 24 8 0
```

结果：

- Decode-only throughput: `28.1911 tok/s`
- End-to-end throughput: `28.1198 tok/s`
- `[svd-local-op-profile] up_ops=700 gate_ops=700 down_ops=700`
- `[svd-local-stage-profile] ffn_svd_total=632.657 ms`

结论：

- Windows Q4 SVD 已不再是 `3 tok/s` 级别的慢路径
- 当前结果已接近此前预期的 `30 tok/s`

补充：

- Windows 上尝试加载 `qwen.gguf.sort_svd.compact.gguf` 做 `F16` 对照时，本机因 `~6.19 GiB` CPU buffer 分配失败而无法完成加载，因此没有同机 F16 数值

### 5.3 Android 真机本地 SVD 对照

手机：

- OnePlus 15
- 8 线程
- 当次读取温度约 `30.3 C`
- governor 读取结果为 `performance`

命令 1：

```bash
adb shell 'cd /data/local/tmp/CE_Ada && LD_LIBRARY_PATH=/data/local/tmp/CE_Ada taskset -a ff ./decode_svd_test /data/local/tmp/CE_Ada/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf 128 8 0'
```

结果：

- Decode-only throughput: `48.3202 tok/s`
- End-to-end throughput: `47.7389 tok/s`

命令 2：

```bash
adb shell 'cd /data/local/tmp/CE_Ada && LD_LIBRARY_PATH=/data/local/tmp/CE_Ada taskset -a ff ./decode_svd_test /data/local/tmp/CE_Ada/qwen.gguf.sort_svd.compact.mobile_q4_0.gguf 128 8 0'
```

结果：

- Decode-only throughput: `34.3708 tok/s`
- End-to-end throughput: `34.0746 tok/s`

结论：

- `mobile_q4_0` 相比 `llama_quant_q4_0` 明显更慢
- 当前不应再把 `mobile_q4_0` 当作默认推荐模型
- Android 本地验证与 Windows + Android 协同默认口径都应优先使用 `llama_quant_q4_0.gguf`

## 6. Windows + Android 协同路径是否能命中这个快推理

结论先行：

- 能，但有条件
- 只有 `0 < offload_rate < 1` 的 partial-rank 协同，Windows 本地保留 rank 才会直接命中这次修复的 Q4 SVD vec 快路径
- `offload_rate >= 0.999` 的整层 FFN 全远端模式不会在 Windows 本地执行这一段快路径
- Android 端若也想吃到同类 Q4 快路径，必须使用包含同一 `ggml-cpu.c` 补丁重编后的 `svd_mobile_server`

### 6.1 Windows 客户端本地保留 rank

调用链：

1. `decode_svd_test` 在传入 `offload host:port` 且提供 `offload_rates` 时启用协同
2. `llama-model.cpp` 将每层 `offload_rate` 传给 `build_ffn_svd_qwen2`
3. `ggml_compute_forward_mul_mat_svd_vec()` 内部根据

```c
k_keep = total_rank - ceilf(offload_rate * total_rank)
```

决定本地保留 rank

4. 当 `k_keep > 0` 时，本地部分仍然在 `ggml_compute_forward_mul_mat_svd_vec()` 内完成
5. 因为该函数现在已支持 `Q4_0 U/V + Q8_0 vec_dot`，所以 Windows 客户端本地 rank 会命中快路径

因此：

- `llama_quant_q4_0.gguf` 作为 Windows 客户端模型
- `offload_rate` 设成 `0.2 / 0.5 / 0.8` 这类 partial-rank

时，Windows 本地 share 可以命中本次快路径。

### 6.2 `offload_rate = 1.0` 的情况

`ggml_compute_forward_ffn_svd_offload()` 在：

- `svd_offload_rate >= 0.999f`
- 输入/输出满足向量条件

时会优先发起整层 FFN 远端请求；请求成功则直接返回，不再做本地 SVD matmul。

因此：

- `offload_rate = 1.0` 时，Windows 本地不会执行本次 Q4 SVD vec 快路径
- 这时能否快，取决于 Android 远端服务本身

### 6.3 Android 远端服务

Android 端 `svd_mobile_server` 最终也链接同一套 `ggml-cpu.c`。

所以如果：

- Android 端是用本次补丁之后的代码重编
- 且服务端模型使用 `Q4_0` SVD tensor

则 Android 端自己的本地 SVD 计算同样可以命中这条 Q4 vec 快路径。

对应模型优先建议直接使用：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`

不再默认建议：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.mobile_q4_0.gguf`

原因是当前真机数据表明 `mobile_q4_0` 会明显拉低吞吐。

反过来，如果 Android 端还在使用旧二进制，即使 Windows 客户端已经更新，远端仍会保持旧行为。

## 7. 当前建议

如果目标是 Windows + ARM 手机协同也尽量命中快路径，建议当前口径如下：

1. Windows 客户端使用新构建的 `decode_svd_test.exe`
2. Windows 客户端模型使用 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
3. Android 端重新全量重编并重新推送 `svd_mobile_server`
4. Android 端默认使用 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
5. 协同验证先用 `0 < offload_rate < 1`，不要直接上 `1.0`
6. 验证时同时看：
   - Windows 端 `[svd-local-op-profile]` 是否非零
   - Android 服务端是否为补丁后新二进制
   - 协同总吞吐是否明显高于旧版低速路径

只有在专门研究手机专用裁剪模型时，才额外保留 `mobile_q4_0` 作为对照项，而不是默认部署项。

若后续需要进一步做真机协同实测，下一步应直接记录：

- Windows 客户端 CPU / 线程数
- Android 端机型 / governor / 温度
- `offload_rate`
- 客户端与服务端模型文件名
- 双端 profile 行
