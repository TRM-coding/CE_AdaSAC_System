# 2026-04-11 离线生成手机端专用量化 SVD 模型

## 1. 目标

本次新增一条离线流程，用于直接生成“手机端专用”的量化 SVD GGUF：

- 输入：`qwen.gguf.sort_svd.compact.gguf`
- 输出：只保留原有模型结构，但把手机端远端执行真正使用到的 SVD `U/V` 张量离线量化

这样手机端服务进程启动后：

- 不再需要首次请求时现场把 SVD tail 权重量化成缓存
- 可以直接加载量化后的 SVD 模型执行

## 2. 新增脚本

脚本：

- [quantize_mobile_svd.py](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/quantize_mobile_svd.py)

功能：

- 读取 compact SVD GGUF
- 只量化下面这些 tensor：
  - `ffn_up_svd_u`
  - `ffn_up_svd_v`
  - `ffn_gate_svd_u`
  - `ffn_gate_svd_v`
  - `ffn_down_svd_u`
  - `ffn_down_svd_v`
- 其余 tensor 原样保留
- 重写 `general.file_type`
- 生成手机端专用 GGUF

当前支持的量化类型：

- `f16`
- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`
- `q2_k`
- `q3_k`
- `q4_k`
- `q5_k`
- `q6_k`
- `iq4_nl`
- `iq4_xs`

## 3. 实际生成命令

本次实际执行的是：

```bash
python /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/quantize_mobile_svd.py \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.mobile_q8_0.gguf \
  --quant q8_0
```

脚本输出：

```text
total_tensors: 422
quantized_svd_tensors: 168
kept_original_tensors: 254
input_tensor_bytes: 6192551936
output_tensor_bytes: 2214483968
```

## 4. 生成结果

输入模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`

输出模型：

- [qwen.gguf.sort_svd.compact.mobile_q8_0.gguf](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.mobile_q8_0.gguf)

体积变化：

- 原始 compact SVD：`5.8G`
- 手机端专用 `Q8_0` SVD：`2.1G`

GGUF 自检：

- `general.file_type = 7`，即 `MOSTLY_Q8_0`
- `blk.0.ffn_down_svd_u.weight = Q8_0`
- `blk.0.ffn_down_svd_v.weight = Q8_0`
- 非 SVD tensor 例如 `blk.0.attn_norm.weight` 仍保持原类型 `F32`

## 5. 联合推理验证

验证目录：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/mobile_model_verify_20260411_144443`

日志：

- client: [client.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/mobile_model_verify_20260411_144443/client.log)
- server: [server.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/mobile_model_verify_20260411_144443/server.log)

### 服务端命令

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./svd_mobile_server \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.mobile_q8_0.gguf \
  7796 8 off
```

注意：

- 这里传的是 `off`
- 因为模型本身已经离线量化好了，不再需要运行时懒量化

### 客户端命令

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  2 8 0 127.0.0.1:7796 1.0
```

### 验证结果

客户端生成：

```text
Generated text: , there
```

服务端统计：

```text
CPU_Mapped model buffer size = 2111.90 MiB
quant_mode=off
create_total=4.854 ms
remote_compute_total=76.397 ms
```

说明：

- 离线量化模型可以直接被 `svd_mobile_server` 加载
- 协同推理输出正常，不是乱码
- 与之前“运行时现场量化”相比，启动后不再有多秒级的首次量化建缓存开销

## 6. 结论

本次已经完成：

1. 新增离线脚本，能够生成手机端专用量化 SVD 模型
2. 实际产出一份 `qwen.gguf.sort_svd.compact.mobile_q8_0.gguf`
3. 用这份模型完成了短联合推理验证
4. 确认这条方案可以替代服务端运行时懒量化

如果后续要继续落地：

- 手机端默认模型建议切到这份离线量化 GGUF
- 若要进一步压缩体积，可继续评估 `q4_k / q5_k / q6_k`
