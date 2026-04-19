# Exp10: Q4 SVD 本地截断修复与单机加速实现

## 1. 目标

本次修改的目标有两个：

1. 修复 `Q4 SVD` 模型在单机本地 `0%` 截断时输出错误的问题。
2. 让单机本地 `SVD truncation` 真正减少计算量，从而带来 decode 提速。

实验对象为：

- 模型：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 可执行程序：`build-release-current/decode_svd_test`
- CPU 资源：`tianruiming-exclusive` cgroup 的 `60-79` 核
- 线程数：`8`
- decode 长度：`8 tokens`

## 2. 问题定位

### 2.1 本地截断没有真正生效

原先 `GGML_OP_MUL_MAT_SVD` 的本地 decode 快路径里，虽然图构建阶段已经把 `k_trunc` 写进了 op，但真正执行时，本地非协同路径仍然按满秩 `total_rank` 在算。

核心问题在：

- 文件：[ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)
- 原逻辑：

```c
const int64_t k_local = *request_started ? k_keep : total_rank;
```

对于单机本地截断：

- `can_offload = false`
- `request_started = 0`

因此这里总是落回 `total_rank`，导致：

- 本地 `50% / 70% / 80%` 截断并没有真正减少 rank
- 理论上应该减少的计算量没有减少

### 2.2 Q4 SVD 在 0% 时输出错误

`Q4 SVD` 的 `U/V` 因子被模型加载阶段放进了 `CPU_AARCH64` 的 extra buffer 中，并被 runtime repack 成 `q4_0_8x8`。

但本地 SVD decode 的手写 vec 快路径直接按原始 `GGML Q4_0` block 布局读取 `u->data / v->data`，没有适配 repack 后的布局，结果就是：

- `0%` 不截断时也会算错
- 典型症状是 top logits 退化成全零附近
- 输出退化成连续 `!`

这说明当时的问题不是“截断让输出坏掉”，而是 `Q4 SVD local fast path` 本身就不正确。

## 3. 实现方式

### 3.1 修复本地截断 rank 选择

修改文件：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)

核心改动：

```c
const int64_t k_local = can_offload
    ? (*request_started ? k_keep : total_rank)
    : k_keep;
```

这样语义变成：

- 协同 offload：保持原先逻辑
- 单机本地 truncation：直接按 `k_keep` 执行

同样的修复做了两处：

1. `ggml_compute_forward_mul_mat_svd_vec()`
2. `ggml_compute_forward_mul_mat_svd()`

### 3.2 修复 workspace 大小

原先临时工作区仍按满秩大小分配。现在改成按实际参与的 rank `rank_work` 分配，避免本地截断后继续按满秩准备中间缓冲区。

核心改动：

```c
const int64_t rank_work = k_keep > 0 && k_keep < v->ne[1] ? k_keep : v->ne[1];
```

随后 `tmp_rhs_elems` 和 `F32 tmp buffer` 都改为基于 `rank_work` 计算。

### 3.3 修复 Q4 SVD 因子布局问题

修改文件：

- [llama-model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp)

关键思路：

- `GGML_OP_MUL_MAT_SVD` 不再允许 `U/V` 因子进入 `CPU_AARCH64` repack buffer
- 保持 `SVD factors` 使用 canonical GGML layout
- 让本地 SVD vec 快路径读取到的内存布局和它的假设一致

具体做法是在 `weight_buft_supported()` 中，针对 `GGML_OP_MUL_MAT_SVD` 不再把它按普通 `GGML_OP_MUL_MAT` 的 extra-buffer 能力去探测，而是保留 op 类型：

```c
op_tensor = ggml_mul_mat(ctx, w, b);
op_tensor->op = GGML_OP_MUL_MAT_SVD;
```

这样 `CPU_AARCH64` 的 extra buffer backend 不会再声称自己支持 `MUL_MAT_SVD` 权重布局，SVD 因子就不会被 repack。

### 3.4 保留对 repacked quantized tensor 的保险回退

在 `ggml_compute_forward_mul_mat_svd_vec()` 中还保留了一层保险：

```c
if ((u->extra != NULL || v->extra != NULL) &&
    (ggml_is_quantized(u->type) || ggml_is_quantized(v->type))) {
    return false;
}
```

如果未来仍有 quantized SVD tensor 被放进 extra buffer，这条逻辑会拒绝继续走手写 vec 路径，避免再次出现“按错误布局读数据”的问题。

## 4. 结果验证

### 4.1 全层截断实验

同机同口径重跑结果如下：

| 截断率 | 生成文本 | Decode-only throughput |
|---|---|---:|
| `0%` | `, there was a little girl named Sally` | `30.9036 tok/s` |
| `20%` | `!!!!!!!!` | `34.0824 tok/s` |
| `50%` | `,  ro ro ro ro ro ro` | `37.3070 tok/s` |
| `70%` | ` once once once once once once once once` | `38.2938 tok/s` |
| `80%` | ` fo fo fo fo fo fo fo fo` | `42.1508 tok/s` |

结论：

- `0%` 基线输出已经恢复正常
- 本地 truncation 已经明确带来速度提升
- 截断越高，短 decode 下速度越高
- 但文本退化也更明显

### 4.2 稀疏分层截断实验

额外做了只截断少数层的实验。

#### 3 层不连续，`20%`

- 层位：`0 / 10 / 20`
- 文本：` reply to the question. 1.`
- 速度：`31.7865 tok/s`

#### 3 层不连续，`50%`

- 层位：`0 / 10 / 20`
- 文本：`, I was a bit of a nerd`
- 速度：`32.3549 tok/s`

#### 5 层不连续，`50%`

- 层位：`0 / 6 / 12 / 18 / 24`
- 文本：`, there was a young man named George`
- 速度：`32.3911 tok/s`

结论：

- 只截少数不连续层也能带来一定提速
- sparse truncation 的速度提升明显小于“全层同时截断”
- 但文本质量通常比“全层大比例截断”更稳定

## 5. 代码落点

本次核心代码修改落在以下两个文件：

- [llama-model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp)
- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)

对应目的分别是：

- `llama-model.cpp`：阻止 `Q4 SVD` 因子被 repack，修复 `0%` 正确性
- `ggml-cpu.c`：让本地 truncation 真正按 `k_keep` 计算，并按截断 rank 分配工作区

## 6. 当前结论

这次实现已经完成了最关键的一步：

- 单机本地 `Q4 SVD` 在 `0%` 时恢复正确输出
- 单机本地 `SVD truncation` 可以真实减少计算量
- decode 速度会随截断率提升而上升

但这次实验也说明：

- “速度更快”与“输出更稳”之间存在明显 tradeoff
- 如果只追求速度，全层高比例截断最有效
- 如果要兼顾输出质量，少量分层截断更值得继续搜索

## 7. 结果日志

本轮主要结果目录：

- [q4_llama_quant_trunc_rerun_20260419_1600](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600)

关键日志：

- [rate_0.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_0.log)
- [rate_20.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_20.log)
- [rate_50.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_50.log)
- [rate_70.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_70.log)
- [rate_80.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_80.log)
- [rate_sparse3_20.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_sparse3_20.log)
- [rate_sparse3_50.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_sparse3_50.log)
- [rate_sparse5_50.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/q4_llama_quant_trunc_rerun_20260419_1600/rate_sparse5_50.log)
