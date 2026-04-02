# SVD 接口与调用链文档

## 1. 目的

这份文档说明当前工程里和 `mul_mat_svd` 相关的新增接口，重点回答两个问题：

1. 新增的 SVD 相关函数分别做什么
2. 从最顶层测试程序到最底层 CPU 算子的调用链是怎样串起来的

本文只描述当前仓库里的实际实现，不讨论理论上的 SVD 推导。

## 2. 新增接口总览

### 2.1 图层和算子层新增的核心接口

#### `ggml_mul_mat_svd`

位置：

- [ggml.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml.c#L2779)
- [ggml.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml.h#L1130)

作用：

- 在 GGML 计算图里创建一个 `GGML_OP_MUL_MAT_SVD` 节点
- 表达的计算含义是：`x * V * U`
- 这里不直接执行计算，只是生成图节点和保存输入张量引用

输入：

- `w`：原始 dense 权重对应的形状参考张量
- `w_v`：SVD 后的 `V`
- `w_u`：SVD 后的 `U`
- `b`：输入激活 `x`
- `k_trunk`：截断参数，当前语义是“截掉后面多少个 rank”

输出：

- 一个 `GGML_TYPE_F32` 的结果张量，`op = GGML_OP_MUL_MAT_SVD`

说明：

- 这个接口只是把 SVD matmul 挂到图里
- 实际执行发生在 GGML CPU backend 的 `ggml_compute_forward_mul_mat_svd`

#### `llm_graph_context::build_mm_svd`

位置：

- [llama-graph.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp#L663)

作用：

- 在 llama.cpp 的图构建层封装一次 SVD 矩阵乘法
- 负责把模型层里的 `w / U / V / cur / rank` 组装成 `ggml_mul_mat_svd(...)`

输入：

- `w`：原始 dense 权重，主要用于输出 shape 参考
- `w_svd_u`：SVD `U`
- `w_svd_v`：SVD `V`
- `cur`：当前层输入
- `rank`：图构建时传下来的截断控制参数

输出：

- 一个 GGML 图节点，通常就是 `GGML_OP_MUL_MAT_SVD`

额外逻辑：

- 当 `w == nullptr` 时，允许从 `U/V` 推断输出 shape
- 这是为了支持紧凑 SVD 模型，即删除原始 dense FFN 张量后仍能正常建图

#### `llm_graph_context::build_ffn_svd_qwen2`

位置：

- [llama-graph.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp#L762)

作用：

- 构建 Qwen2 SVD 版本 FFN 子图
- 把原本 dense FFN 的三段线性层 `up / gate / down` 替换成 SVD 版本

内部流程：

1. `tmp = build_mm_svd(up, up_svd_u, up_svd_v, cur, up_rank)`
2. `cur = build_mm_svd(gate, gate_svd_u, gate_svd_v, cur, gate_rank)`
3. `cur = silu(cur)`
4. `cur = cur * tmp`
5. `cur = build_mm_svd(down, down_svd_u, down_svd_v, cur, down_rank)`

说明：

- 这里保持了原始 FFN 结构
- 只是把线性层从 dense `mul_mat` 换成了 `mul_mat_svd`

#### `llm_build_qwen2_svd`

位置：

- [llama-model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp#L6409)

作用：

- 构建 `QWEN2_SVD` 架构的整张推理图
- 在 transformer block 里，attention 仍然走普通路径，FFN 走 `build_ffn_svd_qwen2`

说明：

- 这是 SVD 模型真正接入 llama.cpp 主图构建流程的入口

#### `GGML_OP_MUL_MAT_SVD`

位置：

- [ggml.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml.h#L460)

作用：

- 新增 GGML 算子类型
- 用来标识“这是一个 SVD 形式的矩阵乘法”

### 2.2 CPU backend 中的辅助函数

#### `ggml_mul_mat_svd_make_dst`

位置：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L1273)

作用：

- 基于一个模板张量，构造临时 `view`
- 主要用于给两阶段计算中的中间张量 `tmp` 和裁剪后的 `u/v` 建立视图

典型用途：

- 第一段 `tmp = x * V`
- 第二段 `dst = tmp * U`

#### `ggml_mul_mat_svd_get_k_keep`

位置：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L1302)

作用：

- 根据 `dst->svd_k_trunk` 计算这次真正参与计算的 rank 数 `k_keep`

当前语义：

- `k_trunk > 0 && k_trunk < total_rank` 时，`k_keep = total_rank - k_trunk`
- 否则 `k_keep = total_rank`

说明：

- 当前实现支持截断
- 但语义是“截掉后面的多少个 rank”，不是“直接传保留前 r 个 rank”

#### `ggml_compute_forward_mul_mat_svd_vec`

位置：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L1311)

作用：

- 单 token decode 热路径的专用快路径
- 适用于当前 SVD 模型常见的 `U/V = F16 or F32` 情况

前提条件：

- 输入 `b` 为连续的 `F32`
- 当前 batch 退化成单 token 向量乘
- `u/v/dst` 都是 contiguous
- `u/v` 类型为 `F16` 或 `F32`

内部两段：

1. 先算 `tmp = x * V`
2. 再算 `dst = tmp * U`

优化点：

- 避免走两次完整通用 `ggml_compute_forward_mul_mat`
- `F16` 分支复用 `ggml_vec_dot_f16_unroll`

#### `ggml_compute_forward_mul_mat_svd`

位置：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L1426)

作用：

- `GGML_OP_MUL_MAT_SVD` 在 CPU backend 上的正式执行入口

执行逻辑：

1. 先尝试走 `ggml_compute_forward_mul_mat_svd_vec`
2. 如果不满足快路径条件，则走通用两阶段路径
3. 第一阶段用 `V` 和 `x` 计算中间结果 `tmp`
4. 第二阶段用 `U` 和 `tmp` 计算最终输出 `dst`

说明：

- 这里严格保持 `x * V * U` 的计算顺序
- 没有把 `U * V` 预先合并

### 2.3 调度分发接口

#### `ggml_compute_forward` 中的 `GGML_OP_MUL_MAT_SVD` 分发

位置：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L2076)

作用：

- 当执行器遇到 `GGML_OP_MUL_MAT_SVD` 节点时，分发到 `ggml_compute_forward_mul_mat_svd`

## 3. 从顶层到算子的完整调用链

下面用当前的测试程序 `decode_svd_test` 说明一次典型的调用链。

### 3.1 测试程序入口

位置：

- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp)

顶层流程：

1. `llama_model_load_from_file(...)` 加载模型
2. `llama_init_from_model(...)` 创建 context
3. 构造 prompt batch
4. 调用 `llama_decode(ctx, batch)`
5. 进入 llama.cpp 正式推理链路

关键调用点：

- 首次 prefill：[decode_svd_model.cpp:126](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp#L126)
- decode 循环：[decode_svd_model.cpp:245](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp#L245)

### 3.2 `llama_decode` 进入模型图构建与执行

`llama_decode(...)` 是 llama.cpp 的统一推理入口。

对当前问题，最关键的是两件事：

1. 它会根据模型架构选择图构建器
2. 它会执行图中的每个节点

### 3.3 模型架构选择：`LLM_ARCH_QWEN2_SVD`

位置：

- [llama-model.cpp:13071](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp#L13071)

当模型架构为 `LLM_ARCH_QWEN2_SVD` 时，代码会选择：

- `llm_build_qwen2_svd`

也就是：

- 原始 `qwen.gguf` 会走 `llm_build_qwen2`
- SVD 模型 `qwen.gguf.sort_svd.gguf` 会走 `llm_build_qwen2_svd`

这一步决定了后面 FFN 是走普通 `mul_mat` 还是 `mul_mat_svd`

### 3.4 `llm_build_qwen2_svd` 构建整层 transformer

位置：

- [llama-model.cpp:6409](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp#L6409)

在每一层 block 里：

1. attention 部分仍然走普通 `build_lora_mm` / `ggml_mul_mat`
2. FFN 部分转交给 `build_ffn_svd_qwen2`

所以当前 SVD 改造主要集中在 FFN，不影响 attention 主链路。

### 3.5 `build_ffn_svd_qwen2` 把 FFN 三段线性层替换成 SVD 版本

位置：

- [llama-graph.cpp:762](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp#L762)

这一层是最重要的图构建节点封装：

1. `up` 线性层走 `build_mm_svd`
2. `gate` 线性层走 `build_mm_svd`
3. 激活后逐元素乘
4. `down` 线性层走 `build_mm_svd`

因此一次 FFN 前向里，最多会出现三次 `GGML_OP_MUL_MAT_SVD`

### 3.6 `build_mm_svd` 生成 GGML SVD 节点

位置：

- [llama-graph.cpp:663](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp#L663)

这里调用：

```cpp
res = ggml_mul_mat_svd(ctx0, w_shape, w_svd_v, w_svd_u, cur, rank);
```

这一步会把：

- shape 参考
- `V`
- `U`
- 当前输入 `cur`
- rank 截断参数

一起封装进 GGML 图节点。

### 3.7 `ggml_mul_mat_svd` 在图中创建 `GGML_OP_MUL_MAT_SVD`

位置：

- [ggml.c:2779](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml.c#L2779)

它主要做三件事：

1. 检查张量 shape 是否可乘
2. 创建结果张量
3. 设置：
   - `result->op = GGML_OP_MUL_MAT_SVD`
   - `result->src[0] = w`
   - `result->src[1] = b`
   - `result->src[2] = w_u`
   - `result->src[3] = w_v`
   - `result->svd_k_trunk = k_trunk`

这一步之后，SVD matmul 已经正式进入 GGML 图。

### 3.8 CPU backend 分发到 `ggml_compute_forward_mul_mat_svd`

位置：

- [ggml-cpu.c:2076](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L2076)

执行器遇到 `GGML_OP_MUL_MAT_SVD` 时，会走：

```c
ggml_compute_forward_mul_mat_svd(params, tensor);
```

这一步标志着“图节点”进入“实际算子执行”。

### 3.9 `ggml_compute_forward_mul_mat_svd` 执行两阶段乘法

位置：

- [ggml-cpu.c:1426](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L1426)

执行顺序：

1. 根据 `svd_k_trunk` 计算 `k_keep`
2. 创建裁剪后的 `V` 视图
3. 第一阶段计算 `tmp = x * V`
4. 创建裁剪后的 `U` 视图
5. 第二阶段计算 `dst = tmp * U`

这就是当前算子真正执行的数学路径。

### 3.10 单 token decode 场景优先走 `ggml_compute_forward_mul_mat_svd_vec`

位置：

- [ggml-cpu.c:1311](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c#L1311)

如果满足 decode 热路径条件，执行顺序是：

1. 每个线程算一段 `tmp`
2. barrier
3. 每个线程再算一段 `dst`

这一层就是当前 PC 和手机端 decode 性能优化的核心。

## 4. 当前 rank 参数的真实语义

当前接口经常容易被误解，所以单独说明。

### 4.1 现在支持什么

当前 `mul_mat_svd` 已经支持“只计算部分秩”。

具体做法是：

- 通过 `svd_k_trunk` 控制保留多少个 rank
- 实际参与计算的是前 `k_keep` 个秩分量

### 4.2 现在的参数含义

当前不是“直接传保留前 `r` 个秩”，而是：

- 传入 `k_trunk`
- 内部换算成 `k_keep = total_rank - k_trunk`

也就是：

- `k_trunk = 0`：不截断，满秩
- `k_trunk = total_rank - r`：等价于保留前 `r` 个秩

### 4.3 当前代码里是否实际启用了截断

当前图构建代码里，大部分地方默认还是按满秩在跑，原因是 `rank` 被设置成 `0` 时，当前语义等价于“不截断”：

- [llama-model.cpp:6496](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp#L6496)

所以要区分两件事：

1. 算子能力层面：已经支持部分秩计算
2. 当前模型图实际配置：默认大多仍在跑满秩

## 5. 最短调用链总结

如果只保留最核心的一条链，可以概括成：

1. `decode_svd_test` 调 `llama_decode`
2. `llama_decode` 根据模型架构选择 `llm_build_qwen2_svd`
3. `llm_build_qwen2_svd` 在 FFN 中调用 `build_ffn_svd_qwen2`
4. `build_ffn_svd_qwen2` 对 `up/gate/down` 调 `build_mm_svd`
5. `build_mm_svd` 调 `ggml_mul_mat_svd`
6. `ggml_mul_mat_svd` 生成 `GGML_OP_MUL_MAT_SVD`
7. GGML CPU backend 分发到 `ggml_compute_forward_mul_mat_svd`
8. `ggml_compute_forward_mul_mat_svd` 最终执行 `x * V * U`

## 6. 相关文件

- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp)
- [llama-model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp)
- [llama-graph.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp)
- [ggml.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml.h)
- [ggml.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml.c)
- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)
