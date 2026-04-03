# SVD 协同卸载优化实验报告

## 1. 本轮目标

本轮目标是在不破坏现有按层卸载率接口的前提下，排查并优化“手机端”明显慢于电脑端的问题，并满足：

- 每层 FFN 可完整卸载到手机端
- 本机双进程调试
- 推理结果正确，不吐乱码
- 手机端使用 8 核 8 线程模拟时，decode 达到约 `9~10 token/s`

## 2. 关键结论

原实现的主要问题不是 TCP 本身，而是卸载粒度过细：

- 旧路径把一层 FFN 拆成 `up / gate / down` 三个独立 SVD RPC
- 单 token decode 时，每生成 1 个 token 需要约 `84` 次远端请求
- 客户端总等待时间基本等于服务端总计算时间，同步点过多

本轮优化后：

- decode 图中，当 `offload_rate >= 1.0` 且 `n_tokens == 1` 时，整层 FFN 改为一次远端请求
- 手机端若加载的是带原始 dense FFN 的 `.sort_svd.gguf`，则优先走 dense FFN 快路径
- 客户端请求数从约 `84/token` 降到约 `28/token`
- 1.5B 模型实测达到 `9.48 token/s`

## 3. 设计思路

### 3.1 两条执行路径

保留原有部分卸载路径：

- `offload_rate > 0.5 && offload_rate < 1.0`
- 仍走 `GGML_OP_MUL_MAT_SVD`
- 电脑端算前缀 rank，手机端算尾部 rank

新增完整 FFN 卸载路径：

- 条件：`svd_offload_enabled == true`、`offload_rate >= 0.999f`、`n_tokens == 1`
- 图构建时生成一个新的 `GGML_OP_FFN_SVD_OFFLOAD`
- 一层 FFN 只发一次 RPC
- 手机端直接返回该层 FFN 的最终输出

### 3.2 手机端快路径

手机端优先策略：

1. 如果模型文件中存在 `ffn_up / ffn_gate / ffn_down` dense 权重，则直接做 3 次 dense matvec
2. 如果 dense 权重不存在，则回退到原有 SVD 路径

这样做的原因：

- 完整卸载时，电脑端只关心最终 FFN 输出
- 手机端不必再按 `up / gate / down` 三次 RPC 暴露中间激活
- dense FFN 的单 token matvec 在 CPU 上比两段 SVD 更接近 llama.cpp 的原生快路径

## 4. 新增接口与函数

### 4.1 GGML 新算子

文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml.h`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml.c`

新增：

- `GGML_OP_FFN_SVD_OFFLOAD`
  - decode 单 token 时的整层 FFN 远端卸载算子

- `ggml_ffn_svd_offload(...)`
  - 构建一个完整 FFN 远端算子节点
  - 输入：
    - `input`
    - `up_u / up_v`
    - `gate_u / gate_v`
    - `down_u / down_v`
  - 输出：
    - 当前层 FFN 的最终输出

### 4.2 图构建层

文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp`

修改函数：

- `llm_graph_context::build_ffn_svd_qwen2(...)`
  - 当 `svd_offload_enabled && n_tokens == 1 && offload_rate >= 0.999f` 时
  - 直接构建 `ggml_ffn_svd_offload(...)`
  - 其余情况保持原来的 3 个 `mul_mat_svd` 节点

### 4.3 CPU 执行层

文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`

新增函数：

- `ggml_compute_forward_ffn_svd_offload(...)`
  - `GGML_OP_FFN_SVD_OFFLOAD` 的 CPU backend 执行入口
  - 先尝试发起完整 FFN RPC
  - RPC 失败时，本地回退到 3 次完整 SVD 计算，保证正确性

### 4.4 协同客户端

文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.h`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.c`

新增：

- `enum ggml_svd_offload_request_kind`
  - `GGML_SVD_OFFLOAD_REQ_MAT`
  - `GGML_SVD_OFFLOAD_REQ_FFN`

- `ggml_svd_offload_begin_ffn_request(...)`
  - 发送完整 FFN 请求
  - 一次请求只带：
    - `layer_id`
    - 当前层输入向量

### 4.5 手机端服务

文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp`

新增/修改：

- `compute_remote_ffn(...)`
  - 完整 FFN 远端执行入口
  - 如果 dense 权重存在，走 dense 快路径
  - 否则回退到 SVD 三段计算

- `compute_dense_matvec(...)`
  - 手机端 dense matvec 内核
  - 复用 `ggml_get_type_traits_cpu(...)` 的 `vec_dot`
  - 使用 OpenMP 按输出行并行

- 启动日志新增：
  - `mobile-side dense FFN fast path layers: X/Y`
  - 用于确认手机端模型是否具备 dense 快路径条件

### 4.6 模型张量映射修复

文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-arch.cpp`

修复：

- 为 `LLM_TENSOR_FFN_UP` 补充张量信息映射
- 这样 `QWEN2_SVD` 才能正确加载包含原始 dense FFN 的非紧凑 `.sort_svd.gguf`

## 5. 网络协议

仍然使用：

- 常驻 TCP 长连接
- 二进制定长头
- `TCP_NODELAY`

本轮协议优化点：

- 原来只支持单算子 `MAT` 请求
- 现在增加 `FFN` 请求类型
- 完整 FFN 请求只传一层输入和最终输出，避免上传/下载中间激活

这仍然是当前最适合后续迁移到手机端的网络方案：

- 支持真实跨设备 IP 通信
- 不引入 HTTP/JSON 序列化负担
- 代码分层清晰

## 6. 推荐运行方式

### 6.1 手机端进程

推荐手机端加载带 dense FFN 的非紧凑 SVD 模型：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=. ./svd_mobile_server \
  ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.gguf \
  7788 \
  8
```

说明：

- `qwen.gguf.sort_svd.gguf` 包含 dense FFN，可触发手机端 dense 快路径
- `qwen.gguf.sort_svd.compact.gguf` 仍可运行，但会回退到 SVD 手机端路径，速度更低

### 6.2 电脑端进程

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=. ./decode_svd_test \
  ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  8 \
  16 \
  0 \
  127.0.0.1:7788 \
  1.0
```

参数说明：

- `8`：生成 token 数
- `16`：电脑端 decode 线程数
- `0`：关闭逐 token 打印
- `127.0.0.1:7788`：手机端地址
- `1.0`：每层 FFN 完整卸载

## 7. 实测结果

测试日期：

- 2026-04-03

测试环境：

- 本机双进程
- 手机端：`8` 线程
- 电脑端：`16` 线程
- 客户端模型：`qwen.gguf.sort_svd.compact.gguf`
- 服务端模型：`qwen.gguf.sort_svd.gguf`

结果：

- `Decode-only throughput: 9.48224 tokens/s`
- `End-to-end throughput: 9.44728 tokens/s`

对应 profile：

- 客户端请求数：`227`（8 token，约 `28` 次请求 / token）
- 客户端远端等待：`504.042 ms`
- 服务端总计算：`716.44 ms`
- 服务端平均每请求：`2.142 ms`

生成文本：

- `, there was a young girl named Lily`

说明：

- 输出正常，无乱码
- 完整 FFN 卸载功能正确
- 满足“8 核 8 线程手机端约 9 token/s decode”目标

### 7.1 当前稳定版本补充实验：电脑端 8 核 8 线程 + 40% 背景负载

测试日期：

- 2026-04-03

测试环境：

- 当前稳定版本代码
- 本机双进程
- 电脑端：`8` 核 `8` 线程，绑核 `0-7`
- 手机端：`8` 核 `8` 线程，绑核 `20-27`
- 电脑端额外背景负载：`stress-ng --cpu 8 --cpu-load 40`
- 客户端模型：`qwen.gguf.sort_svd.compact.gguf`
- 服务端模型：`qwen.gguf.sort_svd.compact.gguf`

结果：

| 场景 | Decode-only throughput | End-to-end throughput |
| --- | --- | --- |
| 全在电脑端推理，不卸载 | `0.306123 tokens/s` | `0.306105 tokens/s` |
| 卸载 `60%` 到手机端 | `0.0889924 tokens/s` | `0.0889909 tokens/s` |
| 卸载 `80%` 到手机端 | `0.0892914 tokens/s` | `0.0892899 tokens/s` |
| 卸载 `100%` 到手机端 | `0.370758 tokens/s` | `0.370732 tokens/s` |

补充 profile：

- 纯本地：
  - `[svd-offload-local] ops=675 total=5540.501 ms local_v=1834.687 ms local_u=2446.208 ms wait_remote=1.848 ms`
- 卸载 `60%`：
  - 客户端：`requests=451`，`wait_remote=734.002 ms`
  - 服务端：`compute_total=671.458 ms`
- 卸载 `80%`：
  - 客户端：`requests=451`，`wait_remote=793.713 ms`
- 卸载 `100%`：
  - 客户端：`requests=227`，`wait_remote=1036.598 ms`
  - 客户端平均每请求：`4.593 ms`

说明：

- 所有场景都生成正常文本 `, there was a young girl named Lily`
- 在“电脑端自身已承受固定背景负载”的条件下，部分卸载 `60% / 80%` 反而显著慢于纯本地
- `100%` 完整卸载是该受限场景下最快的协同方案，但吞吐仍远低于无负载场景
- 这说明当前主要瓶颈仍是“同机双进程 + 同步等待 + 部分卸载拆分开销”，而不是数值正确性问题

## 8. 当前结论与建议

结论：

- 关键优化不是更换网络协议，而是把卸载粒度从“单个 SVD 线性层”提升到“整层 FFN”
- 在整层卸载成立时，手机端使用 dense FFN 权重比继续走 SVD 更高效

建议：

- 调度器后续若决定某层 `offload_rate == 1.0`，直接走完整 FFN 卸载
- 若是 `0.5 < offload_rate < 1.0`，继续走原有 partial-rank SVD 卸载
- 手机端部署时，优先使用保留 dense FFN 的 SVD 模型作为服务端模型
