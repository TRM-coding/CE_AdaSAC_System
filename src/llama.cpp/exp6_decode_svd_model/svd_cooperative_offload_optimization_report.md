# SVD 协同卸载优化实验报告

> 更新说明（2026-04-03 晚）  
> 本文前半部分记录的是一轮“整层 FFN 卸载 / dense 快路径”优化设想与阶段性结果。  
> 在当前干净代码版本的实际回归中，最终稳定落地并验证正确性的路径不是整层 `FFN` 远端执行，而是：
> - 保留 `GGML_OP_MUL_MAT_SVD` 粒度
> - 新增 `up + gate` 合并请求
> - 手机端改为直接复用 ggml 内置 `mul_mat` 两段小图执行尾部 rank
>
> 因此，文中涉及 `GGML_OP_FFN_SVD_OFFLOAD`、手机端 dense FFN 快路径、`9.48 tok/s` 的部分，应视为历史实验记录，不代表当前稳定版本结论。当前稳定版本结论见文末新增的“9. 当前稳定版本回归结果”。

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

- 当时实现为 `offload_rate > 0.5 && offload_rate < 1.0`
- 仍走 `GGML_OP_MUL_MAT_SVD`
- 电脑端算前缀 rank，手机端算尾部 rank

补充说明：

- 上面这个阈值描述对应 2026-04-03 的实现
- 截至 2026-04-07，当前代码已放宽为 `offload_rate > 0.0 && offload_rate < 1.0`

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
- 若是 `0.0 < offload_rate < 1.0`，继续走 partial-rank SVD 卸载
- 手机端部署时，优先使用保留 dense FFN 的 SVD 模型作为服务端模型

## 9. 当前稳定版本回归结果

### 9.1 回归问题定位

在对“当前干净版本”继续修改并复现实验时，出现了两类问题：

1. 数值错误
   - 现象：协同 `75% / 100%` 时生成文本退化成 `",,,,,,,,"` 或乱码
   - 初始怀疑：`up + gate` 合并请求后的 gate 缓存命中错误
   - 实际根因：
     - 原缓存确实存在“按输入指针命中”的潜在风险，因此已改成“输入内容哈希”命中
     - 但真正导致输出错误的直接原因，是手机端那版 `vec_dot` 快路径在当前张量类型下会反复走到 `missing float conversion for SVD tensor type`
     - 远端请求失败后，客户端会把 `remote_out` 清零，最终导致模型输出退化

2. 服务端分配错误
   - 现象：重启新服务端后出现
     - `GGML_ASSERT(ggml_get_no_alloc(ctx) == true) failed`
   - 根因：服务端临时 ggml context 使用了 `ggml_backend_alloc_ctx_tensors(...)`，但 `ggml_init_params.no_alloc` 配错成了 `false`

### 9.2 当前稳定修复

当前稳定版本采用的修复方案如下：

- 客户端保留 `up + gate` 合并请求
  - `up` 发一次远端请求
  - `gate` 直接复用同输入的缓存结果
  - 缓存键改为：
    - `layer_id`
    - `rank_start`
    - `input_hash`

- 手机端不再直接使用那条会失败的 `vec_dot` 类型特征路径
  - 改为构造两段 ggml 小图：
    - `tmp = mul_mat(v_tail, input)`
    - `out = mul_mat(u_tail, tmp)`
  - 即直接复用 ggml 内置 `mul_mat`

- 服务端分配方式修正
  - `ggml_init_params.no_alloc = true`
  - 再调用 `ggml_backend_alloc_ctx_tensors(...)`

### 9.3 本次实际代码变更

涉及文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.h`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.c`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp`

新增/修正点：

- 协议版本升级到 `v2`
- 请求类型支持：
  - `GGML_SVD_OFFLOAD_REQ_MAT`
  - `GGML_SVD_OFFLOAD_REQ_UP_GATE`
- `up + gate` 合并返回双输出
- gate 缓存从“输入地址”改成“输入内容哈希”
- 手机端尾部 rank 计算改为 ggml 内置 `mul_mat`

### 9.4 本次绑核复现实验

测试日期：

- 2026-04-03

测试约束：

- 使用隔离 cgroup：`/sys/fs/cgroup/tianruiming-exclusive`
- 手机端固定绑核：`60-67`
- 电脑端固定绑核：`68-75`
- 两端都使用 `8` 核 `8` 线程
- 所有实验顺序执行，避免核心争用

服务端启动方式：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
sudo bash -lc '
  echo $$ > /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs
  exec taskset -c 60-67 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
    ./svd_mobile_server ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 7788 8
'
```

客户端启动方式：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
taskset -c 68-75 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0 127.0.0.1:7788 1.0
```

结果：

| 场景 | Decode-only throughput | End-to-end throughput | 正确性 |
| --- | --- | --- | --- |
| 单机基线 | `9.71403 tok/s` | `9.69112 tok/s` | 正确 |
| 协同卸载 `75%` | `9.42101 tok/s` | `9.40045 tok/s` | 正确 |
| 协同卸载 `100%` | `6.52324 tok/s` | `6.51379 tok/s` | 正确 |

三组生成文本一致：

- `Generated text: , there was a young girl named Lily`

### 9.5 当前稳定结论

当前稳定版本已经解决“模型计算错误”问题，但还没有解决吞吐提升问题。

结论：

- 正确性已恢复
- `up + gate` 合并协议可以稳定工作
- 手机端“直接复用 ggml 内置 `mul_mat` 两段小图”是可靠的
- 但这条路径每次请求都要新建小图与分配 backend 资源，固定开销很重
- 因此在当前实现下：
  - `75%` 卸载仅接近单机
  - `100%` 卸载明显慢于单机

下一步建议：

1. 把手机端这条 ggml `mul_mat` 执行路径做成常驻执行器，避免每次请求都新建 context / backend / buffer
2. 在保证正确性的前提下，继续保留 `up + gate` 合并请求
3. 不要把“已修复正确性”和“已实现提速”混为一谈，当前只完成了前者

### 9.6 补充实验结论

在当前稳定版本基础上，又补做了两组实验，用来判断“协同卸载是否在电脑端算力受限时更有价值”。

1. 电脑端 `8` 核固定、额外施加 `20%` 背景负载
   - 纯本地：`1.90802 tok/s`
   - 卸载 `70%`：`2.13020 tok/s`
   - 卸载 `80%`：`2.15228 tok/s`
   - 卸载 `100%`：`1.71288 tok/s`
   - 结论：在电脑端被额外占用时，`70% / 80%` 卸载优于纯本地，但 `100%` 仍然更慢

2. 电脑端无背景负载，仅限制可用核心数为 `2 / 4 / 6 / 8`
   - `2` 核时：
     - 本地不卸载：`3.84163 tok/s`
     - 全部卸载：`5.71402 tok/s`
     - 结论：电脑端极弱时，协同明显受益
   - `4` 核时：
     - 本地不卸载：`6.90966 tok/s`
     - 卸载 `70%`：`6.48435 tok/s`
     - 全部卸载：`6.40484 tok/s`
     - 结论：已经接近本地上限，协同收益不稳定
   - `6 / 8` 核时：
     - 本地分别为 `9.09884 / 9.86733 tok/s`
     - 全部卸载分别为 `5.97503 / 6.03659 tok/s`
     - 结论：电脑端本地算力一旦充足，当前协同实现会明显亏损

需要单独强调：

- 本节实验对应的当时代码只在 `offload_rate > 0.5` 时才会触发真实协同
- 因此这里的 `50%` 档位实际上不会真正卸载到手机端
- 截至 2026-04-07，这个阈值已经修成 `offload_rate > 0.0`

综合来看，当前稳定实现的真实结论是：

- 正确性已经恢复
- `up + gate` 合并请求和 ggml `mul_mat` 服务端路径可以稳定工作
- 但收益高度依赖电脑端算力是否受限
- 只有在电脑端核心很少，或者电脑端被额外占用时，协同卸载才更容易体现价值
- 后续优化重点仍然是削减手机端每请求固定开销；`offload_rate > 0.5` 这类粗阈值限制已于 2026-04-07 去掉
