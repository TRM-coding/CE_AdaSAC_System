# SVD 协同卸载实验报告

> 更新说明（2026-04-03 晚）  
> 本文主体记录的是最初的单算子协同卸载实现与早期联调结果。  
> 在后续继续修改时，出现过一次“吞吐上去了但模型输出错误”的回归。当前稳定版本已经修复该问题；最新回归结论、绑核方式和实验数据见文末新增的“9. 当前稳定版本回归补充”。

## 1. 目标

本轮实现目标是在当前 `QWEN2_SVD` 推理链路上加入“电脑端 + 手机端”协同计算能力：

- 建图时保留“按层卸载率”的传入接口
- 当某层卸载率大于 `50%` 时，在单 token decode 热路径触发 SVD 尾部 rank 卸载
- 电脑端本地计算前缀 rank，手机端计算尾部 rank，最终在电脑端合并结果
- 当前调试方式为同一台电脑上启动两个进程：
  - `decode_svd_test` 充当电脑端
  - `svd_mobile_server` 充当手机端

## 2. 设计思路

### 2.1 分层原则

本轮实现按三层拆分：

1. 调用接口层
   - 在 `llama_context_params` 中加入协同卸载配置
   - 由上层直接传入每层卸载率、远端地址、端口和超时

2. 图构建层
   - `QWEN2_SVD` 建图时不再硬编码 rank
   - 每个 `mul_mat_svd` 节点携带：
     - 层号
     - 算子类型（`up / gate / down`）
     - 当前层卸载率

3. 算子执行层
   - 仅在单 token decode 向量快路径执行真实协同拆分
   - prefill 和普通通用 matmul 路径仍保持本地全量计算，避免错误截断

### 2.2 为什么只在 decode 热路径做真实卸载

原因有两个：

1. prefill 输入是多 token batch，网络传输开销大，且当前任务重点是单 token 推理吞吐
2. 当前 `mul_mat_svd` 已有成熟的单 token 向量快路径，最适合在这里插入“本地前缀 + 远端尾部 + 合并”的逻辑

### 2.3 网络协议选择

本轮使用常驻 TCP 长连接 + 二进制定长头 + 原始 `float` payload：

- 不使用 HTTP / JSON
- 打开 `TCP_NODELAY`
- 每次请求只发送：
  - 层号
  - 算子号
  - 卸载率
  - 分界 rank
  - 当前输入向量
- 手机端预加载模型，不重复加载权重

这是当前代码中最适合继续迁移到手机端的做法，协议简单，额外序列化开销小。

### 2.4 当前协同执行语义

这里单独强调当前实现的真实粒度，避免把它理解成“多层整段卸载”：

1. 当前请求粒度是单个 `GGML_OP_MUL_MAT_SVD` 节点
   - 也就是 FFN 中的 `up / gate / down` 每个 SVD 线性层都会单独发起一次请求

2. 电脑端执行顺序是同步合并
   - 先发送当前算子的输入向量和元数据
   - 本地并行计算前缀 rank
   - 等待手机端返回尾部 rank 输出
   - 把本地结果和远端结果相加
   - 当前算子完成后，图执行器才会继续走后续节点

3. 手机端当前只处理“一个算子的一次尾部 rank 计算”
   - 收到请求后只计算 `layer_id + op_id` 对应的尾部贡献
   - 计算完成后立刻回包
   - 不会在手机端连续推进多个层，也不会缓存某一层的中间激活继续往后算

4. 当前不支持“连续若干层全量卸载到端侧，端侧连续执行完这些层后再把中间结果返回”
   - 如果后续需要这个能力，必须把协议升级为“多层子图请求”
   - 也就是一次请求里携带起止层号和初始激活，由手机端连续执行多层后再返回最终中间激活

## 3. 新增接口

### 3.1 `llama_context_params`

文件：
- [llama.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/include/llama.h)

新增字段：

- `const float * svd_offload_rates`
  - 每层卸载率数组，取值范围 `[0, 1]`
- `uint32_t svd_offload_rate_count`
  - 卸载率数组长度
- `const char * svd_offload_host`
  - 远端设备地址
- `uint16_t svd_offload_port`
  - 远端设备端口
- `int32_t svd_offload_timeout_ms`
  - 网络超时
- `bool svd_offload_enabled`
  - 是否启用协同卸载

用途：

- 未来调度器可以在建图前直接填充这些字段，无需再改底层算子接口
- 当 `svd_offload_enabled == false` 且仍传入 `svd_offload_rates` 时，当前测试程序还支持把这些 rate 解释成本地 SVD 截断比例，用于“不协同、只算部分 rank”的本地性能实验

## 4. 新增/修改的核心函数

### 4.1 图构建层

文件：
- [llama-graph.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-graph.cpp)
- [llama-model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp)

函数：

- `llm_graph_context::build_mm_svd(...)`
  - 为每个 `GGML_OP_MUL_MAT_SVD` 节点挂载层号、算子号、卸载率元数据
  - 协同模式下不在建图时截断 prefill 路径
  - 非协同模式下，若传入 per-layer rate，则把它解释为本地 `k_trunc` 比例，直接在图上构造局部截断 SVD

- `llm_graph_context::build_ffn_svd_qwen2(...)`
  - 把当前层卸载率统一传给 `up / gate / down` 三个 SVD 节点

- `llm_build_qwen2_svd::llm_build_qwen2_svd(...)`
  - 从 `llm_graph_params` 中读取每层卸载率
  - 去掉原有实验性质硬编码 rank

### 4.2 GGML 张量元数据

文件：
- [ggml.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml.h)
- [ggml.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml.c)

函数：

- `ggml_mul_mat_svd_set_offload_meta(...)`
  - 给 `GGML_OP_MUL_MAT_SVD` 节点写入：
    - `svd_layer_id`
    - `svd_op_id`
    - `svd_offload_rate`

### 4.3 CPU 协同卸载客户端

文件：
- [ggml-svd-offload.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.h)
- [ggml-svd-offload.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.c)

函数：

- `ggml_svd_offload_set_client_config(...)`
  - 设置客户端网络配置

- `ggml_svd_offload_begin_request(...)`
  - 发送协同请求
  - 请求内容包含层号、算子号、分界 rank、输入向量
  - 一次请求只对应一个 SVD 算子，不包含跨层批处理语义

- `ggml_svd_offload_finish_request(...)`
  - 等待远端返回尾部 rank 贡献

- `ggml_svd_offload_close_client()`
  - 关闭长连接

### 4.4 SVD 单 token 协同执行

文件：
- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)

函数：

- `ggml_compute_forward_mul_mat_svd_vec(...)`
  - 当前核心协同逻辑入口
  - 执行流程：
    1. `thread 0` 发起远端请求
    2. 本地线程并行计算前缀 rank
    3. 等待远端返回尾部 rank 结果
    4. 本地把远端返回结果加到 `dst`
    5. 若网络失败，自动回退到本地全量计算，保证正确性
  - 当前同步点发生在单个 SVD 算子内部，不会跨多个层聚合等待

### 4.5 手机端服务进程

文件：
- [svd_mobile_server.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp)

函数：

- `compute_remote_tail(...)`
  - 根据层号和算子号，从预加载模型中取对应 `U/V`
  - 只计算尾部 rank 对应的输出贡献

用途：

- 当前在本机双进程调试中充当手机端
- 后续可以直接复用到 Android 侧

### 4.6 电脑端测试入口

文件：
- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp)

新增能力：

- 支持传入 `host:port`
- 支持传入单个卸载率
- 支持传入每层 CSV 卸载率或文件路径
- 当不传 `host:port` 但传入 rate 时，启用“本地 SVD 截断模式”

示例：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=./bin ./decode_svd_test \
  ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  4 16 0 127.0.0.1:7788 0.75
```

按层文件示例：

```bash
0.75,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

## 5. 双进程调试方法

### 5.1 启动手机端进程

```bash
cd /home/tianruiming/CE_ADA_LLAMA
./build-release-current/svd_mobile_server \
  src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  7788
```

### 5.2 启动电脑端进程

全本地：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=./bin ./decode_svd_test \
  ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  4 16 0
```

协同卸载：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=./bin ./decode_svd_test \
  ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  4 16 0 127.0.0.1:7788 0.75
```

本地 50% 截断：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
LD_LIBRARY_PATH=./bin ./decode_svd_test \
  ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  8 20 0 '' 0.5
```

## 6. 验证结果

### 6.1 正确性

模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`

Prompt：

- `Once upon a time`

验证结果：

- 全本地输出：
  - `Generated text: , there was a`
- 仅第 0 层卸载 `75%` 输出：
  - `Generated text: , there was a`
- 所有层都卸载 `75%` 输出：
  - `Generated text: , there was a`

说明：

- 本地与协同路径在本轮测试中给出了相同生成结果
- `Top-5 next-token candidates` 在修正 prefill 截断问题后也恢复一致

### 6.2 吞吐

高编号核绑核测试结果：

- CPU 核绑定：
  - 全本地基线：`60-79`
  - 手机端服务：`60-69`
  - 协同电脑端：`70-79`

同一台电脑双进程回环测试结果：

- 全本地：
  - `Decode-only throughput: 8.49069 tokens/s`
  - `End-to-end throughput: 8.47113 tokens/s`
- 仅第 0 层卸载 `75%`：
  - `Decode-only throughput: 4.27591 tokens/s`
  - `End-to-end throughput: 4.27092 tokens/s`
- 所有 28 层都卸载 `75%`：
  - `Decode-only throughput: 0.275865 tokens/s`
  - `End-to-end throughput: 0.275845 tokens/s`
- 非协同、本地每层只保留前 `50%` rank：
  - `Decode-only throughput: 9.09186 tokens/s`
  - `End-to-end throughput: 9.06971 tokens/s`

说明：

- 本地 50% 截断吞吐高于满秩基线，但生成文本已经和满秩结果不一致，因此只能视为性能实验，不满足“保证推理正确性”的要求
- 当前协同实现能保证正确性的前提是：本地前缀 rank 和远端尾部 rank 在同一个 SVD 算子上相加后再继续执行后续图节点

## 7. 结论

本轮已经完成：

- 建图时按层卸载率接口
- `SVD_MUL_MAT` 单 token decode 协同卸载
- 电脑端网络发送 / 接收
- 手机端预加载模型并按层按算子计算尾部 rank
- 本机双进程联调
- 正确性验证

当前未达成项：

- 在“同一台 CPU 电脑上双进程模拟手机端”的条件下，吞吐没有达到 `10 tokens/s`

直接原因：

- 当前手机端服务使用的是标量 C++ 尾部 rank 计算
- 本机双进程会争用同一套 CPU 资源
- 当大量层都触发卸载时，网络往返和远端计算都会显著放大 decode 延迟
- 当前请求粒度是单算子同步请求，FFN 的 `up / gate / down` 会把等待点进一步放大

当前能力边界：

- 已支持：单个 SVD 算子的“本地前缀 + 远端尾部 + 同步合并”
- 不支持：连续多层整段全量卸载后，由手机端连续推进多层再统一返回

## 8. 后续建议

下一步要继续把吞吐压到目标附近，优先做下面三件事：

1. 把 `svd_mobile_server` 的尾部 rank 计算改成复用 GGML/ggml-cpu 向量内核
2. 把多个连续 SVD 算子合并成更粗粒度请求，减少单层 `up / gate / down` 的 RTT 次数
3. 调度器不要让所有层都超过 `50%`，而是只挑少量高收益层触发卸载
4. Android 端落地时让手机端独占自己的 CPU 核，避免像本机双进程这样与电脑端争用同一组核

## 9. 当前稳定版本回归补充

### 9.1 本次回归问题

在后续继续优化时，曾出现下面的错误现象：

- `75%` 卸载输出变成 `thereketlulululululu`
- `100%` 卸载输出变成 `,,,,,,,,`
- 首轮 `Top-5 next-token candidates` 也已经偏离单机基线

最终定位结果：

- `up + gate` 合并缓存确实需要从“输入地址命中”改成“输入内容哈希命中”
- 但真正导致模型算错的直接原因，是手机端那版 `vec_dot` 快路径在当前张量类型下会触发：
  - `request failed: missing float conversion for SVD tensor type`
- 客户端收到失败后把远端结果清零，最终导致生成结果退化

同时，服务端新路径还暴露出一个分配错误：

- `GGML_ASSERT(ggml_get_no_alloc(ctx) == true) failed`

根因是：

- 服务端临时 ggml context 使用了 `ggml_backend_alloc_ctx_tensors(...)`
- 但 `ggml_init_params.no_alloc` 误设为 `false`

### 9.2 当前稳定修复

当前稳定版本的修复方式：

- 客户端：
  - 保留 `up + gate` 合并请求
  - gate 缓存键改为 `layer_id + rank_start + input_hash`

- 服务端：
  - 不再直接依赖那条会失败的 `vec_dot` 类型特征路径
  - 改为直接复用 ggml 内置 `mul_mat`
  - 对尾部 rank 计算构造两段小图：
    - `tmp = mul_mat(v_tail, input)`
    - `out = mul_mat(u_tail, tmp)`
  - 修正 `ggml_init_params.no_alloc = true`

### 9.3 当前实际运行方式

本次回归严格使用隔离核，避免争用：

- cgroup：
  - `/sys/fs/cgroup/tianruiming-exclusive`
- 手机端：
  - 绑核 `60-67`
  - `OMP_NUM_THREADS=8`
- 电脑端：
  - 绑核 `68-75`
  - `OMP_NUM_THREADS=8`
- 实验顺序执行，不并发启动多个客户端

服务端：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
sudo bash -lc '
  echo $$ > /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs
  exec taskset -c 60-67 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
    ./svd_mobile_server ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 7788 8
'
```

客户端基线：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
taskset -c 68-75 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0
```

客户端协同：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
taskset -c 68-75 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0 127.0.0.1:7788 1.0
```

### 9.4 最新实验结果

测试日期：

- 2026-04-03

结果：

| 场景 | Decode-only throughput | End-to-end throughput | 正确性 |
| --- | --- | --- | --- |
| 单机基线 | `9.71403 tok/s` | `9.69112 tok/s` | 正确 |
| 协同卸载 `75%` | `9.42101 tok/s` | `9.40045 tok/s` | 正确 |
| 协同卸载 `100%` | `6.52324 tok/s` | `6.51379 tok/s` | 正确 |

三组输出一致：

- `Generated text: , there was a young girl named Lily`

说明：

- 当前版本已经修复“模型计算错误”
- 但并没有实现“任意卸载率都能提速”
- `75%` 卸载只是接近单机
- `100%` 卸载已经明显慢于单机

### 9.5 当前结论

当前稳定版本的状态应总结为：

- 已完成：
  - 正确性修复
  - `up + gate` 合并请求
  - 手机端改为复用 ggml 内置 `mul_mat`
  - 严格绑核下的单机 / 协同回归验证

- 未完成：
  - 手机端请求执行器常驻化
  - 消除每请求新建小图 / backend / buffer 的高固定开销
  - 在 `75% / 100%` 乃至任意卸载率下的稳定吞吐提升

因此，下一步工作的重点不再是“修输出”，而是“把当前正确路径做快”。

### 9.6 电脑端 20% 负载下的补充实验

测试日期：

- 2026-04-03

测试约束：

- 手机端固定绑核：`60-67`
- 电脑端 decode 固定绑核：`68-75`
- 电脑端背景负载也固定在：`68-75`
- 手机端 `OMP_NUM_THREADS=8`
- 电脑端 `OMP_NUM_THREADS=8`
- 所有场景顺序执行，避免与其它实验争用核心

背景负载说明：

- 使用 `stress-ng --cpu 8 --cpu-load 20`
- 目的是模拟电脑端算力被额外占用时，协同卸载是否有收益

结果：

| 场景 | Decode-only throughput | End-to-end throughput | 正确性 |
| --- | --- | --- | --- |
| 全部在电脑端 decode | `1.90802 tok/s` | `1.90727 tok/s` | 正确 |
| 卸载 `50%` 到手机端 | `1.83690 tok/s` | `1.83620 tok/s` | 正确 |
| 卸载 `70%` 到手机端 | `2.13020 tok/s` | `2.12929 tok/s` | 正确 |
| 卸载 `80%` 到手机端 | `2.15228 tok/s` | `2.15136 tok/s` | 正确 |
| 卸载 `100%` 到手机端 | `1.71288 tok/s` | `1.71228 tok/s` | 正确 |

五组输出一致：

- `Generated text: , there was a young girl named Lily`

说明：

- 在电脑端承受 `20%` 背景负载时，`70%` 和 `80%` 卸载优于纯本地
- 本轮中 `80%` 卸载最好
- `100%` 卸载反而慢于 `70% / 80%`
- 当前代码仍然只在 `offload_rate > 0.5` 时触发真实协同，因此表中的 `50%` 实际上不会真正走远端尾部计算

### 9.7 电脑端 2 / 4 / 6 / 8 核下的补充实验

测试日期：

- 2026-04-03

测试约束：

- 手机端固定绑核：`60-67`
- 手机端固定 `8` 核 `8` 线程
- 电脑端不加背景负载
- 电脑端按不同核数分别绑核并同步设置线程数：
  - `2` 核：`68-69`
  - `4` 核：`68-71`
  - `6` 核：`68-73`
  - `8` 核：`68-75`
- 所有场景顺序执行

完整汇总如下，单位均为 `tok/s`：

| 电脑端核心数 | 本地不卸载 | 卸载 `50%` | 卸载 `70%` | 卸载 `80%` | 全部卸载 |
| --- | --- | --- | --- | --- | --- |
| `2` 核 | `3.84163` | `3.43610` | `5.10245` | `5.43680` | `5.71402` |
| `4` 核 | `6.90966` | `6.26546` | `6.48435` | `6.41904` | `6.40484` |
| `6` 核 | `9.09884` | `8.94079` | `6.12172` | `6.10186` | `5.97503` |
| `8` 核 | `9.86733` | `9.65941` | `6.06631` | `6.09167` | `6.03659` |

对应的 End-to-end throughput 如下，单位均为 `tok/s`：

| 电脑端核心数 | 本地不卸载 | 卸载 `50%` | 卸载 `70%` | 卸载 `80%` | 全部卸载 |
| --- | --- | --- | --- | --- | --- |
| `2` 核 | `3.83842` | `3.43367` | `5.09739` | `5.43086` | `5.70651` |
| `4` 核 | `6.89936` | `6.25621` | `6.47410` | `6.40905` | `6.39378` |
| `6` 核 | `9.08105` | `8.92369` | `6.10528` | `6.08433` | `5.95571` |
| `8` 核 | `9.84457` | `9.63734` | `6.04716` | `6.07055` | `6.01063` |

全部 `16` 组输出均一致：

- `Generated text: , there was a young girl named Lily`

说明：

- 电脑端只有 `2` 核时，卸载比例越高越有利，`100%` 最快
- 电脑端 `4` 核时，`70%` 卸载最好，但相比纯本地优势已很小
- 电脑端提升到 `6 / 8` 核后，本地 decode 明显更强，协同卸载整体不划算
- 当前 `50%` 这一列仍不是严格意义上的半卸载，而是接近本地参考值，因为代码条件是 `offload_rate > 0.5`
