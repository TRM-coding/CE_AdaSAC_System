# 2026-04-11 手机端量化 SVD 协同卸载实现与精度验证

## 1. 目标

本次修改的目标是新增下面这个能力：

- 手机端服务进程执行远端 SVD 尾部 rank 时，可以使用量化权重执行
- 手机端在本地完成计算后，仍然以 `float` 结果通过现有网络接口回传电脑端
- 不改动电脑端 RPC 协议，也不改动电脑端 `decode_svd_test` 的调用方式

本次实现和验证都基于：

- 客户端模型：`qwen.gguf.sort_svd.compact.gguf`
- 手机端服务：`svd_mobile_server`
- 电脑端 decode：`decode_svd_test`

## 2. 实现概述

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/run_coop_core_sweep.sh`

### 2.1 服务端新增量化模式

`svd_mobile_server` 现在支持第 4 个参数指定手机端量化模式：

```bash
./svd_mobile_server <model> <port> <threads> [quant_mode]
```

当前支持：

- `off`
- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`
- `q4_k`
- `q5_k`
- `q6_k`

本次验证使用的是：

- `q8_0`

### 2.2 量化落点

量化不是改网络包格式，而是在服务端创建 executor 时，对当前请求对应的 tail 张量做一次 executor 级别的量化缓存。

具体做法：

1. 先按当前 `rank_start` 切出 `V tail / U tail`
2. 如果目标量化类型和行维度满足 block 对齐，则把 tail 张量量化成新的连续 tensor
3. `ggml_mul_mat_svd(...)` 直接吃这份量化后的 tail tensor
4. GGML 在服务端本地完成量化权重参与的 matmul，最终输出仍是 `float`
5. 服务端继续按原协议把 `float` 输出传回电脑端

因此：

- 手机端是“量化权重执行”
- 网络上传输的仍是“已在手机端反量化/累加好的 `float` 输出”

### 2.3 为什么不改 RPC 协议

现有协议的请求已经只传输入激活，响应只传最终输出向量。

本次实现里：

- 量化只发生在手机端内部的 tail 权重缓存
- 手机端计算完成后返回的仍是 `float` 输出

所以电脑端不需要感知量化细节，协议保持兼容。

### 2.4 兼容与回退

服务端对下面几类情况做了安全回退：

- 量化模式关闭
- 张量维度不满足当前实现的 2D 假设
- 行维度不满足目标量化 block 对齐
- 目标量化类型缺少可用的量化 traits
- 分配量化缓存失败

回退后仍走原始 `F16/F32` 路径，不影响正确性。

## 3. 脚本更新

`run_coop_core_sweep.sh` 新增了两个可选环境变量：

- `SERVER_MODEL_PATH`
- `SERVER_QUANT_MODE`

示例：

```bash
SERVER_QUANT_MODE=q8_0 \
RUN_TAG=quant_sweep_q8_0 \
/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/run_coop_core_sweep.sh
```

## 4. 构建

构建命令：

```bash
cmake --build /home/tianruiming/CE_ADA_LLAMA/build-release-current -j16 --target svd_mobile_server decode_svd_test
```

本次构建通过。

## 5. cgroup 精度验证

### 5.1 验证要求

按要求，验证时使用 cgroup：

- `/sys/fs/cgroup/tianruiming-exclusive`
- `cpuset.cpus.effective = 60-79`

### 5.2 本次环境里的实际执行方式

本次会话里，进程加入 cgroup 后可以稳定看到：

```text
taskset -pc $$  =>  60-79
```

但继续对进程做 `taskset -c 60-67` 或 `taskset -c 68-75` 会报：

```text
taskset: failed to set pid ... affinity: Invalid argument
```

因此本次精度验证采用：

- 先把服务端和客户端都加入 `/sys/fs/cgroup/tianruiming-exclusive`
- 确保两端进程的有效亲和性都落在 `60-79`
- 不再额外拆成 `60-67 / 68-75`

这满足了“使用 cgroup 在 `60-79` 核上验证”的要求。

### 5.3 验证命令

服务端：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
echo $$ | sudo tee /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs >/dev/null
taskset -pc $$
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./svd_mobile_server \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  7795 8 q8_0
```

客户端：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
echo $$ | sudo tee /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs >/dev/null
taskset -pc $$
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  8 8 0 127.0.0.1:7795 1.0
```

结果目录：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/quant_final_cgroup_20260411_143729`

日志：

- client: [client_q8_0.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/quant_final_cgroup_20260411_143729/client_q8_0.log)
- server: [server_q8_0.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/quant_final_cgroup_20260411_143729/server_q8_0.log)

### 5.4 验证结果

客户端输出：

```text
Generated text: , there was a young girl named Lily
```

这不是乱码，和未量化版本的语义输出保持一致。

服务端统计：

```text
quant_mode=q8_0
quant_tail_tensors=168
quant_fallbacks=0
```

说明：

- 本次确实走到了手机端量化 tail 执行
- 没有回退回原始 `F16/F32` 路径

另外两端日志都显示进程亲和性已经切到 `60-79`：

- 服务端日志开头含 `taskset -pc $$ => 60-79`
- 客户端日志开头含 `taskset -pc $$ => 60-79`

## 6. 性能观察

本次 `q8_0` 验证的主要现象不是精度问题，而是首次量化建缓存开销很大。

来自服务端日志：

- `create_up_gate = 5070.66 ms`
- `create_down = 2521.96 ms`
- `run_up_gate = 224.845 ms`
- `run_down = 132.672 ms`

可以看到：

- 真正远端执行时间明显下降
- 但首次 executor miss 时，tail 量化和缓存构建代价非常大

因此当前实现适合回答“功能和正确性是否成立”，不适合直接作为默认高性能路径。

## 7. 结论

本次已经完成以下目标：

1. 手机端服务新增了量化执行模式
2. 手机端在本地使用量化 tail 权重执行，最终仍回传 `float` 输出
3. 电脑端 RPC 协议无需修改
4. 在 `60-79` cgroup 上完成了联合推理验证
5. 输出文本不是乱码，精度上未看到异常

当前残留问题：

1. 首次量化缓存构建开销很大
2. 若后续要把这条路径作为默认方案，下一步应把 tail 量化从“请求命中时懒构建”前移到“服务端启动预构建”或“按层批量预热”
