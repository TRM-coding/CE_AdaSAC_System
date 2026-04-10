# 2026-04-07 协同卸载问题跟进记录

## 目的

在已有两份报告的基础上，继续回答下面两个问题：

1. 为什么“同机模拟电脑端 + 手机端”时，手机端 `FFN up / gate / down` 仍明显慢于电脑端本地执行
2. 当前代码层面做了哪些修复，修复后的效果如何

关联文档：

- `exp6_profile_compare_2026-04-04.md`
- `svd_cooperative_offload_experiment_report.md`
- `svd_cooperative_offload_optimization_report.md`
- `svd_interface_callchain.md`

## 本次实验环境

### 绑核方式

仍使用同一组隔离 cgroup：

- cgroup: `/sys/fs/cgroup/tianruiming-exclusive`
- `cpuset.cpus = 60-79`

进程绑核：

- 手机端服务进程：`60-67`
- 电脑端 decode 进程：`68-75`

两端均使用：

- `8` 核
- `8` 线程
- `OMP_NUM_THREADS=8`

### 本次新增的机器拓扑核对

为排除“60-67 和 68-75 实际不是同类核心”的可能，本次额外检查了机器拓扑。

`lscpu` 与 `numactl --hardware` 的结果表明：

- `60-79` 全部位于 `socket 1`
- `60-79` 全部位于 `NUMA node 1`
- `60-79` 都是 `20-39` 的 SMT sibling

也就是说：

> 本次实验里的“手机端核组”和“电脑端核组”并不存在大小核或跨 NUMA 节点差异，至少从 CPU 拓扑上看，它们属于同一类执行资源。

因此，当前“手机端更慢”的主因，不能继续归因于绑核选错或核型不一致。

## 代码排查结论

本次重新对照了两条执行路径。

### 1. 电脑端本地路径

本地 decode 时，`GGML_OP_MUL_MAT_SVD` 最终走：

- `ggml_compute_forward_mul_mat_svd_vec()`

这是一条专门针对单 token 向量场景的 SVD 快路径：

- 直接在线程池内做 `V * x`
- 再做 `U * tmp`
- 不走 backend graph 调度
- 不做额外 `tensor_set / tensor_get`

### 2. 手机端服务端路径

服务端 `svd_mobile_server.cpp` 中，`down` 和 `up+gate` 走的是：

- 预创建小图
- 每次请求 `ggml_backend_tensor_set(...)`
- 然后 `ggml_backend_graph_compute(...)`
- 最后 `ggml_backend_tensor_get(...)`

这意味着：

> 虽然两端都在同一台机器、同样 8 线程下运行，但两边根本不是同一条 kernel，也不是同一套调度路径。

这仍然是当前性能差距的核心原因。

## 本次落地修改

本次在 `svd_mobile_server.cpp` 中落地的是一组低风险优化，目标是减少服务端小图路径中的重复调度开销，而不改变协议和数值语义。

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp`

### 修改 1：缓存 backend graph plan

对 `RemoteMatExecutor` 和 `RemoteUpGateExecutor`：

- 新增 `ggml_backend_graph_plan_t plan`
- 在创建 executor 时调用 `ggml_backend_graph_plan_create(...)`
- 在析构时释放 `plan`

作用：

- 避免每次请求重新对同一张小图做 plan

### 修改 2：执行时改用 `ggml_backend_graph_plan_compute`

原来每次请求调用：

```cpp
ggml_backend_graph_compute(backend, exec.graph)
```

现在改为：

```cpp
ggml_backend_graph_plan_compute(backend, exec.plan)
```

作用：

- 复用已缓存的 plan
- 降低每请求的调度开销

### 修改 3：给 CPU backend 挂持久 threadpool

在 `main()` 中新增：

- `ggml_threadpool_new(...)`
- `ggml_backend_cpu_set_threadpool(backend, threadpool)`

并补齐异常路径和退出路径的释放。

作用：

- 避免 backend 处于“没有显式 threadpool”的状态
- 为后续继续优化服务端 CPU 路径打基础

### 修改 4：补齐资源生命周期

新增了：

- executor 析构时释放 `plan`
- 退出前先 `clear()` executor cache
- 再释放 backend 和 threadpool

作用：

- 避免 `plan` 在 backend 已释放后再被销毁

## 复测命令

### 手机端

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
echo $$ | sudo tee /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs >/dev/null
exec taskset -c 60-67 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./svd_mobile_server ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 7788 8
```

### 电脑端

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
echo $$ | sudo tee /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs >/dev/null
exec taskset -c 68-75 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0 127.0.0.1:7788 1.0
```

## 本次复测结果

### 客户端结果

```text
[svd-offload-client] up_gate_req=225 down_req=225 other_mat_req=0 \
gate_cache_hits=225 gate_cache_checks=450 miss_invalid=1 miss_layer=27 \
miss_rank=0 miss_hash=197 miss_output=0 miss_data=0 \
send_up_gate=18.229 ms send_down=55.003 ms send_other=0.000 ms \
wait_up_gate=501.053 ms wait_down=265.054 ms wait_other=0.000 ms

Decode-only throughput: 7.19224 tokens/s
End-to-end throughput: 7.15608 tokens/s
```

### 服务端结果

```text
[svd-offload-server] up_gate_req=225 down_req=225 other_mat_req=0 \
up_gate_miss=28 down_miss=28 other_miss=0 \
create_up_gate=2.451 ms create_down=2.245 ms create_other=0 ms \
run_up_gate=461.027 ms run_down=265.971 ms run_other=0 ms
```

## 与 2026-04-04 数据对比

2026-04-04 报告中的对应数据：

- `run_up_gate = 463.871 ms`
- `run_down = 270.490 ms`
- `Decode-only throughput = 7.1204 tok/s`

本次数据：

- `run_up_gate = 461.027 ms`
- `run_down = 265.971 ms`
- `Decode-only throughput = 7.19224 tok/s`

对比后可以看到：

- `run_up_gate` 只下降了约 `2.844 ms`
- `run_down` 只下降了约 `4.519 ms`
- 总体吞吐只从约 `7.12 tok/s` 提升到约 `7.19 tok/s`

这说明：

> 本次“缓存 graph plan + 持久 threadpool”只带来了很小的收益，属于边角优化，并没有改变主瓶颈。

## 追加更新：第二轮修复（当前代码状态）

说明：

- 上面的小节记录的是同一天第一轮“低风险小修”版本
- 下面补充的是继续修复后的当前代码状态
- 当前仓库里的实现，以本节为准

### 第二轮代码改动

本轮保留了 3 个有效修改。

#### 修改 1：服务端不再走 backend 小图执行路径

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp`

核心变化：

- `RemoteMatExecutor` / `RemoteUpGateExecutor` 不再缓存 `ggml_backend_graph_plan_t`
- 改为缓存 `ggml_cplan`
- 输入输出直接绑定到常驻 `std::vector<float>` buffer
- 每次请求直接调用：

```cpp
ggml_graph_compute(exec.graph, &exec.plan)
```

- 图内节点也不再是两段 `mul_mat` 小图，而是直接构造 `ggml_mul_mat_svd(...)`

作用：

- 服务端尾部 rank 计算现在会走到和本地 decode 更接近的 `GGML_OP_MUL_MAT_SVD`
- 去掉了 `tensor_set / tensor_get / backend graph` 这一层额外调度和搬运

#### 修改 2：修正 `offload_rate` 的触发阈值

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`

修改前：

```c
dst->svd_offload_rate > 0.5f
```

修改后：

```c
dst->svd_offload_rate > 0.0f
```

作用：

- `offload_rate = 0.5` 现在会真实触发 partial-rank 协同
- 不再出现“看起来设了半卸载，实际完全没走远端”的歧义

#### 修改 3：保留 `svd_mobile_server` 的 OpenMP 构建修复

修改文件：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/CMakeLists.txt`

作用：

- `svd_mobile_server` 目标现在会显式继承 `GGML_USE_OPENMP`
- 并链接 `OpenMP::OpenMP_CXX`
- 避免服务端目标在某些实现尝试中静默退成单线程

### 第二轮复测结果

### 1. 单机本地基线

客户端命令：

```bash
cd /home/tianruiming/CE_ADA_LLAMA/build-release-current
echo $$ | sudo tee /sys/fs/cgroup/tianruiming-exclusive/cgroup.procs >/dev/null
exec taskset -c 68-75 env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0
```

结果：

```text
[svd-local-op-profile] up_ops=225 gate_ops=225 down_ops=225 \
up_total=213.118 ms gate_total=211.014 ms down_total=210.637 ms \
up_v=31.873 ms gate_v=30.520 ms down_v=178.651 ms \
up_u=179.901 ms gate_u=178.992 ms down_u=31.559 ms

Decode-only throughput: 9.93163 tokens/s
End-to-end throughput: 9.90903 tokens/s
```

### 2. 协同卸载 `offload_rate = 1.0`

客户端结果：

```text
[svd-offload-client] up_gate_req=225 down_req=225 other_mat_req=0 \
gate_cache_hits=225 gate_cache_checks=450 miss_invalid=1 miss_layer=27 \
miss_rank=0 miss_hash=197 miss_output=0 miss_data=0 \
send_up_gate=17.045 ms send_down=55.147 ms send_other=0.000 ms \
wait_up_gate=469.682 ms wait_down=199.985 ms wait_other=0.000 ms

Decode-only throughput: 7.95274 tokens/s
End-to-end throughput: 7.90958 tokens/s
```

服务端结果：

```text
[svd-offload-server] up_gate_req=225 down_req=225 other_mat_req=0 \
up_gate_miss=28 down_miss=28 other_miss=0 \
create_up_gate=1.994 ms create_down=1.613 ms create_other=0 ms \
run_up_gate=433.527 ms run_down=218.160 ms run_other=0 ms
```

对比本地基线可见：

- 本地 `up + gate` 总时间约为 `213.118 + 211.014 = 424.132 ms`
- 服务端 `run_up_gate = 433.527 ms`
- 本地 `down_total = 210.637 ms`
- 服务端 `run_down = 218.160 ms`

这说明：

> 第二轮修复后，远端 `up / gate / down` 的执行时间已经基本贴近本地 SVD 快路径，原先“服务端明显比本地慢一截”的主矛盾已经被显著削弱。

### 2.1 FFN 阶段的同粒度对比

为避免和 `60-79` 整池吞吐基线混淆，这里单独比较“同一组 8 线程电脑端 decode，在 FFN 阶段到底花了多少墙钟时间”。

统计口径：

- 本地：电脑端绑核 `68-75`，`OMP_NUM_THREADS=8`，FFN 直接在本机执行
- 协同：客户端绑核 `68-75`，服务端绑核 `60-67`，两端都为 `OMP_NUM_THREADS=8`
- 统计块分别来自：
  - 本地：`[svd-local-stage-profile]`
  - 客户端：`[svd-offload-client-stage-profile]`
  - 服务端：`[svd-offload-server-stage-profile]`

补充复测结果：

| FFN 阶段 | 本地 | 协同 |
| --- | --- | --- |
| 本地直接执行 FFN | `623.092 ms` | `0 ms` |
| RPC 发送 | `0 ms` | `32.916 ms` |
| RPC 等待返回 | `0 ms` | `737.821 ms` |
| 远端实际执行 | `0 ms` | `710.489 ms` |

这里要特别注意：

- 协同里的 `等待返回` 已经包含了 `远端实际执行`
- 因此客户端视角的 FFN 总墙钟时间是 `32.916 + 737.821 = 770.737 ms`
- 不能把 `710.489 ms` 再额外加一遍

按这组数据继续拆解：

- 协同 FFN 比本地 FFN 多出的总墙钟时间为 `770.737 - 623.092 = 147.645 ms`
- 其中远端实际执行相对本地多 `710.489 - 623.092 = 87.397 ms`
- 发送与额外等待开销为 `32.916 + (737.821 - 710.489) = 60.248 ms`

这组 FFN 粒度数据更直接地说明：

> 当前主问题已经不再是“远端完全算不动 FFN”，而是 FFN 被拆成 RPC 后，客户端必须额外承担发送与等待的墙钟成本。

### 3. 协同卸载 `offload_rate = 0.5`

在修正阈值后，`0.5` 档位会真实触发协同。

客户端结果：

```text
[svd-offload-client] up_gate_req=225 down_req=225 other_mat_req=0 \
gate_cache_hits=225 gate_cache_checks=450 miss_invalid=1 miss_layer=27 \
miss_rank=0 miss_hash=197 miss_output=0 miss_data=0 \
send_up_gate=7.038 ms send_down=22.738 ms send_other=0.000 ms \
wait_up_gate=425.419 ms wait_down=233.555 ms wait_other=0.000 ms

Decode-only throughput: 7.69232 tokens/s
End-to-end throughput: 7.67900 tokens/s
```

服务端结果：

```text
[svd-offload-server] up_gate_req=225 down_req=225 other_mat_req=0 \
up_gate_miss=28 down_miss=28 other_miss=0 \
create_up_gate=3.036 ms create_down=2.572 ms create_other=0 ms \
run_up_gate=406.146 ms run_down=218.657 ms run_other=0 ms
```

这组数据说明两件事：

- `50%` 现在确实在真实卸载，不再是“伪半卸载”
- 但当前协议下，`50%` 并没有比 `100%` 更快，说明剩余瓶颈已经不再是“server 内核太慢”，而是协同调度与同步成本

## 更新后结论

### 结论 1：服务端内核路径不再是当前主矛盾

第一轮结论里，主因是：

- 服务端执行“通用 CPU backend 小图路径”
- 本地执行“专用 SVD 单 token vec 快路径”

第二轮修复后，这个结论不再适用于当前代码状态。

当前状态更准确的描述是：

- 服务端已经切到接近本地的 SVD 执行路径
- `run_up_gate` 和 `run_down` 已经基本贴近本地同类计算
- “手机端算得明显更慢”不再是最主要的解释

### 结论 2：剩余差距主要来自 RPC 等待与过细粒度同步

即使服务端 kernel 已经接近本地，协同 `1.0` 仍只有约 `7.95 tok/s`，距离本地 `9.93 tok/s` 还有差距。

当前更像是下面这些因素在主导：

- 每个 token 仍需发起 `up+gate` 与 `down` 两类 RPC
- 客户端仍要等待远端返回后再做合并
- 协同粒度仍然是“单算子级”

### 结论 3：`offload_rate = 0.5` 的逻辑错误已修复，但这一档目前并不更优

修阈值前：

- `50%` 档位实际上不会触发远端协同

修阈值后：

- `50%` 会真实触发卸载
- 但实测约 `7.69 tok/s`
- 没有优于 `100%` 档位

### 结论 4：下一步重点应转向更粗粒度协同，而不是继续微调 server kernel

下一步更值得做的是：

1. 把 `up / gate / down` 再合并成更粗粒度的 FFN 协同
2. 减少每 token 的同步点和等待点
3. 不再把主要优化时间投入到 `svd_mobile_server.cpp` 的单算子 kernel 微调上

当前最简洁的总结改为：

> 第一轮结论确认了“服务端小图 backend 路径”是核心问题；第二轮修复已经基本把这部分问题解决。当前剩下的性能差距，主要不再是手机端算得慢，而是单算子 RPC 协同本身的等待成本与粒度设计。
