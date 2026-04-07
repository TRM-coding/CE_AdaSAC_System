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

## 本次结论

### 结论 1：当前主因仍然没变

当前协同卸载变慢的主因仍然是：

- 服务端执行的是“通用 CPU backend 小图路径”
- 本地执行的是“专用 SVD 单 token vec 快路径”

两边不是同一套实现，所以即使核数一致，时间也不会自然接近。

### 结论 2：create 开销不是主问题

本次和 2026-04-04 一样，`create_up_gate` / `create_down` 只有约 `2 ms`。

真正慢的仍然是：

- `run_up_gate`
- `run_down`

也就是服务端实际执行小图的这部分计算。

### 结论 3：当前已落地修改不足以修复问题

虽然本次已经减少了部分 backend 重复开销，但结果表明：

- 仅靠缓存 plan
- 仅靠补 threadpool

还不足以把协同路径拉回到本地 `9 tok/s` 左右。

## 本次排查中验证但未保留的方向

本次还尝试过两条更激进的路径，但都没有保留到当前代码：

1. 在服务端直接手写 `U/V` tail matvec
2. 在 OpenMP 构建下进一步修改 ggml，让 backend graph 优先走持久 threadpool worker 而不是现有 OpenMP 路径

结果：

- 第 1 条路径反而更慢
- 第 2 条路径会牵动 `ggml-cpu.c` 中一套只在非 OpenMP 路径下使用的数据结构，修改面过大，当前风险不合适

因此当前代码只保留了低风险的服务端小图优化。

## 下一步建议

如果目标是把协同路径真正拉回到本地 `9 tok/s` 附近，下一步不应继续在“小图调度边角”上打补丁，而应优先改执行粒度。

优先级建议如下：

1. 重新回到“整层 FFN 卸载”或“更粗粒度子图卸载”
2. 至少把 `up / gate / down` 的单算子 RPC 再减少一层同步点
3. 不再把主要希望寄托在 `svd_mobile_server.cpp` 的当前小图 backend 路径上

当前最简洁的总结是：

> 本次已经验证：问题不在核型、不在网络主路径、也不在建图 create，而在服务端实际运行的通用小图执行路径本身。当前已落地的安全优化只能带来很小收益，若要真正修复，需要升级卸载粒度，而不是继续微调单算子 server 小图。
