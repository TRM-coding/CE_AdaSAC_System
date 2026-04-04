# 2026-04-04 协同卸载阶段耗时对比实验记录

## 目的

在“同一台机器模拟电脑端 + 手机端”的条件下，对比：

- 电脑端单机本地执行时，SVD FFN 三个部分的运行时间：`up`、`gate`、`down`
- 手机端服务端执行对应卸载子任务时的运行时间：`up+gate`、`down`

验证如下假设：

> 既然手机端与电脑端是同机模拟、核数相同、网络传输开销可忽略，那么两端纯计算时间应该非常接近。

## 实验前提

### 机器与隔离方式

服务器：`10`

为了避免核心抢占，实验前使用现有 cgroup：

- cgroup: `/sys/fs/cgroup/tianruiming-exclusive`
- 该 cgroup 的 `cpuset.cpus = 60-79`

实验时进一步绑核：

- 手机端服务进程：`60-67`
- 电脑端 decode 进程：`68-75`

两端均使用：

- `8` 核
- `8` 线程
- `OMP_NUM_THREADS=8`

### 模型与可执行文件

工作目录：`/home/tianruiming/CE_ADA_LLAMA/build-release-current`

模型：

- `../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`

程序：

- 客户端：`./decode_svd_test`
- 服务端：`./svd_mobile_server`

## 本次新增 profile

为了完成这次对比，在 `ggml-cpu.c` 中新增了只读观测型 profile 打印，不改变计算逻辑。

新增输出项：

- `[svd-local-op-profile]`
- 分别统计本地单机路径中的：
  - `up_total`
  - `gate_total`
  - `down_total`
  - 以及每个算子的两段 SVD 乘法时间：
    - `*_v`
    - `*_u`

说明：

- `up/gate/down_total` 表示单个算子在本地完整 SVD 路径中的总运行时间累计
- `*_v` 表示第一段 `V * x`
- `*_u` 表示第二段 `U * tmp`

## 实验命令

### 1. 本地单机路径

先启动客户端进程，再放入隔离 cgroup，并绑核到 `68-75`：

```bash
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0
```

### 2. 协同路径

服务端放入隔离 cgroup，并绑核到 `60-67`：

```bash
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./svd_mobile_server ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 7788 8
```

客户端放入隔离 cgroup，并绑核到 `68-75`：

```bash
env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS=8 \
  ./decode_svd_test ../src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf 8 8 0 127.0.0.1:7788 1.0
```

## 实验结果

### A. 单机本地结果

吞吐：

- `Decode-only throughput: 10.3957 tok/s`
- `End-to-end throughput: 10.3745 tok/s`

本地阶段 profile：

```text
[svd-local-op-profile] up_ops=225 gate_ops=225 down_ops=225 \
up_total=203.295 ms gate_total=201.769 ms down_total=200.060 ms \
up_v=30.448 ms gate_v=29.265 ms down_v=169.259 ms \
up_u=171.454 ms gate_u=170.946 ms down_u=30.327 ms
```

整理后：

| 项目 | 总时间 |
| --- | ---: |
| up | `203.295 ms` |
| gate | `201.769 ms` |
| down | `200.060 ms` |
| up + gate | `405.064 ms` |

进一步拆分：

| 项目 | V*x | U*tmp |
| --- | ---: | ---: |
| up | `30.448 ms` | `171.454 ms` |
| gate | `29.265 ms` | `170.946 ms` |
| down | `169.259 ms` | `30.327 ms` |

观察：

- `up/gate` 的主要耗时在第二段 `U * tmp`
- `down` 的主要耗时在第一段 `V * x`
- 这与三个矩阵乘法的维度结构是一致的

### B. 协同路径结果

吞吐：

- `Decode-only throughput: 7.1204 tok/s`
- `End-to-end throughput: 7.08455 tok/s`

客户端 profile：

```text
[svd-offload-client] up_gate_req=225 down_req=225 other_mat_req=0 gate_cache_hits=225 gate_cache_checks=450 miss_invalid=1 miss_layer=27 miss_rank=0 miss_hash=197 miss_output=0 miss_data=0 send_up_gate=17.799 ms send_down=56.495 ms send_other=0.000 ms wait_up_gate=508.050 ms wait_down=265.381 ms wait_other=0.000 ms
```

服务端 profile：

```text
[svd-offload-server] up_gate_req=225 down_req=225 other_mat_req=0 up_gate_miss=28 down_miss=28 other_miss=0 create_up_gate=2.547 ms create_down=2.267 ms create_other=0 ms run_up_gate=463.871 ms run_down=270.490 ms run_other=0 ms
```

整理后：

| 项目 | 时间 |
| --- | ---: |
| 手机端 run_up_gate | `463.871 ms` |
| 手机端 run_down | `270.490 ms` |
| 客户端 wait_up_gate | `508.050 ms` |
| 客户端 wait_down | `265.381 ms` |

观察：

- 客户端等待时间与服务端 run 时间基本一一对应
- 说明网络 / 协议不是主导瓶颈，主耗时仍是服务端实际计算
- `create_up_gate` / `create_down` 都只有约 `2~3 ms`，不是主要问题

## 关键对比

### 1. 本地 up+gate vs 手机端 up_gate

本地：

- `up + gate = 203.295 + 201.769 = 405.064 ms`

手机端：

- `run_up_gate = 463.871 ms`

差值：

- `463.871 - 405.064 = 58.807 ms`

即：

- 手机端 `up+gate` 比电脑端本地对应计算慢约 `14.5%`

### 2. 本地 down vs 手机端 down

本地：

- `down = 200.060 ms`

手机端：

- `run_down = 270.490 ms`

差值：

- `270.490 - 200.060 = 70.430 ms`

即：

- 手机端 `down` 比电脑端本地对应计算慢约 `35.2%`

## 本次实验结论

### 结论 1：同机模拟下，两端对应计算时间并不接近

与原先直觉不同，在当前实现中：

- 手机端服务端执行 `up+gate`，明显慢于电脑端本地执行 `up + gate`
- 手机端服务端执行 `down`，也明显慢于电脑端本地执行 `down`

因此：

> “既然是同一台机器模拟，两端纯计算时间应该非常接近” 这一假设，在当前实现里不成立。

### 结论 2：协同性能下降的主要原因已经定位到“手机端执行路径本身更慢”

在隔离 cgroup 和严格绑核后：

- 单机本地：`10.3957 tok/s`
- 协同卸载：`7.1204 tok/s`

造成下降的主要原因不是核心抢占，而是：

- 被卸载出去的 SVD 子任务，在手机端 server 这条执行路径上确实比电脑端本地更慢

也就是说：

- 卸载确实降低了电脑端本地计算负担
- 但手机端承接这部分计算的效率不够高
- 因而总体吞吐下降

### 结论 3：当前主问题不是 create 开销，而是 run 开销

从服务端 profile 看：

- `create_up_gate = 2.547 ms`
- `create_down = 2.267 ms`
- `run_up_gate = 463.871 ms`
- `run_down = 270.490 ms`

所以当前需要继续排查的是：

- 为什么 `svd_mobile_server` 的小图执行路径，实际 run 时间明显慢于本地同类计算

而不是：

- 为什么建图慢
- 为什么初始化慢

## 下一步建议

下一步优先研究：

1. `svd_mobile_server.cpp` 中 `run_up_gate_executor()` 与 `run_mat_executor()` 的执行路径
2. 与本地 `ggml_compute_forward_mul_mat_svd_vec()` 的差异
3. 为什么同机同核数下：
   - `run_up_gate` 比本地 `up + gate` 更慢
   - `run_down` 比本地 `down` 更慢，且差距更大

建议后续继续做两组细化实验：

- 把手机端 `up` / `gate` 拆开分别跑，确认慢点是否来自 `up+gate` 合并图
- 在服务端进一步加更细的 run 内部分段 profile，拆成：
  - tensor_set
  - backend_graph_compute
  - tensor_get

这样可以继续判断：

- 慢在图执行本身
- 还是慢在输入输出张量搬运与 backend 调度
