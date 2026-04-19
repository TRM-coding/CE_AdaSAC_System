# Exp10: SVD 本地 A/B 核心组拆分实验记录

## 1. 实验目的

本轮实验验证两件事：

1. 本地 A/B 核心组 rank 拆分是否保持输出正确。
2. 在空载和带载两种场景下，拆分版本的速度是否能保持在当前 SVD 基线的 `10%` 范围内。

## 2. 实验环境

- 模型：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 可执行程序：`build-release-current/decode_svd_test`
- token 数：`1`
- 线程数：`8`
- 当前实现：只拆 `x * V`，`tmp * U` 走原 fast path

用户要求的实验 CPU 环境是：

- `tianruiming-exclusive` cgroup
- `60-79` 核
- 从中选 `8` 个核心

但本次实际 shell 所在进程只允许访问 `0-59` 核。现场检查结果是：

- `taskset -c 60-67 ...` 直接报 `Invalid argument`
- 往 `/sys/fs/cgroup/.../cgroup.procs` 写入当前进程也会报 `Permission denied`

因此本轮实际实验只能在当前可访问的 `52-59` 八个核上完成。这个差异需要明确记录，不能假装已经在 `60-79` 上复现。

## 3. 实验命令

### 3.1 空载基线

```bash
taskset -c 52-59 \
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0
```

### 3.2 空载 A/B split

```bash
taskset -c 52-59 \
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off 0 52-55 56-59 0.50
```

语义：

- A 组：`52-55`
- B 组：`56-59`
- `groupA_share = 0.50`

### 3.3 带负载场景

先在 A 组 `52-55` 上加满 4 个 busy loop：

```bash
taskset -c 52-55 sh -c 'for i in 1 2 3 4; do yes > /dev/null & done; wait'
```

然后分别测：

基线：

```bash
taskset -c 52-59 \
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0
```

拆分版：

```bash
taskset -c 52-59 \
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off 0 52-55 56-59 0.25
```

这里 `0.25` 的含义是把更小的 rank 段分给负载更高的 A 组，把大部分 rank 留给更空闲的 B 组。

## 4. 空载结果

### 4.1 三次重复

| 场景 | run1 | run2 | run3 | 平均 |
|---|---:|---:|---:|---:|
| baseline | `30.3909` | `31.1768` | `30.6140` | `30.7272 tok/s` |
| split `share=0.50` | `30.3340` | `31.8831` | `30.3082` | `30.8418 tok/s` |

相对基线变化：

- `+0.37%`

结论：

- 速度没有比当前基线慢 `10%`
- 反而在这组三次重复里略快

### 4.2 正确性

三次空载实验里，基线和 split 的结果都一致：

- `Top-1 token = ","`
- `Generated text: ,`

top-5 logits 排序也一致，说明当前落地方案没有破坏单 token 输出。

## 5. 带负载结果

### 5.1 share 扫描

在 A 组 `52-55` 加满负载后，单次调参结果如下：

| share(A) | Decode-only throughput | 输出 |
|---|---:|---|
| `0.10` | `0.156775 tok/s` | `,` |
| `0.15` | `0.158039 tok/s` | `,` |
| `0.20` | `0.158089 tok/s` | `,` |
| `0.25` | `0.159424 tok/s` | `,` |

这说明在“忙核组只拿更小 rank 段”的方向上，`share=0.25` 是这组测试里最好的点。

### 5.2 三次重复

最终对比基线和 `share=0.25`：

| 场景 | run1 | run2 | run3 | 平均 |
|---|---:|---:|---:|---:|
| baseline + load | `0.156481` | `0.156222` | `0.158715` | `0.157139 tok/s` |
| split `share=0.25` + load | `0.157342` | `0.158414` | `0.158853` | `0.158203 tok/s` |

相对基线变化：

- `+0.68%`

### 5.3 正确性

三次带负载实验中：

- baseline 输出全是 `,`
- split `share=0.25` 输出也全是 `,`

因此当前最终版在带负载条件下也保持了正确输出。

## 6. 失败方案记录

本轮实验里还验证过一个没有保留的激进方案：

- `x * V` 按 A/B 拆
- `tmp * U` 也按 A/B 拆
- 最后合并 partial result

结果是：

1. 早期版本直接出现全零 logits 和 `Generated text: !!!!!!!!`
2. 修补后即便单 token 输出恢复正确，吞吐仍掉到远低于当前基线，明显超过 `10%` 预算

所以最终代码没有保留这条路径，文档中的正式结论全部来自“只拆 `V` 阶段”的稳定实现。

## 7. 结论

在当前代码和当前可访问 CPU 环境下，可以给出下面的结论：

1. 给定核心组 `A/B` 和 `groupA_share`，当前实现已经能稳定按 rank 维拆分 `x * V` 的本地 SVD 计算。
2. 空载条件下，split 版本没有落后于当前 SVD 基线，满足“不低于基线 `10%`”的要求。
3. 在人为给 A 组加负载后，把更小的 rank 段分给 A 组、把更大的 rank 段分给 B 组，方向上是成立的，本轮重复实验中取得了小幅正收益。
4. 输出正确性在空载和带载两组正式实验中都保持住了。

## 8. 当前限制

这次实验也要如实记录限制：

1. 实验不是在目标 `60-79` 核上完成的，而是在当前可访问的 `52-59` 上完成的。
2. 当前收益主要来自 `V` 阶段的 rank 切分；`U` 阶段仍未做真正的双组并行归约。
3. 当前构建是 `GGML_USE_OPENMP`，所以线程到核心组的稳定性依赖静态线程槽位划分，而不是运行时 `sched_getcpu()`。

如果后续要继续推进，最值得继续研究的是：

- 非 OpenMP pthread threadpool 下做更严格的静态绑核验证
- 评估 `U` 阶段是否存在低开销 partial-sum 归并实现
- 在真正目标 cgroup `60-79` 环境里复现实验
