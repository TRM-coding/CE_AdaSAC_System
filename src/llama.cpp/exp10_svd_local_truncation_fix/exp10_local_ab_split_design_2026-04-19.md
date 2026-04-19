# Exp10: SVD 本地 A/B 核心组拆分设计

## 1. 目标

本轮实现的目标是把本地 `GGML_OP_MUL_MAT_SVD` 的 decode 热路径改成可控的双核心组执行：

1. 给定核心组 `A/B` 和截断后保留秩 `k_keep`，能够把 rank 维工作稳定拆到两组线程上。
2. 最终输出必须和当前本地 SVD 基线一致。
3. 空载场景下速度不能比当前 SVD 基线差超过 `10%`。

## 2. 最终落地方案

最终落地的是一个保守版本：

- 只在 `ggml_compute_forward_mul_mat_svd_vec()` 上做本地双组拆分
- 只拆第一阶段 `tmp = x * V`
- 第二阶段 `dst = tmp * U` 保持原来的全线程 fast path

也就是说，最终实现不是“U/V 两段都按 A/B 拆开算”，而是：

```text
x * V:
  rank 维按 share 切成 A 段和 B 段
  A 组线程只算 A 段
  B 组线程只算 B 段

tmp * U:
  仍按原 fast path 执行
```

这样做的原因很直接：

- `x * V` 本来就是沿 rank 维写 `tmp` 的不同切片，天然适合拆分
- `tmp * U` 会对同一个输出向量做 rank 维归约，双组并行会引入 partial sum 合并
- 对当前 `Q4 SVD` 路径，`U` 阶段的双组拆分实现虽然数学上可做，但实际验证里要么正确性不稳，要么性能损失明显超过 `10%`

## 3. 为什么没有保留“双阶段都拆”的方案

中间做过一版更激进的实现：

- `x * V` 按 A/B rank 段并行
- `tmp * U` 也按 A/B rank 段分别算 partial sum
- 最后再把两组 partial result 合并

这版方案的问题有两个：

1. `Q4`/量化 `U` 的局部切片处理很容易踩到布局和打包细节，早期版本直接出现 `Generated text: !!!!!!!!` 和全零 logits。
2. 即便修到单 token 结果正确，额外的中间 buffer、dequant 和 merge 仍把性能打得太低，远超 `10%` 预算。

因此当前版本明确放弃“双阶段都拆”，只保留第一阶段拆分。

## 4. 线程分组设计

### 4.1 用户接口

实验程序 [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp) 新增了三类参数：

- `group A cpu range`
- `group B cpu range`
- `groupA_share`

示例：

```bash
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off 0 52-55 56-59 0.50
```

语义如下：

- `52-55` 是 A 组核心
- `56-59` 是 B 组核心
- `0.50` 表示 `k_keep` 中约 `50%` 的 rank 分给 A 组，剩余分给 B 组

### 4.2 ggml 侧接口

在 [ggml-cpu.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cpu.h) 中增加了：

- `ggml_cpu_set_svd_local_split(...)`
- `ggml_cpu_clear_svd_local_split()`

`decode_svd_model.cpp` 在创建 threadpool 前设置分组配置；未开启时显式清空配置。

### 4.3 分片规则

在 [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c) 中，当前 rank 切分逻辑是：

```text
k_local = 本地实际参与计算的 rank
k_group_a = round(k_local * groupA_share)
```

然后再按 `U`/`tmp` 的 block size 对 `k_group_a` 做对齐，保证不会把量化块从中间截断。

最后两组的工作区间为：

- A 组: `[0, k_group_a)`
- B 组: `[k_group_a, k_local)`

每组内部再按自己线程数做一次均分。

## 5. OpenMP 下的稳定性处理

当前构建启用了 `GGML_USE_OPENMP`。这意味着本地计算线程不是 `ggml` 自己长期持有的 pthread worker，而是 OpenMP 运行时拉起的线程。

这件事直接影响分组判定。

早期实现是按 `sched_getcpu()` 判断“当前线程此刻跑在哪个 CPU”，然后再映射到 A/B 组。空载时这通常没问题，但一旦某组有明显负载，线程可能临时漂到同一个 CPU 上，导致：

- 多个线程拿到相同组内索引
- 某些 rank 切片重复计算
- 某些 rank 切片完全漏算
- 最终 logits 退化成全零或输出 `!`

因此当前最终实现对 `GGML_USE_OPENMP` 单独做了静态槽位映射：

- `params->ith < n_group_a` 的线程视为 A 组
- 后续线程视为 B 组

这保证分组逻辑不依赖运行时漂移。

如果未来切回非 OpenMP 的 pthread threadpool 路径，则仍可以通过 worker 的静态 `cpumask` 去判定所属核心组。

## 6. 正确性策略

当前版本的正确性约束有三条：

1. 只有 `x * V` 做 A/B rank 切分，避免 `U` 阶段并发归约带来的额外误差和复杂同步。
2. rank 切分边界按 block size 对齐，避免量化块被截断。
3. 仍保留 `Q4 SVD` 本地截断修复中的安全回退逻辑：如果量化因子进入了不兼容的 extra layout，则不走手写 vec 路径。

## 7. 为什么这版能满足 10% 速度预算

因为当前只在 `x * V` 阶段引入拆分逻辑：

- 不新增 `dst` partial sum buffer
- 不新增额外 `U` 阶段 merge
- 不引入跨组结果归并

拆分带来的额外成本基本只剩：

- 一次 rank 边界计算
- 两组线程只在自己的 rank 切片上工作

这也是为什么最终版在空载实验里可以把速度稳定压在基线附近，而失败的“双阶段拆分版”做不到。

## 8. 代码落点

本轮新增或主要修改点如下：

- [ggml-cpu.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cpu.h)
- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)
- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp)

对应职责分别是：

- `ggml-cpu.h`：暴露本地 A/B split 配置接口
- `ggml-cpu.c`：在 SVD vec 热路径中执行 rank 拆分
- `decode_svd_model.cpp`：把实验参数转成线程组配置并传给 ggml

## 9. 当前边界

当前版本明确的边界是：

1. 主要针对 decode 单 token vec 热路径。
2. 目前只把 A/B split 应用到 `x * V` 阶段。
3. 在当前 OpenMP 构建下，线程与核心组的对应关系是“静态线程槽位 + 外部 CPU 亲和设置”的组合，不是细粒度运行时调度器。

这版的价值在于先把“核心组分工 + rank 拆分 + 正确性 + 10% 速度约束”这件事稳住，再决定是否值得继续攻 `U` 阶段的真正双组并行归约。
