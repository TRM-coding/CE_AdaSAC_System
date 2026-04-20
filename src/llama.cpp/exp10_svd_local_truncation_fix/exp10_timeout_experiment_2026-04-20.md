# Exp10: 本地 A/B split 次要段超时丢弃实验

## 1. 本次改动

在现有 `exp10` 本地 A/B split 基础上，新增了一条超时逻辑：

- `x * V` 仍按 A/B 两组拆 rank
- 自动把 rank 较小的一组视为“次要部分”
- 若主要部分先完成，并且在 `minor_timeout_ms` 内次要部分仍未完成，则：
  - 丢弃次要部分对应的 `tmp` 切片
  - 只让主要部分线程继续后续 `tmp * U`

当前实验程序 [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp) 新增了可选参数：

- `minor_timeout_ms`

对应 ggml 代码修改点：

- [ggml-cpu.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cpu.h)
- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)

## 2. 实验环境

- 日期：`2026-04-20`
- 模型：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 可执行程序：`build-release-current/decode_svd_test`
- token 数：`1`
- 线程数：`8`
- CPU 资源：从 `60-79` 中选 `8` 个核心，实际使用 `60-67`
- 隔离方式：`sudo` + `cgroup v2 cpuset`

本次实际建立的 cgroup：

- run cgroup：`/sys/fs/cgroup/exp10_timeout_run_6067`
- load cgroup：`/sys/fs/cgroup/exp10_timeout_load_6063`

对应 cpuset：

- 测试进程：`60-67`
- 负载进程：`60-63`

## 3. 实验命令口径

### 3.1 空载 baseline

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0
```

### 3.2 空载 split

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off 0 60-63 64-67 0.25 0
```

语义：

- A 组：`60-63`
- B 组：`64-67`
- `share(A)=0.25`
- A 组是较小 rank 段，也就是超时逻辑里的“次要部分”

### 3.3 带负载 baseline

在 `60-63` 上先起 4 个 busy loop，然后跑 baseline：

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0
```

### 3.4 带负载 split

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off 0 60-63 64-67 0.25 0
```

### 3.5 带负载 timeout split

先扫描 `minor_timeout_ms = 1 / 2 / 5 / 10`，单次结果最好的点是 `2 ms`，然后做 3 次重复：

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off 0 60-63 64-67 0.25 2
```

## 4. 结果

### 4.1 空载

| 场景 | run1 | run2 | run3 | 平均 |
|---|---:|---:|---:|---:|
| baseline | `28.5757` | `31.4842` | `31.4405` | `30.5001 tok/s` |
| split | `28.9628` | `29.0307` | `29.4502` | `29.1479 tok/s` |

相对 baseline：

- split：`-4.43%`

### 4.2 带负载

| 场景 | run1 | run2 | run3 | 平均 |
|---|---:|---:|---:|---:|
| baseline + load | `8.52221` | `8.63419` | `8.54784` | `8.56808 tok/s` |
| split + load | `8.26420` | `8.58112` | `8.30847` | `8.38460 tok/s` |
| timeout split `2 ms` + load | `0.538878` | `0.532321` | `0.596747` | `0.55598 tok/s` |

相对 baseline + load：

- split：`-2.14%`
- timeout split `2 ms`：`-93.51%`

### 4.3 timeout 扫描

| timeout | 单次吞吐 | 输出 |
|---|---:|---|
| `1 ms` | `0.585298 tok/s` | `,` |
| `2 ms` | `0.594802 tok/s` | `,` |
| `5 ms` | `0.557947 tok/s` | `,` |
| `10 ms` | `0.534976 tok/s` | `,` |

## 5. 正确性

所有正式实验中，生成结果都仍然是：

- `Generated text: ,`

也就是说，这次超时逻辑没有把单 token 输出直接打坏，但性能表现明显退化。

## 6. 结论

本次实验可以给出一个明确结论：

1. “主要部分先完成、次要部分超时后直接丢弃”这条逻辑已经接进当前 `exp10` 本地 split 热路径，并且可以实际触发。
2. 在 `60-67` 这 8 个核心、`60-63` 带负载、`64-67` 相对空闲的实验口径下，当前 timeout 方案没有带来收益。
3. 当前最好的 timeout 扫描点是 `2 ms`，但即便如此，带负载吞吐仍只有 `0.55598 tok/s`，相比 baseline `8.56808 tok/s` 明显更慢。
4. 因此，这一版 timeout 丢弃方案在当前实现下不适合作为 `exp10` 的性能优化结论保留。

## 7. 结果目录

本轮结果目录：

- [exp10_timeout_rerun_20260420_6067](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/exp10_timeout_rerun_20260420_6067)

关键汇总：

- [summary.tsv](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/exp10_timeout_rerun_20260420_6067/summary.tsv)
