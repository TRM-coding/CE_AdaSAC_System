# Exp10: 超时丢弃与协同卸载调试实验记录

## 1. 本次实验目标

本轮实验要验证一件很具体的事：

- 当本地主要部分先完成后，如果远端卸载部分在给定时间内还没完成，是否真的会被直接丢弃
- 在这种“直接丢弃远端尾部”的语义下，速度和输出会如何变化

实验过程中还额外排查了两个实现问题：

1. `timeout=0` 是否真的代表“直接不等”
2. 客户端主动放弃远端请求时，服务端是否会被异常打死

## 2. 发现的问题

### 2.1 早期实验口径有误

本轮一开始有一条“全层 `80%` 卸载到高负载远端，但输出仍保持正常文本”的结果。后续排查确认，这条结果不能直接作为正确结论，原因有两个：

1. 当时命令少传了最后一个参数，实际跑的是：
   - `svd_offload_timeout_ms = 2`
   - 不是预期的 `0`
2. 旧实现中，即便显式传 `timeout=0`，客户端仍会先发远端请求，再在本地算完后 abort，这不等价于“直接丢弃远端尾部”

### 2.2 服务端会被客户端 abort 打死

进一步调试发现：

- 客户端主动放弃请求后会关闭 socket
- 服务端 [svd_mobile_server.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp) 的 `send_all()` 原先直接用 `send(..., 0)`
- 在 Linux 下，这有触发 `SIGPIPE` 的风险

结果是：

- 服务端可能在第一次返回结果时被打死
- 后续层全部变成 `connect_failed`
- 客户端又会回退到“本地完整计算”

这就是为什么早期某些“timeout=0 但输出仍正常”的结果实际上不可信。

### 2.3 脏环境导致过慢假象

中途还出现过一组异常慢到 `0.16 tok/s` 左右的结果。后续检查系统负载与 cgroup 发现：

- 前一轮 `4` 核 `100%` 负载实验的 `stress-ng` 还残留在 `60-63`
- 而那一轮客户端实验误用了与残留负载重叠的核

所以那组结果是脏环境造成的，不是当前实现真实表现。

## 3. 本次代码修复

### 3.1 协同卸载请求支持就绪检测与主动放弃

修改文件：

- [ggml-svd-offload.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.h)
- [ggml-svd-offload.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.c)

新增能力：

- `ggml_svd_offload_wait_ready(...)`
- `ggml_svd_offload_abort_request(...)`
- `ggml_svd_offload_get_timeout_ms()`

目的：

- 本地部分算完后，不再无条件进阻塞式 `finish_request`
- 可以先探测远端响应是否已就绪
- 未就绪时主动丢弃这次远端请求

### 3.2 Linux 下避免 `SIGPIPE`

修改文件：

- [ggml-svd-offload.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-svd-offload.c)
- [svd_mobile_server.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp)

修复方式：

- `send()` 增加 `MSG_NOSIGNAL`

目的：

- 客户端主动断开后，服务端不会因为写回结果时触发 `SIGPIPE` 而整个进程退出

### 3.3 `timeout=0` 的语义修正

修改文件：

- [ggml-cpu.c](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c)

现在的语义是：

- 如果是 partial offload
- 且 `svd_offload_timeout_ms == 0`
- 则直接把远端尾部视为“已丢弃”
- 这一层不再发远端请求
- 只保留本地 `k_keep` 前缀 rank 计算

这版语义才真正对应：

- “直接不等”
- “超时后远端部分直接丢弃”

### 3.4 测试程序不再写死 `3000 ms`

修改文件：

- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp)

修复前：

- `svd_offload_timeout_ms` 被硬编码成 `3000`

修复后：

- 支持显式传参
- 默认值改为 `2 ms`

也就是说：

- 最后一个参数传 `0` 才真的表示“直接不等”

## 4. 关键验证实验

### 4.1 全层 `80%`，`timeout=0`，高负载远端

实验口径：

- 远端：`60-63`
- 远端背景负载：`stress-ng --cpu 4 --cpu-load 100`
- 本地：`64-69`
- 所有层都传 `0.8`
- `timeout=0`

结果：

| 配置 | Decode-only throughput | 输出 |
|---|---:|---|
| 全层 `80%`，`timeout=0` | `12.183 tok/s` | `,plenFUFUFUFUFU ` |

结论：

- 修复后，`timeout=0` 的确会让远端尾部被真正丢弃
- 输出不再是正常文本，而是明显退化
- 说明当前实现语义终于和实验预期一致

### 4.2 F16 SVD 模型可运行性验证

本轮额外验证了未量化的 F16 SVD 模型：

- 模型：`qwen.gguf.sort_svd.compact.gguf`

#### 4.2.1 本地 baseline

实验口径：

- 本地：`64-69`
- 线程：`6`
- 不卸载

结果：

| 配置 | Decode-only throughput | 输出 |
|---|---:|---|
| F16 本地 baseline | `9.25092 tok/s` | `, there was a little girl named Lily` |

结论：

- F16 SVD 模型在当前代码下可以正常加载、正常 decode
- 不需要为“能否运行”这件事额外补 F16 专属修复

#### 4.2.2 F16 协同 + `timeout=0`

实验口径：

- 模型：`qwen.gguf.sort_svd.compact.gguf`
- 卸载端：`60-61`
- 本地端：`64-69`
- 只在偶数层做 `80%` 卸载
- `timeout=0`

结果：

| 配置 | Decode-only throughput | 输出 |
|---|---:|---|
| F16 隔层 `80%`，`timeout=0` | `12.232 tok/s` | `,today, The you have been,` |

结论：

- 今天修好的 timeout/offload 语义在 F16 SVD 模型上也能直接工作
- F16 下同样表现为：
  - 速度提升
  - 输出明显退化

### 4.3 SVD 专用 perplexity 工具

为了不污染原始 `llama.cpp` 的 perplexity 工具，本轮新增了一份独立副本：

- [perplexity_svd.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/perplexity_svd.cpp)

对应可执行程序：

- `build-release-current/perplexity_svd_test`

这份工具基于官方 `examples/perplexity/perplexity.cpp` 拷贝而来，只改了两部分：

1. 新增 SVD 参数解析：
   - `--svd-offload-endpoint`
   - `--svd-offload-rates`
   - `--svd-offload-timeout`
2. 在创建 `llama_context` 时，把这些参数写入当前 SVD 推理链路

也就是说：

- 原始 perplexity 工具保持不变
- `exp10` 下这份副本可以直接评估你当前的 SVD 卸载 / 本地截断链路

## 5. F16 困惑度实验

### 5.1 实验口径

模型：

- `qwen.gguf.sort_svd.compact.gguf`

工具：

- `perplexity_svd_test`

语料：

- [ppl_corpus_qwen_out_64k.txt](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/ppl_corpus_qwen_out_64k.txt)

说明：

- 该语料由 `qwen_out.txt` 截取前 `64 KiB` 得到
- 为了控制实验时间，perplexity 评估统一使用：
  - `--ctx-size 512`
  - `--chunks 2`
  - `6` 个本地线程

本轮对比的是：

- `0%`：本地 baseline
- `20% / 40% / 60% / 80%`：把同一卸载率写到全部 28 层

当前这组 perplexity 实验使用的是“本地截断语义”：

- 只传 `--svd-offload-rates`
- 不传 `--svd-offload-endpoint`

这样做的原因是：

- 对于 `timeout=0` 的链路，当前实现已经把 partial offload 解释成“直接丢弃尾部，不发远端请求”
- 在 perplexity 评估中，这和“本地只保留前缀 rank”是等价的
- 也更适合稳定比较不同 rate 对模型质量的影响

### 5.2 结果

| rate | PPL |
|---|---:|
| `0%` | `2.6782` |
| `20%` | `41.6902` |
| `40%` | `1421.1742` |
| `60%` | `17638.0399` |
| `80%` | `18159.2016` |

### 5.3 结论

这组 perplexity 结果说明：

1. F16 SVD baseline 在这份固定语料上的 PPL 很低，约为 `2.6782`
2. 只要开始做全层截断，PPL 会迅速恶化
3. 从 `20%` 到 `40%`，质量损失已经非常剧烈
4. `60%` 和 `80%` 都已经进入极端退化区间，PPL 非常高

这和今天前面的 decode 文本实验是一致的：

- 截断 / 丢弃尾部越多，速度越快
- 但模型质量下降也越快

### 5.4 流式 decode + 每层 `1 ms` 超时

为了真正命中当前单 token SVD decode 热路径，本轮又把 [perplexity_svd.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/perplexity_svd.cpp) 改成了：

- token-by-token streaming decode
- 不再走原始 perplexity 的多 token batch / prefill 评估路径

实验口径：

- 模型：`qwen.gguf.sort_svd.compact.gguf`
- 语料：`ppl_corpus_qwen_out_64k.txt`
- `--ctx-size 512`
- `--chunks 1`
- 本地端：`64-69`
- 远端：`60-61`
- 每层超时预算：`1 ms`

结果如下：

| 卸载率 | 时间 | PPL |
|---|---:|---:|
| `0%` | `56.63 s` | `3.9338` |
| `20%` | `129.80 s` | `20.4561` |
| `40%` | `116.87 s` | `47.6030` |
| `60%` | `118.10 s` | `79.2953` |
| `80%` | `119.46 s` | `306.1602` |

结论：

1. 在真正命中单 token SVD 热路径后，`1 ms` 超时会显著改变 PPL。
2. 卸载率越高，PPL 恶化越明显。
3. 这组实验里，所有协同档位都比 `0%` baseline 更慢，没有体现时间收益。
4. 因此当前“每层 `1 ms` 超时”策略在 streaming decode 困惑度场景下，并不是一个好的速度/质量折中点。

## 6. 今日正式实验

### 5.1 实验口径

本次最终可信实验采用：

- 卸载端：`60-61`
- 卸载端背景负载：`stress-ng --cpu 2 --cpu-load 20`
- 本地端：`64-69`
- 本地端无背景负载
- decode 长度：`8 tokens`
- `timeout=0`
- 只在偶数层做卸载：`0,2,4,...,26`

这意味着：

- 远端尾部不会真的计算后再回传
- 而是直接被视为丢弃

### 5.2 `6` 核纯本地 baseline

| 配置 | Decode-only throughput | 输出 |
|---|---:|---|
| `6` 核本地全算，不卸载 | `26.0236 tok/s` | `, there was a little girl named Sally` |

### 5.3 隔层卸载结果

| 配置 | Decode-only throughput | 输出 |
|---|---:|---|
| 隔层卸载 `20%` | `27.2126 tok/s` | `, there was a boy named Joey who` |
| 隔层卸载 `40%` | `27.6964 tok/s` | `, I was bored with my life.` |
| 隔层卸载 `60%` | `29.0485 tok/s` | `, Batman Batman Batman Batman Batman Batman Batman` |
| 隔层卸载 `80%` | `29.8106 tok/s` | `, Billy was was was was was to` |

### 5.4 相对 baseline 变化

以 `6` 核纯本地 `26.0236 tok/s` 为基线：

| 配置 | 相对 baseline |
|---|---:|
| 隔层卸载 `20%` | `+4.57%` |
| 隔层卸载 `40%` | `+6.43%` |
| 隔层卸载 `60%` | `+11.62%` |
| 隔层卸载 `80%` | `+14.55%` |

## 7. 结论

今天这轮实验可以下几个明确结论：

1. 之前“timeout=0 但输出仍正常”的现象不可信，原因是：
   - 命令口径错误
   - 服务端被 `SIGPIPE` 打死
   - `timeout=0` 语义实现不对
2. 这些问题已经修复：
   - `timeout=0` 现在表示“直接丢弃远端尾部，不发请求”
   - 服务端不会再被客户端 abort 直接打死
3. 修复后的行为是自洽的：
   - 速度会提升
   - 输出会明显退化
4. 在“`2` 核远端 `20%` 背景负载 + `6` 核本地空闲”的今天正式口径下：
   - 卸载比例越高，速度越快
   - 但文本质量退化也越明显
5. 所以当前这条路径的本质不是“在质量不变下白赚速度”，而是：
   - 用直接丢弃远端尾部换更快的本地前缀计算
6. F16 perplexity 实验进一步从整体语料层面验证了这一点：
   - rate 越高，PPL 恶化越快
   - `40%` 以上已经不是“轻微质量波动”，而是非常明显的整体退化
7. 当把 perplexity 工具改成真正命中单 token SVD 热路径的 streaming decode 后，
   - `1 ms` 超时策略会让 PPL 明显恶化
   - 同时总评估时间反而比 `0%` baseline 更长
   - 说明当前真实协同路径仍存在很重的固定开销

## 8. 结果目录

今天的关键结果目录：

- [all_layers_80_server4_load100_20260420](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/all_layers_80_server4_load100_20260420)
- [every_other_offload_load20_clean_20260420_6069](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/every_other_offload_load20_clean_20260420_6069)
- [perplexity_svd_f16_20260420](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420)

关键日志：

- [all_layers_80.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/all_layers_80_server4_load100_20260420/all_layers_80.log)
- [baseline_6core_no_offload_6469.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/every_other_offload_load20_clean_20260420_6069/baseline_6core_no_offload_6469.log)
- [rate20.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/every_other_offload_load20_clean_20260420_6069/rate20.log)
- [rate40.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/every_other_offload_load20_clean_20260420_6069/rate40.log)
- [rate60.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/every_other_offload_load20_clean_20260420_6069/rate60.log)
- [rate80.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/every_other_offload_load20_clean_20260420_6069/rate80.log)
- [f16_rate0.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420/f16_rate0.log)
- [f16_rate_0.2.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420/f16_rate_0.2.log)
- [f16_rate_0.4.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420/f16_rate_0.4.log)
- [f16_rate_0.6.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420/f16_rate_0.6.log)
- [f16_rate_0.8.log](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420/f16_rate_0.8.log)
- [summary.tsv](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp10_svd_local_truncation_fix/results/perplexity_svd_f16_20260420/summary.tsv)
