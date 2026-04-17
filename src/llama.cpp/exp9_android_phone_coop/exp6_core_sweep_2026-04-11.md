# 2026-04-11 协同卸载核心数 Sweep

## 实验目的

在现有 `decode_svd_test + svd_mobile_server` 协同卸载实现上，固定手机端为 `8` 核，比较电脑端 `8 / 6 / 4 / 2` 核时的联合推理吞吐。

## 绑核与 cgroup

- cgroup: `/sys/fs/cgroup/tianruiming-exclusive`
- `cpuset.cpus = 60-79`
- shell 验证：

```text
shell_pid=616382
pid 616382's current affinity list: 60-79
0::/tianruiming-exclusive
```

本次实验中：

- 手机端服务进程固定 `60-67`
- 电脑端 decode 进程分别使用：
  - `68-75` (`8` 核)
  - `68-73` (`6` 核)
  - `68-71` (`4` 核)
  - `68-69` (`2` 核)

## 执行脚本

- 脚本：`/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/run_coop_core_sweep.sh`
- 结果目录：`/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659`

脚本默认参数：

- model: `src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`
- offload rate: `1.0`
- 手机端线程数: `8`
- 生成 token 数: `8`

## 结果汇总

| 配置 | 电脑端核 | 手机端核 | Decode-only throughput | End-to-end throughput |
| --- | --- | --- | --- | --- |
| `pc8_phone8` | `68-75` | `60-67` | `7.23584 tok/s` | `7.19883 tok/s` |
| `pc6_phone8` | `68-73` | `60-67` | `7.23423 tok/s` | `7.19734 tok/s` |
| `pc4_phone8` | `68-71` | `60-67` | `6.50550 tok/s` | `6.47594 tok/s` |
| `pc2_phone8` | `68-69` | `60-67` | `5.59587 tok/s` | `5.57494 tok/s` |

## 观察

1. `8 -> 6` 核几乎没有吞吐变化
   - `7.23584 -> 7.23423 tok/s`
   - 说明当前联合推理瓶颈不在电脑端第 `7-8` 个核

2. 电脑端降到 `4` 核后开始明显掉速
   - `7.23 -> 6.51 tok/s`
   - FFN 之外的本地阶段和 RPC 等待已不足以完全掩盖电脑端本地计算量

3. 电脑端降到 `2` 核后进一步退化到 `5.60 tok/s`
   - 此时电脑端已经成为明显短板

4. 手机端远端执行时间也随电脑端核数变化而变化
   - `pc8_phone8`: `remote_compute_total=759.173 ms`
   - `pc6_phone8`: `remote_compute_total=755.940 ms`
   - `pc4_phone8`: `remote_compute_total=814.028 ms`
   - `pc2_phone8`: `remote_compute_total=830.509 ms`
   - 这说明即使手机端固定在 `60-67`，两端仍会通过共享机器资源发生干扰

## 结论

- 在当前实现和这台机器上，`电脑端 6 核 + 手机端 8 核` 与 `电脑端 8 核 + 手机端 8 核` 基本等价。
- 若要在不明显损失吞吐的前提下给电脑端省核，`6 + 8` 是更稳妥的点。
- `4 + 8` 和 `2 + 8` 都已经出现明显性能下降，不适合作为当前默认配置。

## 日志文件

- summary: [summary.tsv](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/summary.tsv)
- `pc8_phone8`: [client](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc8_phone8_client.log), [server](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc8_phone8_server.log)
- `pc6_phone8`: [client](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc6_phone8_client.log), [server](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc6_phone8_server.log)
- `pc4_phone8`: [client](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc4_phone8_client.log), [server](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc4_phone8_server.log)
- `pc2_phone8`: [client](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc2_phone8_client.log), [server](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/20260411_141659/pc2_phone8_server.log)
