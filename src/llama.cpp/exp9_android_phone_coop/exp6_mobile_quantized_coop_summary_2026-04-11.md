# 2026-04-11 手机端量化协同推理开发与研究总报告

> 2026-04-16 校正：
> 本文是 `2026-04-11` 阶段性总结，不应再直接当作当前部署说明。
> 当前权威构建和运行口径以 [build-rel/README.md](/home/tianruiming/CE_ADA_LLAMA/build-rel/README.md) 与 [build-rel/android-build-notes.md](/home/tianruiming/CE_ADA_LLAMA/build-rel/android-build-notes.md) 为准。
> 之后确认过一个关键问题：如果 Android 端 `svd_mobile_server` 没有和最新 `ggml-cpu.c / ggml-svd-offload.c / llama-graph.cpp` 一起重编并重新推送，远端 FFN 会异常变慢；这不是手机硬件本身的问题。
> 此外，TCP 方式连接的 `adb` 设备不应默认假设 `127.0.0.1:7788` 一定可用，必要时应直接连接手机的可达 IP:port，并在服务端显式使用 `svd` executor。
> 2026-04-18 又确认了一个后续修复：Windows 侧 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf` 之前没有命中 `Q4_0` SVD vec 快路径，原因是 `ggml_compute_forward_mul_mat_svd_vec()` 仅放行 `F32/F16` 的 `U/V`。该问题现已修复，实测见 [exp9_windows_q4_svd_fastpath_2026-04-18.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp9_android_phone_coop/exp9_windows_q4_svd_fastpath_2026-04-18.md)。若要在 Windows + Android 协同链路中吃到这一修复，Android 端也必须用包含同一补丁的 `svd_mobile_server` 重编并重新部署。
> 同日新增的真机对照还表明：Android 本地 `mobile_q4_0` 明显慢于 `llama_quant_q4_0`，因此 `mobile_*` 模型不再应被视为当前默认推荐部署模型。默认口径应优先使用 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`。

## 1. 背景与当前目标

当前实验链路的目标是：

- 电脑端继续使用非量化 SVD 模型做主推理
- 手机端负责协同执行 SVD 尾部 rank
- 手机端执行时尽量使用量化模型，减小内存占用并提升远端执行效率
- 手机端完成计算后，仍以 `float` 结果通过现有网络接口回传电脑端

本次总报告整理的是截至 `2026-04-11` 的当前可用方案、代码实现、实验方法与结论。

## 2. 当前可用方案

### 2.1 推荐的当前方案

当前最稳妥、已经实际跑通的方案是：

- 电脑端模型：
  - `qwen.gguf.sort_svd.compact.gguf`
- 手机端模型：
  - `qwen.gguf.sort_svd.compact.mobile_q8_0.gguf`

对应含义：

- 电脑端：非量化 compact SVD 模型
- 手机端：离线量化后的手机专用 SVD 模型

### 2.2 为什么推荐这套方案

此前也做过“手机端运行时懒量化”的版本，即：

- 手机端先加载非量化 SVD 模型
- 首次命中某个 tail executor 时，再现场把 `U/V tail` 量化成缓存

这条路径功能上是成立的，但首次量化缓存构建代价很高，典型数据达到秒级，不适合作为默认方案。

离线量化手机端模型后：

- 手机端启动后直接加载量化 SVD 权重
- 不再需要首次请求时现场量化
- 协同链路的启动和 steady-state 都明显更合理

## 3. 当前代码改动

### 3.1 服务端量化执行支持

修改文件：

- [svd_mobile_server.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_mobile_server.cpp)

当前服务端支持两种模式：

1. 运行时量化模式
   - 启动参数第 4 位传 `q8_0 / q4_0 / q4_k / ...`
   - 服务端会在创建 executor 时把当前 tail 张量量化成缓存
2. 离线量化模型模式
   - 手机端直接加载已经量化好的 GGUF
   - 启动参数传 `off`
   - 服务端直接对量化张量执行，不再现场量化

无论哪种模式：

- 服务端最终仍回传 `float` 输出
- 电脑端协议和客户端逻辑都无需修改

### 3.2 sweep 脚本补充

修改文件：

- [run_coop_core_sweep.sh](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/run_coop_core_sweep.sh)

新增环境变量：

- `SERVER_MODEL_PATH`
- `SERVER_QUANT_MODE`

因此现在可以：

- 单独指定手机端模型
- 单独指定手机端量化模式

## 4. 离线量化手机端专用 SVD 模型

### 4.1 新增脚本

脚本：

- [quantize_mobile_svd.py](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/quantize_mobile_svd.py)

作用：

- 输入 compact SVD GGUF
- 只量化手机端真正需要的 SVD tensor：
  - `ffn_up_svd_u`
  - `ffn_up_svd_v`
  - `ffn_gate_svd_u`
  - `ffn_gate_svd_v`
  - `ffn_down_svd_u`
  - `ffn_down_svd_v`
- 其余 tensor 原样保留
- 重写 `general.file_type`
- 输出手机端专用 GGUF

### 4.2 本次实际生成结果

实际执行命令：

```bash
python /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp7_generate_sort_svd/quantize_mobile_svd.py \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.mobile_q8_0.gguf \
  --quant q8_0
```

产物：

- [qwen.gguf.sort_svd.compact.mobile_q8_0.gguf](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.mobile_q8_0.gguf)

脚本输出：

```text
total_tensors: 422
quantized_svd_tensors: 168
kept_original_tensors: 254
input_tensor_bytes: 6192551936
output_tensor_bytes: 2214483968
```

文件体积：

- 原始 compact SVD：`5.8G`
- 手机端专用 `mobile_q8_0`：`2.1G`

### 4.3 自检结果

读取 GGUF 后确认：

- `general.file_type = 7`，即 `MOSTLY_Q8_0`
- SVD tensor 如 `blk.0.ffn_down_svd_u.weight`、`blk.0.ffn_down_svd_v.weight` 已变成 `Q8_0`
- 非 SVD tensor 如 `blk.0.attn_norm.weight` 仍保持原始类型

## 5. 验证方法

### 5.1 客户端与服务端角色

- 客户端：
  - `decode_svd_test`
  - 加载非量化 `compact` SVD 模型
- 服务端：
  - `svd_mobile_server`
  - 加载手机端专用离线量化 SVD 模型

### 5.2 当前测试 prompt

当前 `decode_svd_test` 里 prompt 是写死的：

- `Once upon a time`

代码位置：

- [decode_svd_model.cpp:160](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp:160)

### 5.3 cgroup 绑核方法

当前测试使用：

- cgroup：`/sys/fs/cgroup/tianruiming-exclusive`
- 父组 cpuset：`60-79`

为了稳定跑出 `8 / 6 / 4 / 2 / 1` 核电脑端配置，本次采用的是：

- 父组：`60-79`
- 子组切分：
  - 手机端：`60-67`
  - 电脑端：
    - `68-75`
    - `68-73`
    - `68-71`
    - `68-69`
    - `68`

这是因为当前会话里直接用 `taskset -c 60-67` 会报 `Invalid argument`，但通过 cpuset 子组分配后，进程亲和性可以稳定落到目标核心集合。

### 5.4 Android 真机测速口径

手机端吞吐对温度和调度策略非常敏感。协同推理、SVD 本地 decode、官方 dense Q4_0 baseline 必须使用同一套真机测速口径，否则 `40 tok/s` 和 `55 tok/s` 之间的差异可能只是实验环境差异。

推荐真机准备流程：

```bash
adb reboot
adb wait-for-device
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q 1; do sleep 1; done

# InfraPowerTest 的冷机实验会等温度接近 25.0 C。
adb shell 'su -c "cat /sys/class/power_supply/battery/temp"'

adb shell 'su -c "sync && echo 3 > /proc/sys/vm/drop_caches"'
adb shell 'su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'
adb shell 'su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"'
adb shell 'su -c "grep . /sys/devices/system/cpu/cpufreq/policy*/scaling_governor"'
```

推荐 CPU baseline 命令形态：

```bash
adb shell 'su -c "cd /data/local/tmp/CE_Ada && \
  LD_LIBRARY_PATH=/data/local/tmp/CE_Ada \
  taskset -a ff ./llama-bench-fastflags \
    -m /data/local/tmp/CE_Ada/qwen.q4_0.gguf \
    -pg 1,512 \
    -r 2 \
    -t 8 \
    --prio 3 \
    --cpu-mask 0xff \
    --cpu-strict 1"'
```

注意事项：

- `taskset -a ff`、`--cpu-mask 0xff`、`--cpu-strict 1`、`--prio 3` 需要同时记录。
- `-pg 1,512 -r 2` 与裸跑 `-p 0 -n 256` 不是同一测速口径。
- `policy0/policy6` 若仍是 `walt`，结果通常不能和 `performance` governor 下的冷机结果直接比较。
- 之前在同一台 OnePlus 15 上，口径对齐后 dense Q4_0 可到约 `52.5 tok/s`；设备升温到约 `30 C` 后，同类命令可能掉回高 `30 tok/s` 区间。
- 因此，协同推理结果必须同时记录温度、governor、CPU mask、线程数、优先级、token 数和模型文件名。

## 6. 关键实验与结果

### 6.1 运行时量化方案的结论

文档：

- [exp6_quantized_mobile_offload_report_2026-04-11.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp9_android_phone_coop/exp6_quantized_mobile_offload_report_2026-04-11.md)

主要结论：

- 手机端运行时量化执行是可行的
- 输出文本正常，不是乱码
- 但首次 executor miss 时，现场量化 tail 权重代价极高

典型数据：

- `create_total = 7592.62 ms`
- `remote_compute_total = 357.517 ms`

因此这条路径只证明了“功能正确”，不适合作为默认部署方法。

### 6.2 离线手机端量化模型验证

文档：

- [exp6_mobile_quantized_svd_model_report_2026-04-11.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/exp6_mobile_quantized_svd_model_report_2026-04-11.md)

主要结论：

- `mobile_q8_0` 模型可以直接被 `svd_mobile_server` 加载
- 客户端协同生成输出正常
- 不再有运行时懒量化的多秒级首次缓存构建开销

### 6.3 `60-79` cgroup 下单次测速

结果目录：

- [mobile_q8_cgroup_bench_20260411_152703](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/mobile_q8_cgroup_bench_20260411_152703)

配置：

- 电脑端：8 核
- 手机端：8 核
- prompt：`Once upon a time`
- generation：`24` token

结果：

- `Decode-only throughput = 12.524 tok/s`
- `End-to-end throughput = 12.4124 tok/s`

输出文本：

```text
Once upon a time, there was a young girl named Lily. She lived in a small village with her parents and two younger siblings. Lily
```

### 6.4 `8 / 6 / 4 / 2 / 1` 核电脑端 sweep

结果目录：

- [mobile_q8_core_sweep_20260411_153007](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/mobile_q8_core_sweep_20260411_153007)

汇总：

| 配置 | 电脑端核 | 电脑端亲和性 | Decode-only throughput | End-to-end throughput |
| --- | --- | --- | --- | --- |
| `pc8_phone8` | `8` | `68-75` | `14.7892 tok/s` | `14.7143 tok/s` |
| `pc6_phone8` | `6` | `68-73` | `10.8006 tok/s` | `10.7347 tok/s` |
| `pc4_phone8` | `4` | `68-71` | `9.00364 tok/s` | `8.96065 tok/s` |
| `pc2_phone8` | `2` | `68-69` | `7.69039 tok/s` | `7.66318 tok/s` |
| `pc1_phone8` | `1` | `68` | `4.27535 tok/s` | `4.26997 tok/s` |

所有配置的输出文本都正常，均不是乱码。

服务端远端执行总时间：

- `pc8_phone8`：`879.509 ms`
- `pc6_phone8`：`1303.48 ms`
- `pc4_phone8`：`1696.9 ms`
- `pc2_phone8`：`1756.73 ms`
- `pc1_phone8`：`2980.59 ms`

## 7. 当前研究结论

截至目前，可以明确得到下面这些结论。

### 7.1 功能层面

已经证明：

1. 手机端量化推理可行
2. 电脑端非量化推理可行
3. 两端混合协同部署可行
4. 网络协议不需要改
5. 输出文本正常，不是乱码

### 7.2 部署层面

当前最佳实践不是“手机端运行时懒量化”，而是：

- 离线生成手机端专用量化 SVD 模型
- 服务端直接加载该量化模型
- 客户端仍使用原始非量化 compact SVD 模型

### 7.3 性能层面

当前数据表明：

- 手机端专用离线量化模型显著减少了模型体积
- 也避免了运行时首次量化缓存构建的秒级抖动
- 在这套新方法下，电脑端核数越少，联合吞吐越明显下降

与此前旧实验不同，本轮离线量化方案没有出现“8 核和 6 核几乎等价”的现象，而是表现出更清晰的电脑端核数瓶颈。

## 8. 当前残留问题

虽然当前方案已经可用，但还有几个明确的后续点。

### 8.1 prompt 仍是写死的

当前 `decode_svd_test` 仍把 prompt 固定成：

- `Once upon a time`

如果要做更系统的精度与可用性验证，建议把 prompt 改成命令行参数。

### 8.2 目前只系统验证了 `mobile_q8_0`

脚本支持更多量化类型，但当前真正完整生成并做协同验证的是：

- `q8_0`

后续值得继续评估：

- `q4_k`
- `q5_k`
- `q6_k`

这些类型可能在模型体积和远端速度之间给出更优折中。

### 8.3 仍缺正式汇总脚本

虽然现在已经有：

- 单次测速
- 核数 sweep
- 离线量化模型生成脚本

但还没有一条统一脚本把：

- 生成手机端模型
- 启动服务端
- 跑 cgroup sweep
- 汇总结果

串成完全自动化流程。

## 9. 当前建议

如果接下来继续推进，我建议优先做这三件事。

1. 把 `decode_svd_test` 的 prompt 改成命令行参数
2. 用同样方法生成并测试 `mobile_q4_k`、`mobile_q5_k`、`mobile_q6_k`
3. 新增一个“一键生成手机端量化模型并完成 cgroup sweep”的实验脚本

## 10. 关联文档

本总报告对应的关键子文档如下：

- [svd_interface_callchain.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/svd_interface_callchain.md)
- [exp6_cooperative_offload_followup_2026-04-07.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp9_android_phone_coop/exp6_cooperative_offload_followup_2026-04-07.md)
- [exp6_quantized_mobile_offload_report_2026-04-11.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp9_android_phone_coop/exp6_quantized_mobile_offload_report_2026-04-11.md)
- [exp6_mobile_quantized_svd_model_report_2026-04-11.md](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/exp6_mobile_quantized_svd_model_report_2026-04-11.md)
