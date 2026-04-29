# Exp12 调度算法实现与本机实验报告

日期：2026-04-28

## 实现内容

根据 `algorithm.pdf` 第五章，实现了两类调度算法：

- 本地 DP 调度：逐层枚举 SVD 截断率，在主路径时延约束下最小化损失，并实现相邻两层不可同时裁剪约束。
- 超时预算分配：按 `weight = alpha_l * q_l` 比例把总补偿等待时间分配给被裁剪层，并输出可接入 runtime 的 `timeouts.txt`。
- 边端联合调度：先求本地 DP；若本地无解，则从后向前搜索分割点，选择满足总时限约束且卸载后层数量最少的方案。
- SVD 系统接入：输出 `rates.txt` 和 `timeouts.txt`，分别作为 `decode_svd_test` 的 per-layer SVD rate 与 per-layer minor timeout 输入。

主要文件：

- `scheduler.py`: 纯调度算法实现和 CLI。
- `run_exp12_local.py`: 本机实验驱动，不使用 adb。
- `README.md`: 使用方式。

## 实验设置

- 模型：`src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 程序：`build-release-current/decode_svd_test`
- token：`1`
- 电脑端核心：`60-67`
- 手机端模拟核心：`68-75`
- 加压核心：`60-63`
- cgroup：`/sys/fs/cgroup/exp12_algorithms_run`
- 说明：本轮没有使用 adb；手机端执行时间与传输时间来自本机测量后生成的 profile 模型。

实验命令：

```bash
python3 src/llama.cpp/exp12_algorithms/run_exp12_local.py \
  --use-cgroup \
  --out-dir src/llama.cpp/exp12_algorithms/results/cgroup_20260428
```

## 调度结果

结果文件：

- `results/cgroup_20260428/profile.json`
- `results/cgroup_20260428/schedule.json`
- `results/cgroup_20260428/rates.txt`
- `results/cgroup_20260428/experiment.json`

联合调度输出：

- 模式：`edge_end`
- 分割点：`m = 11`
- 电脑端保留层：`0-11`
- 手机端卸载层：`12-27`
- 估计总时延：`72.5993 ms`
- 电脑端前缀主路径：`52.9435 ms`
- 传输时延模型：`4.44 ms`
- 手机端后缀执行模型：`15.2158 ms`

生成的 SVD rate：

```text
0,0.75,0,0.75,0,0.75,0,0.75,0,0.75,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

可以看到 DP 遵守了“相邻层不能同时裁剪”：被裁剪层集中在 `1,3,5,7,9,11`，没有连续层同时裁剪。

## 实测结果

| 场景 | Decode-only throughput | generation_decode | 输出 |
|---|---:|---:|---|
| cgroup 空载 baseline | `23.2841 tok/s` | `42.9478 ms` | `,` |
| cgroup 加压 baseline | 异常值，已剔除 | 异常值，已剔除 | `,` |
| 调度 rate 文件实跑 | `30.4527 tok/s` | `32.8378 ms` | `,` |

调度 rate 文件实跑命令由脚本生成，核心参数等价于：

```bash
env LD_LIBRARY_PATH=./build-release-current/bin \
taskset -c 60,61,62,63,64,65,66,67 \
./build-release-current/decode_svd_test \
  ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf \
  1 8 0 off src/llama.cpp/exp12_algorithms/results/cgroup_20260428/rates.txt \
  60,61,62,63 64,65,66,67 0.5 0
```

## 结论

第五章的调度策略已经接到当前 SVD 系统：算法能输出本地截断率文件，并可被 `decode_svd_test` 正常消费。高负载 profile 下，本地 DP 无解后会进入边端联合调度，选择最少后层卸载方案；本轮选择从第 `12` 层开始卸载。

本轮“手机端”仍是本机时延模型，没有启动 adb，也没有真实远端推理。因此报告中的联合总时延用于验证调度算法和接口闭环；真实手机端吞吐、传输和量化精度还需要后续接入 adb/网络 server 后再测。
