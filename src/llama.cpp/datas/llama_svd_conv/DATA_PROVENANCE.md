# llama_svd_conv 数据说明

本文档说明 `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv` 目录下两张图的数据来源，以及如果需要重新测量，应当如何重新运行。

## 当前目录内容

- `operator_performance_comparison.png`
  算子性能对比图
- `model_inference_comparison.png`
  模型推理速度对比图
- `generate_perf_charts.py`
  生成上述两张图的脚本
- `IM2COL_EXPERIMENT.md`
  llama.cpp 自带 im2col 卷积路径的单独实验记录
- `im2col_experiment_results.json`
  上述 im2col 实验的结构化数据

## 图中数据来自哪里

当前图中使用的数据，来自串行 benchmark 的最终结果，并已经写入：

- `generate_perf_charts.py`
- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp8_resnet50/EXPERIMENT_REPORT_fold_unfold.md`

也就是说：

- 图的最终数值目前是“写死”在 `generate_perf_charts.py` 里的
- 如果重新测量了性能，需要同时更新：
  - `generate_perf_charts.py`
  - `EXPERIMENT_REPORT_fold_unfold.md`

## 当前采用的数据口径

为了避免 benchmark 之间互相抢占资源，当前图中所有数据都使用以下口径：

- 串行执行
- 不并发运行多个 benchmark
- CPU 绑定到 `60-79`
- `threads=8`

模型级 benchmark 采用：

- `warmup=5`
- `repeat=20`

## 模型推理时间图的数据来源

第二张图 `model_inference_comparison.png` 的三组数据分别来自下面三个命令。

### 1. llama.cpp 官方推理

对应可执行程序：

- `./build-release-current/run_resnet50_ggml`

对应模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf`

命令：

```bash
taskset -c 60-79 ./build-release-current/run_resnet50_ggml \
    --model /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --top-k 1 \
    --benchmark \
    --warmup 5 \
    --repeat 20
```

### 2. PyTorch 前向传播

对应脚本：

- `src/llama.cpp/exp8_resnet50/benchmark_pytorch_resnet50_forward.py`

对应模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model`

命令：

```bash
taskset -c 60-79 /home/tianruiming/miniconda3/envs/pytorch/bin/python \
    src/llama.cpp/exp8_resnet50/benchmark_pytorch_resnet50_forward.py \
    --model-dir /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --warmup 5 \
    --repeat 20
```

### 3. 结合 oneDNN 的前向传播

对应可执行程序：

- `./build-release-current/run_resnet50`

对应模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf`

命令：

```bash
taskset -c 60-79 ./build-release-current/run_resnet50 \
    --model /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --top-k 1 \
    --benchmark \
    --warmup 5 \
    --repeat 20
```

## 算子性能图的数据来源

第一张图 `operator_performance_comparison.png` 的数据来自典型卷积尺寸的单独 benchmark。

### 1. llama.cpp 官方 conv 路径 + 我们 oneDNN conv 路径

对应可执行程序：

- `./build-release-current/bench_conv_ops`

命令：

```bash
taskset -c 60-79 ./build-release-current/bench_conv_ops
```

这个程序会输出三类卷积尺寸下的三组 llama.cpp / oneDNN 数据：

- `ggml_ms`
- `onednn_ms`：oneDNN execute-only，权重 reorder 和 primitive 创建不计入计时
- `onednn_full_call_ms`：oneDNN 完整调用口径，memory desc、primitive desc、权重 reorder、dst 分配、primitive 创建、execute 和 stream wait 都计入计时

当前选取的卷积尺寸是：

- `7x7 stem`
- `3x3 block`
- `1x1 bottleneck`

### 2. PyTorch 官方 conv 算子

当前使用的是单独的 PyTorch `Conv2d` benchmark，命令如下。为了让旧图更接近“大家都包含算子/框架调用开销”的公平口径，`operator_performance_comparison.png` 现在使用 `onednn_full_call_ms`，不再使用 execute-only 的 `onednn_ms`。

```bash
taskset -c 40-59 /home/tianruiming/miniconda3/envs/pytorch/bin/python - <<'PY'
import time, statistics, torch
from torch import nn

torch.set_num_threads(8)
torch.set_num_interop_threads(1)

def bench(name, ic, oc, ih, iw, kh, kw, stride, padding, warmup=5, repeat=20):
    conv = nn.Conv2d(ic, oc, (kh, kw), stride=stride, padding=padding, bias=True).eval()
    x = torch.randn(1, ic, ih, iw)
    with torch.inference_mode():
        for _ in range(warmup):
            conv(x)
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            conv(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    print(f"{name}\tpytorch_ms={statistics.median(times):.6f}")

bench("7x7 stem", 3, 64, 224, 224, 7, 7, 2, 3)
bench("3x3 block", 128, 128, 28, 28, 3, 3, 1, 1)
bench("1x1 bottleneck", 256, 1024, 14, 14, 1, 1, 1, 0)
PY
```

重测后的公平口径数据另存于：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/operator_fair_conv_results.json`

## 如果要重新画图，怎么做

### 步骤 1：重新测量数据

按上面的命令重新跑：

1. `run_resnet50_ggml`
2. `run_resnet50`
3. `benchmark_pytorch_resnet50_forward.py`
4. `bench_conv_ops`
5. PyTorch `Conv2d` benchmark

注意事项：

- 不要并发跑这些命令
- 最好保持相同 CPU 绑定方式，例如 `taskset -c 60-79`
- 保持相同线程数，例如 `threads=8`

### 步骤 2：把新数值写回图脚本

编辑：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/generate_perf_charts.py`

需要修改的地方：

- `make_operator_chart()` 里的 `ggml` / `pytorch` / `onednn`
- `make_model_chart()` 里的 `values`

### 步骤 3：重新生成图

命令：

```bash
python /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/generate_perf_charts.py
```

生成后的文件会覆盖：

- `operator_performance_comparison.png`
- `model_inference_comparison.png`

### 步骤 4：同步更新实验报告

重新测量后，建议同步修改：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp8_resnet50/EXPERIMENT_REPORT_fold_unfold.md`

避免图中的数值和报告中的表格数值不一致。

## 当前模型文件说明

模型推理图使用的是原始模型，不是 SVD 模型：

- 原始 GGUF 模型：
  `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf`

- PyTorch 原始模型目录：
  `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model`

SVD 模型：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-svd-r50-f32.gguf`

当前这两张图中，模型推理对比图没有使用 SVD 模型。
