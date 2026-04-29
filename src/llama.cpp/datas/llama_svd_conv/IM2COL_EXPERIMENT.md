# llama.cpp 自带 im2col 卷积实验

本文件单独记录 llama.cpp 自带 `im2col` 卷积路径的模型级和算子级性能结果。

这里的 “llama.cpp 自带 im2col 卷积路径” 指的是：

- `run_resnet50_ggml`
- 内部卷积实现走 `ggml_conv_2d`
- `ggml_conv_2d` 的卷积实现是 `im2col + matmul` 路径

## 实验口径

- CPU 绑定：`60-79`
- 线程数：`8`
- 不并发运行其他 benchmark
- 模型：原始 GGUF 模型
  `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf`
- 图片：
  `/tmp/coco_cat.jpg`

## 模型推理速度

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

结果：

- `forward_ms_mean = 117.316`
- `forward_ms_median = 115.75`
- `forward_ms_min = 112.179`
- `forward_ms_max = 151.637`

## 算子速度

命令：

```bash
taskset -c 60-79 ./build-release-current/bench_conv_ops
```

当前只取其中的 `ggml_ms`，因为这里关注的是 llama.cpp 自带 `im2col` 路径。

结果：

| 卷积类型 | 算子耗时 |
|---|---:|
| `7x7 stem` | `10.0863 ms` |
| `3x3 block` | `3.77347 ms` |
| `1x1 bottleneck` | `1.33198 ms` |

## 数据文件

同一份结果也保存成了 JSON，便于后续脚本读取：

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv/im2col_experiment_results.json`

## 如果要重跑

1. 先确保没有其他 benchmark 并发运行。
2. 再按上面的两条命令分别重跑：
   - `run_resnet50_ggml`
   - `bench_conv_ops`
3. 把新数值同步更新到：
   - `im2col_experiment_results.json`
   - `generate_perf_charts.py`
   - 如有需要，再同步更新 `DATA_PROVENANCE.md` 和实验报告。
