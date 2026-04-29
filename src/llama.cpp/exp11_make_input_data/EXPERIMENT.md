# exp11 测试实验文档

## 实验目的

验证 `make_synthetic_minibatch.py` 能够依据 CE-AdaSAC Synthetic Data Generator 方法，为 ResNet50 和 Qwen2.5-1.5B 生成不依赖真实数据的小批量测试输入，并记录模型对目标输出的置信度。

## 实验设置

- 日期：2026-04-28
- 工作目录：`/home/tianruiming/CE_ADA_LLAMA`
- Conda 环境：`pytorch`
- PyTorch：运行时由命令输出确认
- Transformers：运行时由命令输出确认
- 模型：
  - `src/llama.cpp/models/resnet50_hf_model`
  - `src/llama.cpp/models/qwen2_5_1_5b`

## 生成命令

本次生成采用默认小批量规模：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n pytorch python src/llama.cpp/exp11_make_input_data/make_synthetic_minibatch.py --task all --device cuda:0
```

若 GPU 5 繁忙，可将 `CUDA_VISIBLE_DEVICES` 改为 `6` 或 `7`。

## 验证命令

```bash
conda run -n pytorch python - <<'PY'
import json
from pathlib import Path
import torch

data_dir = Path("src/llama.cpp/exp11_make_input_data/data")
for name in [
    "resnet50_synthetic_minibatch_manifest.json",
    "qwen2_5_1_5b_synthetic_minibatch_manifest.json",
]:
    manifest = json.loads((data_dir / name).read_text())
    print(name)
    print("  batch_size:", manifest["batch_size"])
    print("  target_prob_mean:", manifest["target_prob_mean"])
    print("  target_prob_min:", manifest["target_prob_min"])

resnet = torch.load(data_dir / "resnet50_synthetic_minibatch.pt", map_location="cpu")
qwen = torch.load(data_dir / "qwen2_5_1_5b_synthetic_minibatch.pt", map_location="cpu")
print("resnet pixel_values:", tuple(resnet["pixel_values"].shape), resnet["pixel_values"].dtype)
print("qwen inputs_embeds:", tuple(qwen["inputs_embeds"].shape), qwen["inputs_embeds"].dtype)
PY
```

## 结果记录

实际生成和验证后的关键结果记录在：

- `data/resnet50_synthetic_minibatch_manifest.json`
- `data/qwen2_5_1_5b_synthetic_minibatch_manifest.json`

manifest 中的 `history` 字段保存优化过程中的 loss、cross entropy、目标概率均值和目标概率最小值，可用于判断合成输入是否让模型逐步变得更确信。

本次实际结果：

| 模型 | batch | 输入形状 | dtype | 步数 | 目标概率均值 | 目标概率最小值 | 数据文件大小 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 | 8 | `[8, 3, 224, 224]` | `torch.float16` | 80 | 0.999928 | 0.999612 | 3.5 MiB |
| Qwen2.5-1.5B | 4 | `[4, 16, 1536]` | `torch.float16` | 30 | 0.999170 | 0.998887 | 200 KiB |

生成输出：

```text
saved: /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp11_make_input_data/data/resnet50_synthetic_minibatch.pt
saved: /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp11_make_input_data/data/qwen2_5_1_5b_synthetic_minibatch.pt
```

验证输出摘要：

```text
resnet 8 0.9999282360076904 0.9996121525764465 (8, 3, 224, 224) torch.float16
qwen 4 16 0.999170184135437 0.9988868832588196 (4, 16, 1536) torch.float16
```

## 注意事项

- ResNet50 的 `pixel_values` 可直接输入 HF 模型，`images_uint8` 只用于检查或可视化。
- Qwen 的 `inputs_embeds` 是主测试数据。`nearest_input_ids` 是最近邻投影，会损失连续优化得到的信息，不应作为严格等价输入。
- 若只想快速冒烟测试，可把 `--resnet-steps` 和 `--qwen-steps` 调小；若希望更高目标置信度，可增加步数。
