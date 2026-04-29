# exp11 使用说明

## 环境

建议使用用户指定的 conda 环境：

```bash
conda run -n pytorch python src/llama.cpp/exp11_make_input_data/make_synthetic_minibatch.py --help
```

如需指定 GPU 5、6、7 中的一张，例如 GPU 5：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n pytorch python src/llama.cpp/exp11_make_input_data/make_synthetic_minibatch.py --task all --device cuda:0
```

这里 `cuda:0` 指的是 `CUDA_VISIBLE_DEVICES` 映射后的第一张可见卡。

## 一键生成默认小批量数据

```bash
cd /home/tianruiming/CE_ADA_LLAMA
CUDA_VISIBLE_DEVICES=5 conda run -n pytorch python src/llama.cpp/exp11_make_input_data/make_synthetic_minibatch.py --task all --device cuda:0
```

默认会读取：

- `src/llama.cpp/models/resnet50_hf_model`
- `src/llama.cpp/models/qwen2_5_1_5b`

默认会写入：

- `src/llama.cpp/exp11_make_input_data/data/resnet50_synthetic_minibatch.pt`
- `src/llama.cpp/exp11_make_input_data/data/qwen2_5_1_5b_synthetic_minibatch.pt`
- 对应的两个 manifest JSON。

## 分模型生成

只生成 ResNet50：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n pytorch python src/llama.cpp/exp11_make_input_data/make_synthetic_minibatch.py --task resnet50 --device cuda:0
```

只生成 Qwen2.5-1.5B：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n pytorch python src/llama.cpp/exp11_make_input_data/make_synthetic_minibatch.py --task qwen --device cuda:0
```

## 常用参数

- `--resnet-batch-size`: ResNet50 小批量大小，默认 8。
- `--resnet-steps`: ResNet50 输入优化步数，默认 80。
- `--qwen-batch-size`: Qwen 小批量大小，默认 4。
- `--qwen-seq-len`: Qwen 合成 embedding 序列长度，默认 16。
- `--qwen-steps`: Qwen 输入优化步数，默认 30。
- `--out-dir`: 输出目录，默认 `exp11_make_input_data/data`。
- `--seed`: 随机种子，默认 `20260428`。

## 读取示例

ResNet50:

```python
import torch
from transformers import AutoModelForImageClassification

data = torch.load("src/llama.cpp/exp11_make_input_data/data/resnet50_synthetic_minibatch.pt", map_location="cpu")
model = AutoModelForImageClassification.from_pretrained("src/llama.cpp/models/resnet50_hf_model", local_files_only=True).eval()
logits = model(pixel_values=data["pixel_values"].float()).logits
```

Qwen:

```python
import torch
from transformers import AutoModelForCausalLM

data = torch.load("src/llama.cpp/exp11_make_input_data/data/qwen2_5_1_5b_synthetic_minibatch.pt", map_location="cpu")
model = AutoModelForCausalLM.from_pretrained("src/llama.cpp/models/qwen2_5_1_5b", local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    out = model(
        inputs_embeds=data["inputs_embeds"].cuda().to(torch.bfloat16),
        attention_mask=data["attention_mask"].cuda(),
        position_ids=data["position_ids"].cuda(),
    )
```

