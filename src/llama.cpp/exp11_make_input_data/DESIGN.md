# exp11 小批量合成测试数据设计文档

## 目标

本实验在 `exp11_make_input_data` 中为两个模型构建小批量测试数据：

- `resnet50_hf_model`: 图像分类模型，输入为图像张量。
- `qwen2_5_1_5b`: causal language model，输入为 Transformer token embedding。

数据构建依据论文 `1571178804 paper (1).pdf` 的 Synthetic Data Generator 方法：不依赖原始训练集，随机初始化输入，对目标类别或目标词表 id 使用交叉熵损失反向优化输入本身，使模型对目标输出产生高置信预测。

## 方法映射

论文中的目标为：

1. 选择目标类别 `c_hat`。CNN 中是类别号，Transformer 中是词表索引。
2. 随机初始化输入 `X`。
3. 前向得到 `F(X)`。
4. 用目标 `c_hat` 计算交叉熵 `L(X)`。
5. 对输入 `X` 做梯度下降，直到模型对目标输出较高置信。

本实现对应如下：

- ResNet50: 优化 unconstrained raw image，经 `sigmoid` 映射到 `[0, 1]`，再按 HF processor 的 ImageNet mean/std 归一化后送入模型。
- Qwen2.5-1.5B: 优化连续 `inputs_embeds`，在最后一个位置的 logits 上对目标 token id 做交叉熵。离散 token 不可直接梯度下降，因此 `inputs_embeds` 是主数据；同时提供最近邻 token 投影，供只能接收 token ids 的流程做近似检查。

## 输出文件

默认输出目录为 `src/llama.cpp/exp11_make_input_data/data`。

ResNet50:

- `resnet50_synthetic_minibatch.pt`
- `resnet50_synthetic_minibatch_manifest.json`

`.pt` 文件字段：

- `pixel_values`: `[B, 3, 224, 224]`，float16，已经按模型 mean/std 归一化，可直接传给 `AutoModelForImageClassification(pixel_values=...)`。
- `images_uint8`: `[B, 3, 224, 224]`，uint8，用于可视化或转图片。
- `target_class_ids`: 目标类别 id。
- `target_labels`: 目标类别名称。
- `target_probs`: 生成结束时目标类别概率。
- `top5_class_ids`, `top5_probs`: 生成结束时 top-5 结果。
- `history`: 优化日志。
- `metadata`: 模型路径、步数、学习率、耗时等。

Qwen2.5-1.5B:

- `qwen2_5_1_5b_synthetic_minibatch.pt`
- `qwen2_5_1_5b_synthetic_minibatch_manifest.json`

`.pt` 文件字段：

- `inputs_embeds`: `[B, T, hidden_size]`，float16，是主要合成输入。
- `nearest_input_ids`: `[B, T]`，将每个 embedding 投影到最近词表 embedding 后得到的近似 token ids。
- `attention_mask`, `position_ids`: 与 `inputs_embeds` 配套。
- `target_token_ids`, `target_texts`: 目标词表 id 和可读文本。
- `projected_texts`: `nearest_input_ids` 解码文本，仅作近似参考。
- `target_probs`: 最后位置目标 token 概率。
- `top5_token_ids`, `top5_probs`: 最后位置 top-5 结果。
- `history`: 优化日志。
- `metadata`: 模型路径、序列长度、步数、学习率、耗时等。

## 设计取舍

- 不下载外部数据，不使用 ImageNet、文本语料或 prompt 数据集，满足数据无关构建。
- ResNet50 默认 batch size 为 8，目标类别均匀覆盖 `0..999` 的若干点。
- Qwen 默认 batch size 为 4、序列长度 16，目标 token 从常见英文和中文片段中取首 token，保证目标 token 可解释。
- Qwen 采用连续 embedding 数据作为主产物，这是对论文 Transformer 场景中“优化输入”的忠实工程实现；离散 token 投影不是无损表示。

