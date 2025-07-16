# GPTJPipeline 评测功能说明

## 概述

基于 `MINI_PIPE_EVAL.py` 的思路，为 `GPTJPipeline` 类添加了完整的评测功能，可以在 MiniPile 数据集上评测模型性能，并统计云端、边缘和网络传输的时间开销。

## 新增功能

### 1. 数据集加载 (`load_and_tokenize_dataset`)
- 自动下载和缓存 MiniPile 验证集
- 使用与管道相同的 tokenizer 进行分词
- 支持批处理和数据整理

### 2. 前向推理 (`forward_logits`)
- 完整的前向传播，生成 logits
- 自动管理缓存状态
- 支持序列级处理

### 3. 评测方法 (`evaluate_minipile`)
- 计算平均交叉熵损失和困惑度
- 详细的时间统计（云端、边缘、网络传输）
- 支持限制评测批次数（用于快速测试）
- 返回完整的评测指标

## 使用方法

### 基本使用

```python
from detection.Loader.gptJLoader_mini import GPTJPipeline

# 创建管道实例
pipeline = GPTJPipeline(
    model_name='AI-ModelScope/gpt-j-6b',
    device_cloud='cuda:0',
    device_edge='cpu'
)

# 运行评测
results = pipeline.evaluate_minipile(
    batch_size=1,
    cache_dir="./minipile_cache",
    max_batches=10  # 可选：限制评测批次数
)

# 查看结果
print(f"平均损失: {results['avg_loss']:.4f}")
print(f"困惑度: {results['perplexity']:.2f}")
print(f"云端时间: {results['cloud_time']:.2f}s")
print(f"边缘时间: {results['edge_time']:.2f}s")
print(f"网络传输时间: {results['net_time']:.2f}s")
```

### 快速测试

运行提供的测试脚本：

```bash
cd /hy-tmp/sdpcos_2025/code
python test_gptj_pipeline_eval.py
```

### 完整评测

```bash
cd /hy-tmp/sdpcos_2025/code/detection/Loader
python gptJLoader_mini.py
```

## 返回指标

评测方法返回包含以下指标的字典：

- `avg_loss`: 平均交叉熵损失
- `perplexity`: 困惑度（exp(avg_loss)）
- `cloud_time`: 云端总计算时间（秒）
- `edge_time`: 边缘总计算时间（秒）
- `net_time`: 网络传输总时间（秒）
- `total_batches`: 评测的总批次数

## 性能分析

评测功能提供详细的性能分析：

1. **时间分布**: 显示云端、边缘和网络传输的时间占比
2. **效率指标**: 计算每批次、每层的平均处理时间
3. **瓶颈识别**: 帮助识别系统性能瓶颈

## 注意事项

1. **内存要求**: 确保有足够内存加载 GPT-J-6B 模型
2. **数据集下载**: 首次运行会下载 MiniPile 数据集（约6GB）
3. **设备配置**: 建议云端使用 GPU，边缘使用 CPU
4. **批次限制**: 使用 `max_batches` 参数进行快速测试

## 实现细节

评测功能基于 `MINI_PIPE_EVAL.py` 的核心思路：

1. 使用相同的数据预处理流程
2. 实现标准的语言模型损失计算
3. 添加详细的时间统计
4. 保持与原始模型的兼容性

这使得评测结果可以与标准 GPT-J 模型进行公平比较。
