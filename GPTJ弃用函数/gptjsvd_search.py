# -----------------------------------------------------------------------------
# make_data_pid 随机输出空间维度 —— 对齐到 GPT-J 单层输出 (seq_len=1, hidden=4096)
TOTAL_NUMBER        = 1024         # 合成样本总数
BATCH_SIZE          = 8            # 小批量大小

# 1. 随机空间维度：对应 transformer 输出 shape=(batch, seq_len, hidden)
CHANNEL             = 4096         # GPT-J-6B hidden_size
DIM1                = 1            # seq_len
DIM2                = 1            # 单 token 时宽度取 1

# 2. 分类头输出：vocab_size
OUTPUT_SIZE         = 50400        # GPT-J-6B 词表大小

# 学习率 / 其他超参
LEARNING_RATE       = 1e-4         # 学习率
WARM_LR             = 1e-5         # 预热学习率
RANDN_MAGNIFICATION = 1.0          # 随机噪声放大倍数
CONFIDENCE          = 0.9          # 目标置信度
TARGET_ACC          = 0.8          # 目标准确率
# -----------------------------------------------------------------------------
# 小批量数据保存路径
SAVE_BATCH_PATH = "gptj_input_data/generated_batch.pt"
# -----------------------------------------------------------------------------

import os
import gc
import torch
import torch.nn as nn
from detection.DataGenerator_gptJ import InputOptimizer
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from detection.Loader.mymodel_file.gptJ_edge import gptJ_edge
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer


class LayerWrapper(nn.Module):
    """包装器模块，用于将层包装成nn.Module以供torch.profiler使用"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x_input, attn_input):
        # 模拟v_cache为None的情况（第一个token）
        _, output = self.layer.forward_cache(x_input, None, attn_input)
        return output


def init_flops(model_name='AI-ModelScope/gpt-j-6b', device='cuda:0'):
    """
    计算每一层在不同SVD压缩率下的FLOPS
    
    Args:
        model_name: 模型名称
        device: 计算设备
        
    Returns:
        dict: {layer_idx: [flops_0.0, flops_0.1, ..., flops_0.9]}
    """
    print(f"🚀 开始初始化FLOPS计算...")
    print(f"📋 配置信息:")
    print(f"   - 模型: {model_name}")
    print(f"   - 设备: {device}")
    
    # 使用 ModelScope 下载模型
    print(f"📥 使用ModelScope下载模型 {model_name}...")
    model_dir = snapshot_download(
        repo_id=model_name,
        cache_dir='./gpt-j-6b'
    )
    print(f"✅ 模型下载完成，路径: {model_dir}")
    
    # 加载tokenizer用于获取输入规格
    print(f"🔤 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载edge模型
    print(f"🖥️  加载边缘模型...")
    original_edge = gptJ_edge(model_name=model_dir).to(device)
    num_layers = original_edge.num_layers
    print(f"📊 模型共有 {num_layers} 层")
    
    # 准备输入数据用于FLOPS计算
    batch_size = 1
    seq_len = 1
    hidden_size = original_edge.model.config.n_embd
    num_heads = original_edge.model.config.n_head
    
    input_x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16).to(device)
    attn_weights = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float16).to(device)
    
    print(f"📏 输入张量规格:")
    print(f"   - input_x: {input_x.shape}")
    print(f"   - attn_weights: {attn_weights.shape}")
    
    flops_dict = {}
    for layer_idx in range(num_layers):
        print(f"\n🔧 处理第 {layer_idx + 1}/{num_layers} 层...")
        flops_array = []
        for k in range(10):
            reduce_rate = k / 10.0
            print(f"  📊 计算 reduce_rate = {reduce_rate:.1f} 的FLOPS...")
            try:
                if reduce_rate == 0.0:
                    test_layer = original_edge.layers[layer_idx]
                    wrapper_module = LayerWrapper(test_layer).to(device)
                else:
                    svd_layer = SVDED_GPTJ_EDGE_Layer(
                        gptj_edge_layer=original_edge.layers[layer_idx],
                        reduce_rate=reduce_rate,
                        device=device,
                        svd_device=device
                    )
                    wrapper_module = LayerWrapper(svd_layer).to(device)
                test_input_x = input_x.clone().detach()
                test_attn_weights = attn_weights.clone().detach()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True
                ) as prof:
                    _ = wrapper_module(test_input_x, test_attn_weights)
                flops = prof.key_averages().total_average().flops or 0
                flops_array.append(flops)
                print(f"    ✅ reduce_rate {reduce_rate:.1f}: {flops:,.0f} FLOPs")
                del wrapper_module
                if reduce_rate != 0.0:
                    del svd_layer
                del test_input_x, test_attn_weights
                torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f"    ❌ reduce_rate {reduce_rate:.1f} 计算失败: {e}")
                flops_array.append(0)
                torch.cuda.empty_cache(); gc.collect()
        flops_dict[layer_idx] = flops_array
        print(f"  🎯 第 {layer_idx + 1} 层完成: {flops_array}")
        torch.cuda.empty_cache(); gc.collect()
    print(f"\n🎉 FLOPS计算完成！")
    del original_edge; torch.cuda.empty_cache(); gc.collect()
    return flops_dict


def save_flops_dict(flops_dict, filename="flops_dict.pt"):
    """保存FLOPS字典到文件"""
    torch.save(flops_dict, filename)
    print(f"💾 FLOPS字典已保存到 {filename}")


def load_flops_dict(filename="flops_dict.pt"):
    """从文件加载FLOPS字典"""
    flops_dict = torch.load(filename)
    print(f"📂 FLOPS字典已从 {filename} 加载")
    return flops_dict


def print_flops_analysis(flops_dict):
    """打印FLOPS分析报告"""
    print(f"\n📊 FLOPS分析报告")
    print("="*80)
    for layer_idx, flops_array in flops_dict.items():
        print(f"\n🔢 第 {layer_idx} 层:")
        print(f"   reduce_rate: " + " ".join([f"{k/10:.1f}" for k in range(10)]))
        print(f"   FLOPs (M):   " + " ".join([f"{f/1e6:6.1f}" for f in flops_array]))
        original_flops = flops_array[0] if flops_array[0] > 0 else 1
        print(f"   压缩比例:    " + " ".join([f"{f/original_flops:6.2f}" if f>0 else "  N/A " for f in flops_array]))
    print(f"\n{'='*80}")


def generate_input_space(
    device: str = 'cuda:0',
    no_weight: bool = True
):
    """
    Generate a small batch of synthetic inputs for a GPT-J-6B model.
    Returns:
        input_data, output_label, label, highest_loss, lowest_loss
    """
    optimizer = InputOptimizer(
        model_name='AI-ModelScope/gpt-j-6b',
        device='cuda:0',
        batch_size=20,
        seq_len=64,
        hidden_size=4096,
        lr=1.2e-3,
        kd=1e-3  # example derivative gain
    )

    optimized_input,output_lable = optimizer.optimize(num_steps=50000, print_every=20)
    return optimized_input,output_lable


def save_generated_batch(
    device: str = "cuda:0",
    no_weight: bool = True,
    path: str = SAVE_BATCH_PATH
) -> str:
    """
    调用 generate_input_space 生成一个小批量输入，并将结果保存到本地文件。
    Returns: 最终保存文件的绝对路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    optimized_batch,output_lable=generate_input_space()
    batch = {
        "input_data": optimized_batch,
        "output_label":output_lable
    }
    torch.save(batch, path)
    return os.path.abspath(path)


def load_generated_batch(
    path: str = SAVE_BATCH_PATH
) -> dict:
    """
    从文件中读取之前保存的小批量数据。
    Returns: 包含 input_data, output_label, label, highest_loss, lowest_loss。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such batch file: {path}")
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    # 示例：生成并保存小批量数据
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_dir = snapshot_download(repo_id='AI-ModelScope/gpt-j-6b', cache_dir='./gpt-j-6b')
    # model = gptJ_edge(model_name=model_dir).to(device)
    save_path = save_generated_batch( device=device)
    print(f"Batch saved to: {save_path}")

    # 示例：加载并检查数据
    batch = load_generated_batch(save_path)
    print({k: getattr(batch[k], 'shape', batch[k]) for k in batch})
