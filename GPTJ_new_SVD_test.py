import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from detection.Loader.mymodel_file.gptJ_cloud import gptJ_cloud
from detection.Loader.mymodel_file.gptJ_edge import gptJ_edge
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer

class SVD_GPTJ_Edge_Model(nn.Module):
    """包含所有SVD层的完整edge模型，兼容原始edge模型接口"""
    def __init__(self, original_edge, svd_reduce_rate, device='cuda:0', svd_device='cuda:0'):
        super().__init__()
        self.device = device
        self.svd_device = svd_device
        self.num_layers = original_edge.num_layers
        self.max_ctx = original_edge.max_ctx
        self.v_cache = [None] * self.num_layers
        
        # 用SVD压缩的层替换原始edge层
        self.svd_layers = nn.ModuleList()
        for i in range(self.num_layers):
            original_edge_layer = original_edge.layers[i]
            if(i>=0):
                self.svd_layers.append(original_edge_layer)
                continue
            if(i%2):
                # 奇数层跳过压缩
                svd_layer = SVDED_GPTJ_EDGE_Layer(
                    gptj_edge_layer=original_edge_layer,
                    reduce_rate=svd_reduce_rate,
                    device=device,
                    svd_device=svd_device
                )
                self.svd_layers.append(svd_layer)
            else:
                # 偶数层进行SVD压缩
                svd_layer = SVDED_GPTJ_EDGE_Layer(
                    gptj_edge_layer=original_edge_layer,
                    reduce_rate=svd_reduce_rate,
                    device=device,
                    svd_device=svd_device
                )
                self.svd_layers.append(svd_layer)
    
    def forward_cache(self, x, layer_idx, attn_weights):
        """
        兼容原始edge模型的forward_cache接口
        Args:
            x: 输入tensor
            layer_idx: 层索引
            attn_weights: 注意力权重
        Returns:
            tuple: (v_cache, output_x) - 与原始edge模型相同的返回格式
        """
        # 使用SVD压缩的层进行前向传播
        self.v_cache[layer_idx], output_x = self.svd_layers[layer_idx].forward_cache(
            x, self.v_cache[layer_idx], attn_weights
        )
        
        # 应用sliding window到缓存
        if self.v_cache[layer_idx] is not None and self.v_cache[layer_idx].size(1) > self.max_ctx:
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, -self.max_ctx:, :]

        return self.v_cache[layer_idx], output_x

class GPTJPipeline:
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cuda:0', svd_reduce_rate=0.5):
        # 使用 ModelScope 下载模型
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir='./gpt-j-6b'
        )
        
        # 使用本地模型路径加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 设置 pad_token 为 eos_token（GPT-J 没有 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.cloud       = gptJ_cloud(model_name=model_dir).to(device_cloud)
        # 强制 edge 放在 CPU
        original_edge    = gptJ_edge (model_name=model_dir).to('cuda:0')
        self.embed       = self.cloud.model.transformer.wte
        self.ln_f        = self.cloud.model.transformer.ln_f
        self.lm_head     = self.cloud.model.lm_head
        self.num_layers  = len(self.cloud.q_weights)
        
        # SVD压缩参数
        self.svd_reduce_rate = svd_reduce_rate
        
        # 创建整个SVD edge模型
        # 如果有GPU，先在GPU上进行SVD分解，然后移到CPU
        svd_device = device_cloud if torch.cuda.is_available() else 'cuda:0'
        
        self.edge = SVD_GPTJ_Edge_Model(
            original_edge=original_edge,
            svd_reduce_rate=svd_reduce_rate,
            device='cuda:0',  # 最终运行在CPU上
            svd_device=svd_device  # 但SVD分解在GPU上进行
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        标准前向传播（无缓存）
        
        Args:
            input_ids: [batch_size, seq_len] token序列
            attention_mask: 可选的注意力掩码
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] 输出logits
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # 移动嵌入层到输入设备
        self.embed = self.embed.to(device)
        self.ln_f = self.ln_f.to(device)
        self.lm_head = self.lm_head.to(device)
        
        # Token嵌入
        x = self.embed(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # 逐层协同计算
        for layer_idx in range(self.num_layers):
            # 1. 云侧计算：Q, K矩阵和注意力权重
            q, k, attn_weights = self.cloud.forward_no_cache(x, layer_idx)
            
            # 2. 将注意力权重传输到边侧 (模拟网络传输)
            attn_weights_edge = attn_weights.to(self.device_edge)
            x_edge = x.to(self.device_edge)
            
            # 3. 边侧计算：V矩阵和后续操作
            v, x_out = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
            
            # 4. 将结果传回云侧 (或保持在边侧，根据下一层的需要)
            x = x_out.to(device)
        
        # 最终层归一化和输出投影
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs   = input_ids.copy()

        # reset caches for a fresh generation
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.edge.v_cache[i] = None

        # 上下文窗口大小
        max_ctx = self.cloud.max_ctx
        
        # 预热缓存：将 prompt 中每个 token 走一次 forward_cache
        for pos, token_id in enumerate(input_ids):
            
            # clamp 位置，防止越界
            pos_clamped = pos if pos < max_ctx else max_ctx - 1
            cur_id = torch.tensor([[token_id]]).to(self.embed.weight.device)
            
            # GPT-J 没有位置embedding，直接使用 token embedding
            x = self.embed(cur_id)
            
            # 逐层处理 - 云端计算QK，边端计算V
            for layer_idx in range(self.num_layers):
                # 云端计算：在GPU上计算QK和注意力权重，保持K缓存在GPU
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                
                # 边端计算：在CPU上计算V和输出，保持V缓存在CPU
                x_cpu = x.to('cuda:0')
                attn_cpu = attn_weights.to('cuda:0')
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                
                # 将处理后的x传回云端
                x = x_cpu.to(self.embed.weight.device)
        
        # 真实生成阶段
        for token_idx in range(max_length):
            cur_id = torch.tensor([[outputs[-1]]]).to(self.embed.weight.device)
            x = self.embed(cur_id)
            
            # 逐层处理 - 云端计算QK，边端计算V
            for layer_idx in range(self.num_layers):
                # 云端计算：在GPU上计算QK和注意力权重，保持K缓存在GPU
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)

                # 边端计算：在CPU上计算V和输出，保持V缓存在CPU
                x_cpu = x.to('cuda:0')
                attn_cpu = attn_weights.to('cuda:0')
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                
                # 将处理后的x传回云端
                x = x_cpu.to(self.embed.weight.device)
            
            # final normalization and LM head to get logits
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            # 用 top-k + 温度采样代替贪心 argmax
            next_logits = logits[:, -1, :] / temperature
            topk_vals, topk_idx = torch.topk(next_logits, k=top_k, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            next_id = topk_idx[0, torch.multinomial(probs, num_samples=1).item()].item()
            outputs.append(next_id)
            
            if next_id == self.tokenizer.eos_token_id:
                break
            
        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=True)

if __name__ == "__main__":
    model_name = 'AI-ModelScope/gpt-j-6b'
    device_cloud = 'cuda:0' if torch.cuda.is_available() else 'cuda:0'
    device_edge = 'cuda:0'
    
    # 测试不同的SVD压缩率
    svd_rates = [0]
    
    for svd_rate in svd_rates:
        try:
            pipeline = GPTJPipeline(
                model_name=model_name, 
                device_cloud=device_cloud, 
                device_edge=device_edge,
                svd_reduce_rate=svd_rate
            )
            
            prompt = "China is a "
            
            overall_start_time = time.time()
            generated_text = pipeline.generate(prompt, max_length=20)
            overall_end_time = time.time()
            
            print(f"Generated text: {generated_text}")
            print(f"Generation time: {overall_end_time - overall_start_time:.2f}s")
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()