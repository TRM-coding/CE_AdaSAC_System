import torch
import torch.nn as nn
import time
from transformers import GPT2Tokenizer, AutoTokenizer
from detection.Loader.mymodel_file.gpt2_cloud import gpt2_cloud
from detection.Loader.mymodel_file.gpt2_edge  import gpt2_edge
from detection.SVD_model import SVDED_GPT2_EDGE_Layer

class SVD_GPT2_Edge_Model(nn.Module):
    """包含所有SVD层的完整edge模型"""
    def __init__(self, original_edge, svd_reduce_rate, device='cpu'):
        super().__init__()
        self.device = device
        self.num_layers = original_edge.num_layers
        self.max_ctx = original_edge.max_ctx
        self.v_cache = [None] * self.num_layers
        
        # 用SVD压缩的层替换原始edge层
        self.svd_layers = nn.ModuleList()
        for i in range(self.num_layers):
            # 获取原始edge层
            if(i%2):
                original_edge_layer = original_edge.layers[i]
                # 创建SVD压缩层
                svd_layer = SVDED_GPT2_EDGE_Layer(
                    gpt2_edge_layer=original_edge_layer,
                    reduce_rate=0,
                    device=device
                )
                self.svd_layers.append(svd_layer)
            else:
                original_edge_layer = original_edge.layers[i]
                # 创建SVD压缩层
                svd_layer = SVDED_GPT2_EDGE_Layer(
                    gpt2_edge_layer=original_edge_layer,
                    reduce_rate=svd_reduce_rate,
                    device=device
                )
                self.svd_layers.append(svd_layer)
    
    def forward_all_layers(self, x, attn_weights_list):
        """
        处理所有层的前向传播
        x: Tensor [batch_size, seq_len, hidden]
        attn_weights_list: List[Tensor] - 每层的注意力权重
        返回: 最终的x输出
        """
        current_x = x
        for layer_idx in range(self.num_layers):
            attn_weights = attn_weights_list[layer_idx]
            # 使用SVD压缩的层
            self.v_cache[layer_idx], current_x = self.svd_layers[layer_idx].forward_cache(
                current_x, self.v_cache[layer_idx], attn_weights
            )
            
            # 应用sliding window到缓存
            if self.v_cache[layer_idx] is not None and self.v_cache[layer_idx].size(1) > self.max_ctx:
                self.v_cache[layer_idx] = self.v_cache[layer_idx][:, -self.max_ctx:, :]
        
        return current_x

class GPT2Pipeline:
    def __init__(self, model_name='gpt2', device_cloud='cuda:3', device_edge='cuda:3', svd_reduce_rate=0.5, use_compile=True):
        # 离线加载 tokenizer
        # self.tokenizer   = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/tianruiming/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
            local_files_only=True
        )
        self.cloud       = gpt2_cloud(model_name=model_name).to(device_cloud)
        # 强制 edge 放在 CPU
        original_edge    = gpt2_edge (model_name=model_name).to('cpu')
        self.embed       = self.cloud.model.transformer.wte
        self.ln_f        = self.cloud.model.transformer.ln_f
        self.lm_head     = self.cloud.model.lm_head
        self.num_layers  = len(self.cloud.q_weights)
        
        # SVD压缩参数
        self.svd_reduce_rate = svd_reduce_rate
        self.use_compile = use_compile
        
        # 创建整个SVD edge模型
        self.svd_edge_model = SVD_GPT2_Edge_Model(
            original_edge=original_edge,
            svd_reduce_rate=svd_reduce_rate,
            device='cpu'
        )
        
        print(f"Initialized GPT2Pipeline with SVD compression rate: {self.svd_reduce_rate}")
        # 移除torch.compile相关代码，因为现在我们按层处理，不需要编译整个模型
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs   = input_ids.copy()

        # reset caches for a fresh generation
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.svd_edge_model.v_cache[i] = None

        # 统计变量
        cloud_time = 0.0
        edge_time  = 0.0
        layer_calls= 0
        net_time=0.0
        badwith=10 #MB/s

        # 上下文窗口大小
        max_ctx = self.cloud.max_ctx

        # 预热缓存：将 prompt 中每个 token 走一次 forward_cache
        for pos, token_id in enumerate(input_ids):
            # clamp 位置，防止越界
            pos_clamped = pos if pos < max_ctx else max_ctx - 1
            cur_id = torch.tensor([[token_id]]).to(self.embed.weight.device)
            pos_id = torch.tensor([[pos_clamped]]).to(self.embed.weight.device)
            x = self.embed(cur_id) + self.cloud.model.transformer.wpe(pos_id)
            
            # 按照原始逻辑：逐层处理
            for layer_idx in range(self.num_layers):
                # cloud on GPU
                torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                torch.cuda.synchronize()
                cloud_time += time.time() - t0
                
                # edge on CPU: 同时把 x 和 attn_weights 都搬到 cpu
                x_cpu = x.to('cpu')
                attn_cpu = attn_weights.to('cpu')
                elements = attn_cpu.numel() * attn_cpu.element_size()  # B
                net_time += elements / badwith / 1024 / 1024  # s
                
                t1 = time.time()
                # 使用SVD压缩层处理单层
                self.svd_edge_model.v_cache[layer_idx], x_cpu = self.svd_edge_model.svd_layers[layer_idx].forward_cache(
                    x_cpu, self.svd_edge_model.v_cache[layer_idx], attn_cpu
                )
                edge_time += time.time() - t1
                
                # 应用sliding window到缓存
                if self.svd_edge_model.v_cache[layer_idx] is not None and self.svd_edge_model.v_cache[layer_idx].size(1) > self.svd_edge_model.max_ctx:
                    self.svd_edge_model.v_cache[layer_idx] = self.svd_edge_model.v_cache[layer_idx][:, -self.svd_edge_model.max_ctx:, :]
                
                # 回到 GPU 继续下一层
                x = x_cpu.to(self.embed.weight.device)
                elements = x.numel() * x.element_size()  # B
                net_time += elements / badwith / 1024 / 1024
                layer_calls += 1

        # 真实生成阶段
        for _ in range(max_length):
            cur_id = torch.tensor([[outputs[-1]]]).to(self.embed.weight.device)
            # clamp 位置，防止越界
            idx = len(outputs) - 1
            pos_clamped = idx if idx < max_ctx else max_ctx - 1
            pos_id = torch.tensor([[pos_clamped]]).to(self.embed.weight.device)
            x = self.embed(cur_id) + self.cloud.model.transformer.wpe(pos_id)
            
            # 按照原始逻辑：逐层处理
            for layer_idx in range(self.num_layers):
                # use cache-enabled forward so attention spans all previous tokens
                torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                torch.cuda.synchronize()
                cloud_time += time.time() - t0

                x_cpu = x.to('cpu')
                attn_cpu = attn_weights.to('cpu')
                elements = attn_cpu.numel() * attn_cpu.element_size()  # B
                net_time += elements / badwith / 1024 / 1024  # s
                
                t1 = time.time()
                # 使用SVD压缩层处理单层
                self.svd_edge_model.v_cache[layer_idx], x_cpu = self.svd_edge_model.svd_layers[layer_idx].forward_cache(
                    x_cpu, self.svd_edge_model.v_cache[layer_idx], attn_cpu
                )
                edge_time += time.time() - t1
                
                # 应用sliding window到缓存
                if self.svd_edge_model.v_cache[layer_idx] is not None and self.svd_edge_model.v_cache[layer_idx].size(1) > self.svd_edge_model.max_ctx:
                    self.svd_edge_model.v_cache[layer_idx] = self.svd_edge_model.v_cache[layer_idx][:, -self.svd_edge_model.max_ctx:, :]
                
                # 回到 GPU 继续下一层
                x = x_cpu.to(self.embed.weight.device)
                elements = x.numel() * x.element_size()  # B
                net_time += elements / badwith / 1024 / 1024
                layer_calls += 1
            
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

        # 打印平均耗时
        if layer_calls > 0:
            print(f"Avg GPU(cloud) per-layer: {cloud_time/layer_calls}s, CPU(edge) per-layer: {edge_time/layer_calls}s, net: {net_time/layer_calls}s")
            print(f"Avg GPU(cloud) per-token: {cloud_time/layer_calls+net_time/layer_calls}s, CPU(edge) per-token: {edge_time/layer_calls}s")
        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=True)
    
if __name__ == "__main__":
    model_name = 'gpt2'
    device_cloud = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    device_edge  = 'cpu' if torch.cuda.is_available() else 'cpu'
    
    # 可以调整SVD压缩率，0表示不压缩，1表示完全压缩
    svd_reduce_rate = 0.7
    
    pipeline = GPT2Pipeline(
        model_name=model_name, 
        device_cloud=device_cloud, 
        device_edge=device_edge,
        svd_reduce_rate=svd_reduce_rate,
        use_compile=False  # 启用torch.compile优化
    )
    prompt = "Once upon a time"
    generated_text = pipeline.generate(prompt, max_length=50)
    print(generated_text)
