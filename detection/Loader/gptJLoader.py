import torch
import time
import os
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from mymodel_file.gptJ_cloud import gptJ_cloud
from mymodel_file.gptJ_edge import gptJ_edge

class GPTJPipeline:
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cpu'):
        # 使用 ModelScope 下载模型
        print(f"Downloading model {model_name} using ModelScope...")
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir='./gpt-j-6b'
        )
        print(f"Model downloaded to: {model_dir}")
        
        # 使用本地模型路径加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 设置 pad_token 为 eos_token（GPT-J 没有 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cloud = gptJ_cloud(model_name=model_dir).to(device_cloud)
        # 强制 edge 放在 CPU
        self.edge = gptJ_edge(model_name=model_dir).to('cpu')
        
        # 获取 embedding 和输出层
        self.embed = self.cloud.model.transformer.wte
        self.ln_f = self.cloud.model.transformer.ln_f
        self.lm_head = self.cloud.model.lm_head
        self.num_layers = len(self.cloud.q_weights)

    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs = input_ids.copy()

        # reset caches for a fresh generation
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.edge.v_cache[i] = None

        # 统计变量
        cloud_time = 0.0
        edge_time = 0.0
        layer_calls = 0
        net_time = 0.0
        bandwidth = 10  # MB/s

        # 上下文窗口大小
        max_ctx = self.cloud.max_ctx

        # 预热缓存：将 prompt 中每个 token 走一次 forward_cache
        for pos, token_id in enumerate(input_ids):
            # clamp 位置，防止越界
            pos_clamped = pos if pos < max_ctx else max_ctx - 1
            cur_id = torch.tensor([[token_id]]).to(self.embed.weight.device)
            
            # GPT-J 没有位置embedding，直接使用 token embedding
            x = self.embed(cur_id)
            
            for layer_idx in range(self.num_layers):
                # cloud on GPU
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                cloud_time += time.time() - t0
                
                # edge on CPU: 把 x 和 attn_weights 都搬到 cpu
                x_cpu = x.to('cpu')
                attn_cpu = attn_weights.to('cpu')
                elements = attn_cpu.numel() * attn_cpu.element_size()  # B
                net_time += elements / bandwidth / 1024 / 1024  # s
                
                t1 = time.time()
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                edge_time += time.time() - t1
                print(f"edge_time_{layer_idx}:",time.time()-t1)
                # 回到 GPU 继续下一层
                x = x_cpu.to(self.embed.weight.device)
                elements = x.numel() * x.element_size()  # B
                net_time += elements / bandwidth / 1024 / 1024
                layer_calls += 1

        # 真实生成阶段
        for _ in range(max_length):
            cur_id = torch.tensor([[outputs[-1]]]).to(self.embed.weight.device)
            x = self.embed(cur_id)
            
            for layer_idx in range(self.num_layers):
                # use cache-enabled forward so attention spans all previous tokens
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                cloud_time += time.time() - t0

                x_cpu = x.to('cpu')
                attn_cpu = attn_weights.to('cpu')
                t1 = time.time()
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                edge_time += time.time() - t1
                
                elements = attn_cpu.numel() * attn_cpu.element_size()  # B
                net_time += elements / bandwidth / 1024 / 1024
                x = x_cpu.to(self.embed.weight.device)
                elements = x.numel() * x.element_size()  # B
                net_time += elements / bandwidth / 1024 / 1024
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
            print(f"Avg GPU(cloud) per-layer: {cloud_time/layer_calls:.4f}s, CPU(edge) per-layer: {edge_time/layer_calls:.4f}s, net: {net_time/layer_calls:.4f}s")
            print(f"Avg GPU(cloud) per-token: {cloud_time/layer_calls+net_time/layer_calls:.4f}s, CPU(edge) per-token: {edge_time/layer_calls:.4f}s")
            
        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=True)

if __name__ == "__main__":
    model_name = 'AI-ModelScope/gpt-j-6b'
    device_cloud = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_edge = 'cpu'
    
    pipeline = GPTJPipeline(model_name=model_name, device_cloud=device_cloud, device_edge=device_edge)
    prompt = "Once upon a time"
    generated_text = pipeline.generate(prompt, max_length=50)
    print(generated_text)
    prompt = "Once upon a time"
    generated_text = pipeline.generate(prompt, max_length=50)
    print(generated_text)

