import torch
import time
from transformers import GPT2Tokenizer, AutoTokenizer
from mymodel_file.gpt2_cloud import gpt2_cloud
from mymodel_file.gpt2_edge  import gpt2_edge

class GPT2Pipeline:
    def __init__(self, model_name='gpt2', device_cloud='cuda:3', device_edge='cuda:3'):
        # 离线加载 tokenizer
        # self.tokenizer   = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/tianruiming/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
            local_files_only=True
        )
        self.cloud       = gpt2_cloud(model_name=model_name).to(device_cloud)
        # 强制 edge 放在 CPU
        self.edge        = gpt2_edge (model_name=model_name).to('cpu')
        self.embed       = self.cloud.model.transformer.wte
        self.ln_f        = self.cloud.model.transformer.ln_f
        self.lm_head     = self.cloud.model.lm_head
        self.num_layers  = len(self.cloud.q_weights)

    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs   = input_ids.copy()

        # reset caches for a fresh generation
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.edge.v_cache[i] = None

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
                elements=attn_cpu.numel()*attn_cpu.element_size()#B
                net_time+=elements/badwith/1024/1024#s
                t1 = time.time()
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                edge_time += time.time() - t1
                # 回到 GPU 继续下一层
                x = x_cpu.to(self.embed.weight.device)
                elements=x.numel()*x.element_size()#B
                net_time+=elements/badwith/1024/1024
                layer_calls += 1

        # 真实生成阶段
        for _ in range(max_length):
            cur_id = torch.tensor([[outputs[-1]]]).to(self.embed.weight.device)
            # clamp 位置，防止越界
            idx = len(outputs) - 1
            pos_clamped = idx if idx < max_ctx else max_ctx - 1
            pos_id = torch.tensor([[pos_clamped]]).to(self.embed.weight.device)
            x = self.embed(cur_id) + self.cloud.model.transformer.wpe(pos_id)
            for layer_idx in range(self.num_layers):
                # use cache-enabled forward so attention spans all previous tokens
                torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                torch.cuda.synchronize()
                cloud_time += time.time() - t0

                x_cpu = x.to('cpu')
                attn_cpu = attn_weights.to('cpu')
                t1 = time.time()
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                edge_time += time.time() - t1
                elements=attn_cpu.numel()*attn_cpu.element_size()#B
                net_time+=elements/badwith/1024/1024
                x = x_cpu.to(self.embed.weight.device)
                elements=x.numel()*x.element_size()#B
                net_time+=elements/badwith/1024/1024
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
    
    pipeline = GPT2Pipeline(model_name=model_name, device_cloud=device_cloud, device_edge=device_edge)
    prompt = "Once upon a time"
    generated_text = pipeline.generate(prompt, max_length=50)
    print(generated_text)
