import torch
import time
import os
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from mymodel_file.gptJ_cloud import gptJ_cloud
from mymodel_file.gptJ_edge import gptJ_edge
from datasets import load_dataset

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

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for evaluation.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token IDs
            attention_mask: Optional tensor of shape [batch_size, seq_len] for padding mask
            
        Returns:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # 移动模型组件到输入设备
        self.embed = self.embed.to(device)
        self.ln_f = self.ln_f.to(device)
        self.lm_head = self.lm_head.to(device)
        
        # 获取 token embeddings
        x = self.embed(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # 逐层处理
        for layer_idx in range(self.num_layers):
            # Cloud 部分：计算 Q, K 和注意力权重
            # 使用 forward_no_cache 因为评测时不需要缓存
            q, k, attn_weights = self.cloud.forward_no_cache(x, layer_idx)
            
            # Edge 部分：计算 V 和最终输出
            # 将数据移动到 CPU (edge device)
            x_cpu = x.to('cpu')
            attn_weights_cpu = attn_weights.to('cpu')
            
            # 在 CPU 上计算
            _, x_cpu = self.edge.forward_no_cache(x_cpu, layer_idx, attn_weights_cpu)
            
            # 将结果移回原设备
            x = x_cpu.to(device)
        
        # 最终层归一化和语言模型头
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(self, prompt, max_length=50, temperature=0.4, top_k=50):
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
            # x = self.embed(torch.tensor(outputs).to(self.embed.weight.device))
            
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


from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
def load_and_tokenize_dataset(cache_dir: str='./minipile_cache', tokenizer=None, batch_size: int = 1):
    """
    Loads and tokenizes the MiniPile dataset.

    Args:
        cache_dir: Directory where MiniPile is cached/downloaded.
        tokenizer: Tokenizer for tokenizing the dataset.
        batch_size: Batch size for evaluation.

    Returns:
        A DataLoader for the tokenized dataset.
    """
    # Load dataset
    ds = load_dataset("JeanKaddour/minipile", split="validation", cache_dir=cache_dir)

    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Group the dataset into blocks of block_size (use consistent max_length)
    block_size = 512  # Use the same as tokenization max_length
    def group_texts(examples):
        all_ids = sum(examples["input_ids"], [])
        total_len = (len(all_ids) // block_size) * block_size
        blocks = [all_ids[i:i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": blocks}

    lm_dataset = tokenized.map(group_texts, batched=True, remove_columns=["attention_mask"])

    # DataLoader setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(lm_dataset, batch_size=batch_size, collate_fn=data_collator)

    return dataloader

from tqdm import tqdm
import math
from torch import nn
def evaluate_minipile_gptj(model, batch_size: int = 1, cache_dir: str = "./minipile_cache", Dataloader=None) -> dict:
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

   
    tokenizer = model.tokenizer  # already initialized in the pipeline
    dataloader = None
    if Dataloader is None:
        dataloader = load_and_tokenize_dataset(cache_dir, tokenizer, batch_size)
    else:
        dataloader = Dataloader

    
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    # Evaluation loop
    total_loss = 0.0
    total_batches = 0

    # model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # 拿到完整的 input_ids, attention_mask, 和已经被 collator 设好 -100 的 labels
            input_ids    = batch['input_ids'].to(device)       # [B, T]
            attention_mask = batch['attention_mask'].to(device)# [B, T]
            labels       = batch['labels'].to(device)          # [B, T], pad 已经是 -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits  = outputs                     # [B, T, V]


            # 手动 shift：logits 丢掉最后一位，labels 丢掉第一位
            # shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
            # shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

            shift_logits=logits
            labels=labels

            # 计算交叉熵 loss，ignore_index=-100 会跳过所有 pad 位置
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),  # [(B*(T-1)), V]
                shift_labels.view(-1)                          # [(B*(T-1))]
            )
            
            # Debug: 打印loss信息
            if batch_idx < 3:
                print(f"Batch {batch_idx} loss: {loss.item():.4f}")
                
            total_loss   += loss.item()
            total_batches+= 1



        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss)

    return {"avg_loss": avg_loss, "perplexity": perplexity}

if __name__ == "__main__":
    model_name = 'AI-ModelScope/gpt-j-6b'
    device_cloud = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_edge = 'cuda:0'
    
    pipeline = GPTJPipeline(model_name=model_name, device_cloud=device_cloud, device_edge=device_edge)
    dataloader=load_and_tokenize_dataset(tokenizer=pipeline.tokenizer)
    evaluate_minipile_gptj(model=pipeline,Dataloader=dataloader)
   
    # prompt = "Once upon a time"
    # generated_text = pipeline.generate(prompt, max_length=50)
    # print(generated_text)
    # prompt = "China is a"
    # generated_text = pipeline.generate(prompt, max_length=50)
    # print(generated_text)

