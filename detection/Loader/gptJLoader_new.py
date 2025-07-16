import torch
import torch.nn as nn
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from mymodel_file.gptJ_cloud import gptJ_cloud
from mymodel_file.gptJ_edge import gptJ_edge
import torch.nn.functional as F
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer

class CloudEdgeCollaborativeGPTJ(nn.Module):
    """
    云边协同GPTJ-6B推理模块
    
    协同计算流程：
    1. 云侧：计算 Q, K 矩阵以及注意力权重 (Q @ K^T)
    2. 边侧：计算 V 矩阵和后续的注意力计算
    3. 云侧将注意力权重传给边侧，边侧完成剩余计算
    
    支持两种模式：
    - forward: 标准前向传播，无缓存机制
    - generate: 生成模式，带K/V缓存优化
    """
    
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cpu'):
        super().__init__()
        
        # 下载或加载模型
        if not torch.cuda.is_available():
            device_cloud = 'cpu'
            
        print(f"📥 使用ModelScope下载模型 {model_name}...")
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir='./gpt-j-6b'
        )
        print(f"✅ 模型下载完成，路径: {model_dir}")
        
        # 加载tokenizer
        print(f"🔤 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载云端和边端模型
        print(f"☁️  加载云端模型到 {device_cloud}...")
        self.cloud = gptJ_cloud(model_name=model_dir).to(device_cloud)
        
        print(f"🖥️  加载边缘模型到 {device_edge}...")
        self.edge = gptJ_edge(model_name=model_dir).to(device_edge)
        
        # 获取共享组件
        self.embed = self.cloud.model.transformer.wte
        self.ln_f = self.cloud.model.transformer.ln_f
        self.lm_head = self.cloud.model.lm_head
        self.num_layers = len(self.cloud.q_weights)
        
        # 保存设备信息
        self.device_cloud = device_cloud
        self.device_edge = device_edge
        
        print(f"🎯 云边协同模型初始化完成，共 {self.num_layers} 层")
        
    def reset_cache(self):
        """重置所有缓存"""
        self.cloud.k_cache = [None] * self.num_layers
        self.edge.v_cache = [None] * self.num_layers
    
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
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50, do_sample=True):
        """
        生成文本（带缓存优化）
        
        Args:
            input_ids: [batch_size, seq_len] 初始token序列
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k采样
            do_sample: 是否使用采样
            
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens] 生成的完整序列
        """
        device = input_ids.device
        batch_size, initial_seq_len = input_ids.shape
        
        # 重置缓存
        self.reset_cache()
        
        # 移动嵌入层到正确设备
        self.embed = self.embed.to(device)
        self.ln_f = self.ln_f.to(device)
        self.lm_head = self.lm_head.to(device)
        
        generated_ids = input_ids.clone()
        
        # 正确的缓存初始化：逐个token处理完整的prompt
        with torch.no_grad():
            for i in range(initial_seq_len):
                current_token = input_ids[:, i:i+1]  # [batch_size, 1]
                x = self.embed(current_token)
                
                # 逐层处理，正确更新隐藏状态
                for layer_idx in range(self.num_layers):
                    # 1. 云侧：使用缓存计算Q, K和注意力权重
                    q, k_all, attn_weights = self.cloud.forward_cache(x, layer_idx)
                    
                    # 2. 传输到边侧
                    x_edge = x.to(self.device_edge)
                    attn_weights_to_edge = attn_weights.to(self.device_edge)
                    
                    # 3. 边侧：使用缓存计算V和后续操作
                    v_all, x_out = self.edge.forward_cache(x_edge, layer_idx, attn_weights_to_edge)
                    
                    # 4. 传回云侧用于下一层
                    x = x_out.to(device)
        
        # 逐个生成新token
        for step in range(max_new_tokens):
            with torch.no_grad():
                # 只对最后一个token进行前向传播
                current_token = generated_ids[:, -1:]  # [batch_size, 1]
                logits = self._forward_with_cache(current_token)
                
                # 只取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
                
                # 生成下一个token
                if do_sample:
                    if top_k > 0:
                        # Top-k采样
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心解码
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # 检查是否生成了结束符
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return generated_ids
    
    def _forward_with_cache(self, input_ids):
        """
        带缓存的前向传播（用于生成）
        
        Args:
            input_ids: [batch_size, seq_len] 当前输入token
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        device = input_ids.device
        x = self.embed(input_ids)
        
        for layer_idx in range(self.num_layers):
            # 1. 云侧：使用缓存计算Q, K和注意力权重
            q, k_all, attn_weights = self.cloud.forward_cache(x, layer_idx)
            
            # 2. 传输到边侧的数据优化：
            # 对于生成，我们只需要最新token与所有历史token的注意力权重
            # attn_weights shape: [batch, num_heads, seq_q, seq_k]
            x_edge = x.to(self.device_edge)
            attn_weights_to_edge = attn_weights.to(self.device_edge)
            
            # 3. 边侧：使用缓存计算V和后续操作
            v_all, x_out = self.edge.forward_cache(x_edge, layer_idx, attn_weights_to_edge)
            
            # 4. 传回云侧用于下一层
            x = x_out.to(device)
        
        # 最终输出
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=1.0, top_k=50, do_sample=True):
        """
        文本生成的便捷接口
        
        Args:
            prompt: 输入文本提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k采样
            do_sample: 是否使用采样
            
        Returns:
            generated_text: 生成的完整文本
        """
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # 移动到云侧设备
        input_ids = input_ids.to(self.device_cloud)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample
            )
        
        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_transfer_stats(self):
        """
        获取数据传输统计信息（用于分析网络开销）
        
        Returns:
            dict: 包含传输数据量的统计信息
        """
        # 这里可以添加实际的传输量统计
        # 主要传输：注意力权重从云到边，最终输出从边到云
        attention_transfer_size = 0
        output_transfer_size = 0
        
        return {
            "attention_transfer_mb": attention_transfer_size / (1024**2),
            "output_transfer_mb": output_transfer_size / (1024**2),
            "total_transfer_mb": (attention_transfer_size + output_transfer_size) / (1024**2)
        }

from tqdm import tqdm
import math

class EVALER():
    def load_and_tokenize_dataset(self,cache_dir: str, tokenizer, batch_size: int = 1):
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


    def evaluate_minipile_gptj(self,model, batch_size: int = 1, cache_dir: str = "./minipile_cache", Dataloader=None) -> dict:
        """
        Evaluates a GPTJ-6B model instance on the MiniPile dataset.

        Args:
            model: A transformers.GPTJForCausalLM instance.
            batch_size: Batch size for evaluation.
            cache_dir: Directory where MiniPile is cached/downloaded.

        Returns:
            A dict with keys:
                - "avg_loss": Average cross-entropy loss.
                - "perplexity": Exponential of the average loss.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Load and tokenize dataset
        tokenizer = model.tokenizer  # already initialized in the pipeline
        dataloader = None
        if Dataloader is None:
            dataloader = self.load_and_tokenize_dataset(cache_dir, tokenizer, batch_size)
        else:
            dataloader = Dataloader

        # Initialize loss function with ignore_index=-100 to skip padding tokens
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
                shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
                shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

                # 计算交叉熵 loss，ignore_index=-100 会跳过所有 pad 位置
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),  # [(B*(T-1)), V]
                    shift_labels.view(-1)                          # [(B*(T-1))]
                )
                
               
                total_loss   += loss.item()
                total_batches+= 1


            avg_loss = total_loss / total_batches
            perplexity = math.exp(avg_loss)

        return {"avg_loss": avg_loss, "perplexity": perplexity}


if __name__ == "__main__":
    # 使用示例
    model_name = 'AI-ModelScope/gpt-j-6b'
    device_cloud = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_edge = 'cuda:0'
    
    # 创建云边协同模型
    collaborative_model = CloudEdgeCollaborativeGPTJ(
        model_name=model_name,
        device_cloud=device_cloud,
        device_edge=device_edge
    )
    
    # 测试文本生成
    prompt = "Once upon a time, in a distant galaxy"
    print(f"🔸 输入提示: {prompt}")
    
    generated_text = collaborative_model.generate_text(
        prompt, 
        max_new_tokens=30,
        temperature=0.8,
        top_k=50
    )
    
    print(f"🔸 生成文本: {generated_text}")
    
    # 测试标准前向传播
    input_ids = collaborative_model.tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(device_cloud)
    
    with torch.no_grad():
        logits = collaborative_model.forward(input_ids)
        print(f"🔸 Forward输出形状: {logits.shape}")

    eval=EVALER()

    dataloader=eval.load_and_tokenize_dataset(cache_dir='./minipile_cache',tokenizer=collaborative_model.tokenizer)
    eval.evaluate_minipile_gptj(collaborative_model,Dataloader=dataloader)
