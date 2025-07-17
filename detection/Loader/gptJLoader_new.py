import torch
from torch import nn
from transformers import AutoTokenizer
import time
from mymodel_file.gptJ_cloud import gptJ_cloud
from mymodel_file.gptJ_edge import gptJ_edge


class GPTJCloudEdgeCollaborator(nn.Module):
    """
    GPT-J 云边协同模型
    云侧：完成Q、K的计算和attention权重计算
    边侧：完成V的计算和最终的attention输出
    """
    
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cpu'):
        super().__init__()
        
        self.device_cloud = device_cloud
        self.device_edge = device_edge
        
        # 初始化云侧和边侧模型
        print(f"初始化云侧模型 (设备: {device_cloud})...")
        self.cloud = gptJ_cloud(model_name=model_name).to(device_cloud)
        
        print(f"初始化边侧模型 (设备: {device_edge})...")
        self.edge = gptJ_edge(model_name=model_name).to(device_edge)
        
        # 获取共享的组件（embedding和输出层）
        self.embed = self.cloud.model.transformer.wte.to(device_cloud)
        self.ln_f = self.cloud.model.transformer.ln_f.to(device_cloud)
        self.lm_head = self.cloud.model.lm_head.to(device_cloud)
        
        # 模型配置
        self.num_layers = len(self.cloud.q_weights)
        self.vocab_size = self.cloud.model.config.vocab_size
        
        # 初始化tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # 如果直接加载失败，尝试从本地路径加载
            self.tokenizer = AutoTokenizer.from_pretrained('./gpt-j-6b/AI-ModelScope/gpt-j-6b')
            
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播用于数据集评估
        Args:
            input_ids: [batch_size, seq_len] token ids
            attention_mask: [batch_size, seq_len] attention mask (1=valid, 0=padding)
        Returns:
            logits: [batch_size, seq_len, vocab_size] 预测logits
        """
        # 1. Embedding
        x = self.embed(input_ids.to(self.device_cloud))  # [B, T, D]
        
        batch_size, seq_len = input_ids.shape
        
        # 2. 如果没有提供attention_mask，根据pad_token_id生成
        if attention_mask is None:
            if self.tokenizer.pad_token_id is not None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)
        
        # 创建position_ids
        position_ids = torch.arange(seq_len, device=self.device_cloud).unsqueeze(0).expand(batch_size, -1)
        
        # 3. 逐层处理
        for layer_idx in range(self.num_layers):
            # 云侧：计算Q、K和attention权重（传入attention_mask和position_ids）
            q, k, attn_weights = self.cloud.forward_no_cache(
                x, layer_idx, position_ids, attention_mask.to(self.device_cloud)
            )
            
            # 将数据传输到边侧设备
            x_edge = x.to(self.device_edge)
            attn_weights_edge = attn_weights.to(self.device_edge)
            
            # 边侧：计算V和最终输出
            _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
            
            # 将结果传回云侧
            x = x_edge.to(self.device_cloud)
        
        # 4. 最终的Layer Norm和LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    # def forward(self, input_ids):
    #     """
    #     前向传播用于数据集评估
    #     Args:
    #         input_ids: [batch_size, seq_len] token ids
    #     Returns:
    #         logits: [batch_size, seq_len, vocab_size] 预测logits
    #     """
    #     # 1. Embedding
    #     x = self.embed(input_ids.to(self.device_cloud))  # [B, T, D]
        
    #     # 2. 逐层处理
    #     for layer_idx in range(self.num_layers):
    #         # 云侧：计算Q、K和attention权重
    #         q, k, attn_weights = self.cloud.forward_no_cache(x, layer_idx)
            
    #         # 将数据传输到边侧设备
    #         x_edge = x.to(self.device_edge)
    #         attn_weights_edge = attn_weights.to(self.device_edge)
            
    #         # 边侧：计算V和最终输出
    #         _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
            
    #         # 将结果传回云侧
    #         x = x_edge.to(self.device_cloud)
        
    #     # 3. 最终的Layer Norm和LM Head
    #     x = self.ln_f(x)
    #     logits = self.lm_head(x)
        
    #     return logits
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_p=0.9, do_sample=True):
        """文本生成方法 - 修复版本"""
        self.eval()
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs = input_ids.copy()
        
        print(f"开始生成，初始prompt: '{prompt}'")
        print(f"目标生成长度: {max_length} tokens")
        print(f"初始token数: {len(input_ids)}")
        
        # 统计时间
        cloud_time = 0
        edge_time = 0
        transfer_time = 0
        
        with torch.no_grad():
            # 逐token生成
            for step in range(max_length):
                if step % 5 == 0:
                    print(f"生成进度: {step}/{max_length}")
                
                # 处理完整的序列
                current_ids = torch.tensor([outputs]).to(self.device_cloud)  # [1, current_seq_len]
                x = self.embed(current_ids)  # [1, current_seq_len, hidden_size]
                
                # 创建position_ids和attention_mask
                seq_len = len(outputs)
                position_ids = torch.arange(seq_len, device=self.device_cloud).unsqueeze(0)  # [1, seq_len]
                
                # 创建attention_mask（生成时所有token都是有效的）
                attention_mask = torch.ones_like(current_ids)
                
                # 逐层处理
                for layer_idx in range(self.num_layers):
                    # 云侧计算（传入position_ids和attention_mask）
                    t0 = time.time()
                    q, k, attn_weights = self.cloud.forward_no_cache(
                        x, layer_idx, position_ids, attention_mask
                    )
                    cloud_time += time.time() - t0
                    
                    # 数据传输到边侧
                    t1 = time.time()
                    x_edge = x.to(self.device_edge)
                    attn_weights_edge = attn_weights.to(self.device_edge)
                    transfer_time += time.time() - t1
                    
                    # 边侧计算
                    t2 = time.time()
                    _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
                    edge_time += time.time() - t2
                    
                    # 数据传回云侧
                    t3 = time.time()
                    x = x_edge.to(self.device_cloud)
                    transfer_time += time.time() - t3
                
                # 最终处理
                x = self.ln_f(x)
                logits = self.lm_head(x)  # [1, current_seq_len, vocab_size]
                
                # 只使用最后一个位置的logits进行采样
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # 采样下一个token
                if do_sample:
                    # 应用temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-p采样
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # 移除累积概率超过top_p的token
                        sorted_indices_to_remove = cumulative_probs > top_p
                        if len(sorted_indices_to_remove) > 1:
                            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # 从分布中采样
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    # 贪心解码
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                outputs.append(next_token_id)
                
                # 调试信息：显示生成的token
                if step < 10:
                    token_text = self.tokenizer.decode([next_token_id])
                    print(f"  Step {step}: token_id={next_token_id}, token='{token_text}'")
                
                # 检查是否遇到结束token
                if next_token_id == self.tokenizer.eos_token_id:
                    print("遇到结束token，停止生成")
                    break
        
        # 生成完成的处理代码保持不变...
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        # 输出统计信息
        total_time = cloud_time + edge_time + transfer_time
        generated_tokens = len(outputs) - len(input_ids)
        
        print(f"\n生成完成!")
        print(f"总时间: {total_time:.3f}s")
        if total_time > 0:
            print(f"云侧时间: {cloud_time:.3f}s ({cloud_time/total_time*100:.1f}%)")
            print(f"边侧时间: {edge_time:.3f}s ({edge_time/total_time*100:.1f}%)")
            print(f"传输时间: {transfer_time:.3f}s ({transfer_time/total_time*100:.1f}%)")
        print(f"生成的token数: {generated_tokens}")
        if generated_tokens > 0:
            print(f"平均每token时间: {total_time/generated_tokens:.3f}s")
        
        return generated_text
    
    def forward_with_cache(self, input_ids, use_cache=True):
        """
        带缓存的前向传播（用于生成时的优化）
        注意：这个方法暂时未实现，因为要求忽略缓存策略
        """
        return self.forward(input_ids)
    
    def reset_cache(self):
        """重置所有缓存"""
        # 由于我们忽略缓存策略，这个方法为空
        pass
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'cloud_device': self.device_cloud,
            'edge_device': self.device_edge,
            'model_name': 'GPT-J Cloud-Edge Collaborator'
        }


# 方便的工厂函数
def create_gptj_cloud_edge_model(model_name='AI-ModelScope/gpt-j-6b', 
                                 device_cloud='cuda:0', 
                                 device_edge='cuda:0'):
    """
    创建GPT-J云边协同模型的工厂函数
    
    Args:
        model_name: 模型名称或路径
        device_cloud: 云侧设备
        device_edge: 边侧设备
    
    Returns:
        GPTJCloudEdgeCollaborator: 云边协同模型实例
    """
    return GPTJCloudEdgeCollaborator(
        model_name=model_name,
        device_cloud=device_cloud,
        device_edge=device_edge
    )



from datasets import load_dataset
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
            shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
            shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

            # shift_logits=logits
            # labels=labels

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

# 示例使用
if __name__ == "__main__":
    # 创建云边协同模型
    model = create_gptj_cloud_edge_model(
        device_cloud='cuda:0',
        device_edge='cuda:0'
    )
    
    # 生成文本示例 - 使用更保守的参数
    prompt = "The future of artificial intelligence is"
    generated_text = model.generate(
        prompt=prompt,
        max_length=20,  # 减少长度先测试
        temperature=0.7,  # 降低temperature
        top_p=0.8,       # 降低top_p
        do_sample=False  # 先用贪心解码测试
    )
    
    print(f"\n生成结果:")
    print(f"原始prompt: {prompt}")
    print(f"完整生成文本: {generated_text}")


    dataloader=load_and_tokenize_dataset(tokenizer=model.tokenizer)
    evaluate_minipile_gptj(model=model,Dataloader=dataloader)
    # prompt='China is a'
    # generated_text = model.generate(
    #     prompt=prompt,
    #     max_length=20,  # 减少长度先测试
    #     temperature=0.7,  # 降低temperature
    #     top_p=0.8,       # 降低top_p
    #     do_sample=False  # 先用贪心解码测试
    # )
    
    # print(f"\n生成结果:")
    # print(f"原始prompt: {prompt}")
    # print(f"完整生成文本: {generated_text}")
    
    # # 数据集评估示例
    # print(f"\n模型信息: {model.get_model_info()}")
    
    # # 测试forward方法
    # print(f"\n测试forward方法:")
    # test_input = model.tokenizer.encode(prompt, return_tensors='pt')
    # print(f"输入shape: {test_input.shape}")
    # with torch.no_grad():
    #     logits = model.forward(test_input)
    #     print(f"输出logits shape: {logits.shape}")
    #     print(f"预测下一个token ID: {torch.argmax(logits[0, -1]).item()}")
    #     next_token = model.tokenizer.decode([torch.argmax(logits[0, -1]).item()])
    #     print(f"预测下一个token: '{next_token}'")
