import pickle
import torch
import os
from detection.Loader.mymodel_file.gptJ_edge import gptJ_edge_layer
from detection.Loader.mymodel_file.gptJ_cloud import gptJ_cloud
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from modelscope.utils.hub import snapshot_download
from transformers import AutoTokenizer
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer
import os
import gc
import copy

datafolder="./GPTJ_SVD_DATA"
cloud_flops=0
net_dict=0
with open(f"{datafolder}/cloud_flops_dict.pickle","rb") as f:
    cloud_flops=pickle.load(f)

with open(f"{datafolder}/net_dict.pickle","rb") as f:
    net_dict=pickle.load(f)


class gptJ_edge(nn.Module):
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b',svd=False):
        super().__init__()
        
        # 如果是 HuggingFace 仓库名，使用 ModelScope 下载
        if not os.path.exists(model_name):
            print(f"Downloading model {model_name} using ModelScope...")
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir='./gpt-j-6b'
            )
        else:
            model_path = model_name

        self.layers = nn.ModuleList()
        self.num_layers = 28

    
    def forward_no_cache(self, x, layer_idx, attn_weights):
        return self.layers[layer_idx].forward_no_cache(x, attn_weights)
    
    # 在 gptJ_edge_layer 类中添加 clear 方法
    def clear(self):
        for layeri in self.layers:
            if isinstance(layeri,gptJ_edge_layer):
                del layeri
                continue
            del layeri.v_svd['U']
            del layeri.v_svd['V']
            del layeri.v_svd['bias']
            del layeri.out_proj_svd['U']
            del layeri.out_proj_svd['V']
            del layeri.out_proj_svd['bias']
            del layeri.fc_in_svd['U']
            del layeri.fc_in_svd['V']
            del layeri.fc_in_svd['bias']
            del layeri.fc_out_svd['U']           
            del layeri.fc_out_svd['V']        
            del layeri.fc_out_svd['bias']
            del layeri
        del self.layers
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.layers=nn.ModuleList()
    
    def add_layer(self,layeri,device,dtype=torch.float16):
        layer=copy.deepcopy(layeri)
        # layer=layer.to(dtype)
        
        if isinstance(layer,SVDED_GPTJ_EDGE_Layer):
            layer.v_svd['U']=layer.v_svd['U'].to(device=device,dtype=dtype)
            layer.v_svd['V']=layer.v_svd['V'].to(device=device,dtype=dtype)
            layer.v_svd['bias']=layer.v_svd['bias'].to(device=device,dtype=dtype)
            layer.out_proj_svd['U']=layer.out_proj_svd['U'].to(device=device,dtype=dtype)
            layer.out_proj_svd['V']=layer.out_proj_svd['V'].to(device=device,dtype=dtype)
            layer.out_proj_svd['bias']=layer.out_proj_svd['bias'].to(device=device,dtype=dtype)
            layer.fc_in_svd['U']=layer.fc_in_svd['U'].to(device=device,dtype=dtype)
            layer.fc_in_svd['V']=layer.fc_in_svd['V'].to(device=device,dtype=dtype)
            layer.fc_in_svd['bias']=layer.fc_in_svd['bias'].to(device=device,dtype=dtype)
            layer.fc_out_svd['U']=layer.fc_out_svd['U'].to(device=device,dtype=dtype)            
            layer.fc_out_svd['V']=layer.fc_out_svd['V'].to(device=device,dtype=dtype)            
            layer.fc_out_svd['bias']=layer.fc_out_svd['bias'].to(device=device,dtype=dtype)
        
        # 将整个层移动到设备，并确保数据类型
        layer = layer.to(device=device,  )
        self.layers.append(layer)
        gc.collect()
        return
from torch.profiler import profile, ProfilerActivity

class GPTJCloudEdgeCollaborator(nn.Module):
    """
    GPT-J 云边协同模型
    云侧：完成Q、K的计算和attention权重计算
    边侧：完成V的计算和最终的attention输出
    """
    
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0'):
        super().__init__()
        
        self.device_cloud = device_cloud
        self.device_edge = device_cloud
        self.flops_edge=0
        self.flops_cloud=0
        
        # 初始化云侧和边侧模型
        print(f"初始化云侧模型 (设备: {device_cloud})...")
        self.cloud = gptJ_cloud(model_name=model_name,dtype=torch.int8)
        self.cloud =self.cloud.to(device_cloud)
        
        print(f"初始化边侧模型 (设备: {device_cloud})...")
        self.edge = gptJ_edge(model_name=model_name,svd=True).to(device_cloud)
        
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

        # 标记是否使用SVD
        self.use_svd = False
        self.svd_layers = None
        self.origin_layers= []

    def forward(self, input_ids, attention_mask=None,reduce_rate=0,need_embedding=True,flops_eval=False):
        """
        前向传播用于数据集评估
        Args:
            input_ids: [batch_size, seq_len] token ids
            attention_mask: [batch_size, seq_len] attention mask (1=valid, 0=padding)
        Returns:
            logits: [batch_size, seq_len, vocab_size] 预测logits
        """
        # 1. Embedding
        x=0
        batch_size=0
        seq_len=0
        if need_embedding:
            x = self.embed(input_ids.to(self.device_cloud))  # [B, T, D]
        
            batch_size, seq_len = input_ids.shape
        else:
            batch_size=input_ids.shape[0]
            seq_len=input_ids.shape[1]
            x=input_ids
        
        # 2. 如果没有提供attention_mask，根据pad_token_id生成
        temp_=torch.ones(input_ids.shape[0],input_ids.shape[1])
        if attention_mask is None:
            if self.tokenizer.pad_token_id is not None:
                attention_mask = (temp_ != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones(input_ids.shape[0],input_ids.shape[1])
        
        # 创建position_ids
        position_ids = torch.arange(seq_len, device=self.device_cloud).unsqueeze(0).expand(batch_size, -1)
        
        prof_cloud = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        )
        prof_edge = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        )

        # 3. 逐层处理
        for layer_idx in range(self.num_layers):
            # 云侧：计算Q、K和attention权重（传入attention_mask和position_ids）
            
            if(flops_eval):
                prof_cloud.start()
            q, k, attn_weights = self.cloud.forward_no_cache(
                    x, layer_idx, position_ids, attention_mask.to(self.device_cloud)
            )
            if flops_eval:
                prof_cloud.stop()
           
            # 将数据传输到边侧设备
            x_edge = x.to(self.device_edge)
            attn_weights_edge = attn_weights.to(self.device_edge)
            

            if flops_eval:
                prof_edge.start()
            _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
            if flops_eval:
                prof_edge.stop()
    

            # 将结果传回云侧
            x = x_edge.to(self.device_cloud)
        
        # 4. 最终的Layer Norm和LM Head
        if flops_eval:
            self.flops_cloud = sum(evt.flops for evt in prof_cloud.key_averages())
            self.flops_edge =sum(evt.flops for evt in prof_edge.key_averages())
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, prompt, max_length=50, temperature=1.0, top_p=0.9, do_sample=True):
        """文本生成方法 - 修复版本"""
        self.eval()
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs = input_ids.copy()
        
        print(f"开始生成，初始prompt: '{prompt}'")
        print(f"目标生成长度: {max_length} tokens")
        print(f"初始token数: {len(input_ids)}")
        print(f"使用SVD压缩: {self.use_svd}")
        
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
              
                    q, k, attn_weights = self.cloud.forward_no_cache(
                        x, layer_idx, position_ids, attention_mask
                    )
                   
                    
                    # 数据传输到边侧
                 
                    x_edge = x.to(self.device_edge)
                    attn_weights_edge = attn_weights.to(self.device_edge)
                   
                    
                    # 边侧计算
                 
                    # if self.use_svd:
                    #     # 使用SVD压缩层
                    #     x_edge = self.svd_layers[layer_idx].forward_no_cache(x_edge, attn_weights_edge)
                    # else:
                    #     # 使用原始层
                    #     _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
                    _, x_edge = self.edge.forward_no_cache(x_edge,layer_idx,attn_weights_edge)
                    
                    x = x_edge
                    
                    torch.cuda.empty_cache()
                
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
            'model_name': 'GPT-J Cloud-Edge Collaborator',
            'use_svd': self.use_svd,
            'svd_enabled_layers': self.num_layers if self.use_svd else 0
        }

    def get_compression_info(self):
        """获取压缩信息"""
        if not self.use_svd:
            return {"compressed": False}
        
        compression_info = {
            "compressed": True,
            "total_layers": self.num_layers,
            "layer_details": []
        }
        
        for i, layer in enumerate(self.edge.layers):
            layer_info = {
                "layer_idx": i,
                # "reduce_rate": layer.reduce_rate,
                # 可以添加更多SVD相关信息
            }
            compression_info["layer_details"].append(layer_info)
        
        return compression_info





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
svd_layers={}
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
    avg_loss=0
    perplexity=0
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

def load_svd_layer(layer_idx,reduce_rate):
    # print(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth")
    gc.collect()
    if((layer_idx,reduce_rate) in svd_layers):
        return svd_layers[(layer_idx,reduce_rate)]
    if os.path.exists(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_origin.pth"):
        newlayer=torch.load(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_origin.pth",weights_only=False,map_location='cpu')
        if hasattr(newlayer,"original_layer"):
            del newlayer.original_layer
        newlayer=newlayer.half()
        svd_layers[(layer_idx,reduce_rate)]=newlayer
        return newlayer
    elif os.path.exists(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth"):
        newlayer=torch.load(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth",weights_only=False,map_location='cpu')
        if hasattr(newlayer,"original_layer"):
            del newlayer.original_layer
        newlayer=newlayer.half()
        svd_layers[(layer_idx,reduce_rate)]=newlayer
        return newlayer
    else :
        return None

def get_model(model:GPTJCloudEdgeCollaborator,reduce_list:list):
    try:
        model.edge.clear()
        device=model.device_cloud
        layer_idx=0
        for rate in reduce_list:
            newlayer=load_svd_layer(layer_idx=layer_idx,reduce_rate=rate)
            model.edge.add_layer(newlayer,device,dtype=torch.int8)
            layer_idx=layer_idx+1
    except Exception as e:
        print("Exceptions in get_model:")
        print(e)
        # traceback.print_exc()
    # model.edge.clear()



# 示例使用
if __name__ == "__main__":
    # 方法1：创建普通模型后手动应用压缩
    model=GPTJCloudEdgeCollaborator(device_cloud=f'cuda:{0}')
    
    # loss_spi=get_loss(collaboration,batch_input,batch_output)
    
    # 定义每层的压缩率（28层，每层压缩率不同）\
    #alpha=0.9
    reduce_rates=[0.1,0.2,0.1,0.1,0.3,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.1,0.3,0.1,0.1,0.1,0.2,0.4,0.1,0.1,0.1,0.3,0.1,0.1,0.4,0.1,0.2]
    get_model(model,reduce_rates)
    # 应用SVD压缩
    # model.apply_svd_compression(reduce_rates, svd_device='cuda:0')
    

    
    # 生成文本测试
    prompt = "The future of artificial intelligence is"
    generated_text = model.generate(
        prompt=prompt,
        max_length=20,
        temperature=0.7,
        top_p=0.8,
        do_sample=False
    )
    
    print(f"压缩信息: {model.get_compression_info()}")
    print(f"生成结果: {generated_text}")

    dataloader=load_and_tokenize_dataset(tokenizer=model.tokenizer)
    loss=evaluate_minipile_gptj(model=model,Dataloader=dataloader)
    print(loss)
    first_batch = next(iter(dataloader))
    # 拿到完整的 input_ids, attention_mask, 和已经被 collator 设好 -100 的 labels
    input_ids    = first_batch['input_ids'].to('cuda:0')       
        # [B, T], pad 已经是 -100

    # input=torch.randint(1,100)
    model.forward(input_ids=input,flops_eval=True)
    print("flops_edge:",model.flops_edge)
    print("flops_cloud:",model.flops_cloud)


