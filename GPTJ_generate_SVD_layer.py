import torch
from torch import nn
from transformers import AutoTokenizer
import time
from detection.Loader.mymodel_file.gptJ_cloud import gptJ_cloud
from detection.Loader.mymodel_file.gptJ_edge import gptJ_edge
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import os

cloud_flops_dict={}
net_dict={}

class GPTJCloudEdgeCollaborator(nn.Module):
    """
    GPT-J 云边协同模型
    云侧：完成Q、K的计算和attention权重计算
    边侧：完成V的计算和最终的attention输出
    """
    
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cuda:0'):
        super().__init__()
        
        self.device_cloud = device_cloud
        self.device_edge = device_edge
        
        # 初始化云侧和边侧模型
        print(f"初始化云侧模型 (设备: {device_cloud})...")
        self.cloud = gptJ_cloud(model_name=model_name).to(device_cloud)
        
        print(f"初始化边侧模型 (设备: {device_edge})...")
        self.edge = gptJ_edge(model_name=model_name,svd=True).to(device_edge)
        
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

    def apply_svd_compression(self, reduce_rates, svd_device='cuda:0'):
        """
        对边侧模型应用SVD压缩
        Args:
            reduce_rates: list[float] - 每一层的压缩率，长度应等于模型层数
            svd_device: str - 进行SVD分解的设备，默认为'cpu'
        Returns:
            self - 返回自身以支持链式调用
        """
        print(f"开始对边侧模型应用SVD压缩...")
        print(f"模型层数: {self.num_layers}")
        print(f"压缩率: {reduce_rates}")
        print(f"SVD分解设备: {svd_device}")
        
        # 检查压缩率列表长度
        if len(reduce_rates) != self.num_layers:
            raise ValueError(f"压缩率列表长度 ({len(reduce_rates)}) 不等于模型层数 ({self.num_layers})")
        
        # 检查压缩率范围
        for i, rate in enumerate(reduce_rates):
            if not 0 <= rate <= 1:
                raise ValueError(f"第{i}层的压缩率 {rate} 不在有效范围 [0, 1] 内")
        
        # 创建SVD压缩后的层
        # self.svd_layers = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            print(f"正在压缩第 {layer_idx+1}/{self.num_layers} 层，压缩率: {reduce_rates[layer_idx]:.3f}")
            
            # 获取原始边侧层
            original_layer = self.edge.layers[layer_idx]
            if(layer_idx>=len(self.origin_layers)):
                print(f'保留第{layer_idx}层原始层')
                # neworigin=original_layer.to('cpu')
                self.origin_layers.append(original_layer)
            # 创建SVD压缩层
            svd_layer=None
            if(reduce_rates[layer_idx]==0):
                svd_layer=original_layer
            else:
                svd_layer = SVDED_GPTJ_EDGE_Layer(
                    gptj_edge_layer=original_layer,
                    reduce_rate=reduce_rates[layer_idx],
                    device=self.device_edge,
                    svd_device=svd_device
                )
            self.edge.layers[layer_idx]=svd_layer
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 清理原始层以节省内存
            # original_layer.clear()
        
        # 标记使用SVD
        self.use_svd = True
        
        # 清理原始edge模型以节省内存
        # del self.edge.layers
        torch.cuda.empty_cache()
        
        print("SVD压缩完成!")
        return self

    def forward(self, input_ids, attention_mask=None,reduce_rate=0):
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
            if os.path.exists(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}.pth"):
                continue
            # 云侧：计算Q、K和attention权重（传入attention_mask和position_ids）
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 监控 CPU 和 CUDA
                record_shapes=True,      # 记录张量形状
                with_flops=True,         # 计算 FLOPs
                with_stack=True,         # 记录调用栈
                profile_memory=True      # 记录内存使用
            ) as prof_cloud:
                q, k, attn_weights = self.cloud.forward_no_cache(
                    x, layer_idx, position_ids, attention_mask.to(self.device_cloud)
                )
            events = prof_cloud.events()
            total_flops_cloud = sum(
                                        event.flops
                                        for event in events
                                        if event.flops is not None and event.flops > 0
                                    )
            cloud_flops_dict[layer_idx]=total_flops_cloud/batch_size/seq_len/seq_len
            
            net_dict[layer_idx]=x.numel()*x.element_size()+seq_len
            # 将数据传输到边侧设备
            x_edge = x.to(self.device_edge)
            attn_weights_edge = attn_weights.to(self.device_edge)
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 监控 CPU 和 CUDA
                record_shapes=True,      # 记录张量形状
                with_flops=True,         # 计算 FLOPs
                with_stack=True,         # 记录调用栈
                profile_memory=True      # 记录内存使用
            ) as prof_edge:
                if next(self.edge.parameters()).device == 'cpu':
                    self.edge=self.edge.to('cuda:0')
                _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
            # self.edge=self.edge.to('cpu')
            events = prof_edge.events()
            total_flops_edge = sum(
                                            event.flops
                                            for event in events
                                            if event.flops is not None and event.flops > 0
                                        )
            self.edge.layers[layer_idx].flops=total_flops_edge/seq_len/seq_len

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 监控 CPU 和 CUDA
                record_shapes=True,      # 记录张量形状
                with_flops=True,         # 计算 FLOPs
                with_stack=True,         # 记录调用栈
                profile_memory=True      # 记录内存使用
            ) as prof_edge_origin:
                if next(self.origin_layers[layer_idx].parameters()).device=='cpu':
                    self.origin_layers[layer_idx]=self.origin_layers[layer_idx].to('cuda:0')
                _,_ = self.origin_layers[layer_idx].forward_no_cache(x_edge,attn_weights_edge)
            # self.origin_layers[layer_idx]=self.origin_layers[layer_idx].to('cpu')
            events=prof_edge_origin.events()
            total_flops_edge_origin = sum(
                                            event.flops
                                            for event in events
                                            if event.flops is not None and event.flops > 0
                                        )
            self.origin_layers[layer_idx].flops=total_flops_edge_origin/seq_len/seq_len
            
            if(self.origin_layers[layer_idx].flops<self.edge.layers[layer_idx].flops):
                print(f"Origin Model Layer SAVED :{layer_idx} in flops:{self.origin_layers[layer_idx].flops}")
                torch.save(self.origin_layers[layer_idx],f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_origin.pth")
            else:
                print(f"SVDED Model Layer SAVED :{layer_idx} in flops:{self.edge.layers[layer_idx].flops}")
                torch.save(self.edge.layers[layer_idx],f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth")

            # 将结果传回云侧
            x = x_edge.to(self.device_cloud)
        
        # 4. 最终的Layer Norm和LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        import pickle
        with open('./GPTJ_SVD_DATA/net_dict.pickle', 'wb') as f:
            pickle.dump(net_dict, f, protocol=pickle.HIGHEST_PROTOCOL) 
        with open('./GPTJ_SVD_DATA/cloud_flops_dict.pickle', 'wb') as f:
            pickle.dump(cloud_flops_dict, f, protocol=pickle.HIGHEST_PROTOCOL) 
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
                    # if self.use_svd:
                    #     # 使用SVD压缩层
                    #     x_edge = self.svd_layers[layer_idx].forward_no_cache(x_edge, attn_weights_edge)
                    # else:
                    #     # 使用原始层
                    #     _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
                    _, x_edge = self.edge.forward_no_cache(x_edge,layer_idx,attn_weights_edge)
                    edge_time += time.time() - t2
                    
                    # 数据传回云侧
                    t3 = time.time()
                    x = x_edge
                    transfer_time += time.time() - t3
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


def create_compressed_gptj_model(model_name='AI-ModelScope/gpt-j-6b',
                                device_cloud='cuda:0',
                                device_edge='cuda:0',
                                reduce_rates=None,
                                svd_device='cuda:0'):
    """
    创建压缩版GPT-J云边协同模型的工厂函数
    
    Args:
        model_name: 模型名称或路径
        device_cloud: 云侧设备
        device_edge: 边侧设备
        reduce_rates: list[float] - 每层的压缩率，如果为None则使用默认值
        svd_device: SVD分解使用的设备
    
    Returns:
        GPTJCloudEdgeCollaborator: 压缩后的云边协同模型实例
    """
    # 创建基础模型
    model = GPTJCloudEdgeCollaborator(
        model_name=model_name,
        device_cloud=device_cloud,
        device_edge=device_edge
    )
    
    # 如果没有提供压缩率，使用默认值（每层压缩30%）
    if reduce_rates is None:
        reduce_rates = [0.3] * model.num_layers
        print(f"使用默认压缩率: 每层压缩30%")
    
    # 应用SVD压缩
    model.apply_svd_compression(reduce_rates, svd_device)
    
    return model



import sys
# 示例使用
if __name__ == "__main__":
    # 方法1：创建普通模型后手动应用压缩
    
    
    # 定义每层的压缩率（28层，每层压缩率不同）
    # reduce_rates = [0.1] *  + [0.0] * 24  # 前14层压缩20%，后14层压缩30%
    # rates=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    args = float(sys.argv[1])
    reduce_list=[]
    
    reduce_list.append([args for _ in range(28)])
    
    # 应用SVD压缩
    tensor=torch.ones(1,1,dtype=torch.long,device='cuda:0')
    for reduce_rates in reduce_list:
        model = create_gptj_cloud_edge_model(
            device_cloud='cuda:0',
            device_edge='cuda:0'
        ).to('cuda:0')
        model.apply_svd_compression(reduce_rates, svd_device='cuda:0')
        model.forward(tensor,reduce_rate=reduce_rates[0])
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # # 方法2：直接创建压缩模型
    # compressed_model = create_compressed_gptj_model(
    #     device_cloud='cuda:0',
    #     device_edge='cuda:0',
    #     reduce_rates=[0.25] * 28,  # 每层压缩25%
    #     svd_device='cuda:0'
    # )
    
    
   

    


