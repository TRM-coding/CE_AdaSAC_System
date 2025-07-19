




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
        del self.layers
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.layers=nn.ModuleList()
    
    def add_layer(self,layer):
        if isinstance(layer,SVDED_GPTJ_EDGE_Layer):
            layer.v_svd['U']=layer.v_svd['U'].to('cuda:0')
            layer.v_svd['V']=layer.v_svd['V'].to('cuda:0')
            layer.v_svd['bias']=layer.v_svd['bias'].to('cuda:0')
            layer.out_proj_svd['U']=layer.out_proj_svd['U'].to('cuda:0')
            layer.out_proj_svd['V']=layer.out_proj_svd['V'].to('cuda:0')
            layer.out_proj_svd['bias']=layer.out_proj_svd['bias'].to('cuda:0')
            layer.fc_in_svd['U']=layer.fc_in_svd['U'].to('cuda:0')
            layer.fc_in_svd['V']=layer.fc_in_svd['V'].to('cuda:0')
            layer.fc_in_svd['bias']=layer.fc_in_svd['bias'].to('cuda:0')
            layer.fc_out_svd['U']=layer.fc_out_svd['U'].to('cuda:0')            
            layer.fc_out_svd['V']=layer.fc_out_svd['V'].to('cuda:0')            
            layer.fc_out_svd['bias']=layer.fc_out_svd['bias'].to('cuda:0')            
        
        self.layers.append(layer.to('cuda:0'))

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

    def forward(self, input_ids, attention_mask=None,reduce_rate=0,need_embedding=True):
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
        
        # 3. 逐层处理
        for layer_idx in range(self.num_layers):
            # 云侧：计算Q、K和attention权重（传入attention_mask和position_ids）
            
            q, k, attn_weights = self.cloud.forward_no_cache(
                    x, layer_idx, position_ids, attention_mask.to(self.device_cloud)
            )
           
            # 将数据传输到边侧设备
            x_edge = x.to(self.device_edge)
            attn_weights_edge = attn_weights.to(self.device_edge)
            
            _, x_edge = self.edge.forward_no_cache(x_edge, layer_idx, attn_weights_edge)
 
    

            # 将结果传回云侧
            x = x_edge.to(self.device_cloud)
        
        # 4. 最终的Layer Norm和LM Head
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


def load_svd_layer(layer_idx,reduce_rate):
    print(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth")
    if os.path.exists(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_origin.pth"):
        newlayer=torch.load(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_origin.pth",weights_only=False,map_location='cpu')
        return newlayer
    elif os.path.exists(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth"):
        newlayer=torch.load(f"./GPTJ_SVD_DATA/gptj_svd_layer{layer_idx}_reduce_rate{reduce_rate}_svd.pth",weights_only=False,map_location='cpu')
        return newlayer
    else :
        return None

def count_flops(collaboration_model:GPTJCloudEdgeCollaborator):
    cloud=0
    edge=0
    for layer in collaboration_model.edge.layers:
        edge+=layer.flops
    for layer,flops in cloud_flops.items():
        cloud+=flops
    return cloud,edge

    

def load_batch(filepath="./GPTJ_inputbatch.pkl"):
    try:
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        print(f"数据已成功加载: {filepath}")
        
        # 打印加载信息
        if 'metadata' in data_dict:
            metadata = data_dict['metadata']
            print(f"生成时间: {metadata.get('generation_time', 'Unknown')}")
            print(f"批次大小: {metadata.get('batch_size', 'Unknown')}")
            print(f"序列长度: {metadata.get('seq_len', 'Unknown')}")
            print(f"隐藏层大小: {metadata.get('hidden_size', 'Unknown')}")
        
        print(f"Input embeddings 形状: {data_dict['shapes']['input_embeds']}")
        print(f"Target 形状: {data_dict['shapes']['target']}")
        
        return data_dict
        
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
        raise
    except Exception as e:
        print(f"加载失败: {e}")
        raise

batch_input=load_batch()['input_embeds'].half().to('cuda:0')
batch_output=load_batch()['target'].half().to('cuda:0')
import math
def get_loss(model:GPTJCloudEdgeCollaborator):
    criterion = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=batch_input,need_embedding=False)
        logits  = torch.log_softmax(outputs, dim=-1)              # [B, T, V]

        loss = criterion(logits,batch_output)

    return loss
import random
import time



def get_model(model:GPTJCloudEdgeCollaborator,reduce_list:list):
    model.edge.clear()
    layer_idx=0
    for rate in reduce_list:
        newlayer=load_svd_layer(layer_idx=layer_idx,reduce_rate=rate)
        model.edge.add_layer(newlayer)
        layer_idx=layer_idx+1
collaboration=GPTJCloudEdgeCollaborator()
EDGE_DEVICE=1e12
import numpy as np
def count_F(specice_i:dict,alpha):
    timee=specice_i['edge_time'].to('cpu')
    loss=specice_i['loss'].to('cpu')
    return (1-alpha)*np.exp(1-timee)-alpha*(np.exp(1-(loss-1)**2)-1)

def sigmoid(x):
        return (1-np.exp(-x))/(1+np.exp(-x))
def ifexpand(max_cut,cut_i,alpha)->bool:
    if(abs(cut_i-alpha*max_cut)==0):
        return True
    score=1/abs((cut_i-alpha*max_cut))
    p=sigmoid(score)
    r=random.random()
    if(r<p):
        return True
    else:
        return False

def warm_asto(warm_epoch):
    print("---------start_warm----------------")
    alpha_cp=[]
    warm_res={}
    init_species=[]
    species_map={}
    F_map={}
    init_size=100
    # 记录每轮迭代的数据
    warm_log = {
        'iterations': [],
        'total_time': 0
    }
    for i in range(init_size):
        listi=[round(random.randint(0, 8) * 0.1, 1)for j,_ in enumerate(range(28))]
        init_species.append(tuple(listi))
    for _1 in range(warm_epoch):
        # 记录当前迭代开始时间
        iteration_start = time.time()
        
        alpha=random.randint(0,int(1/0.1))*0.1
        if(len(alpha_cp)<(1//0.1)):
            alpha=len(alpha_cp)*0.1
            alpha_cp.append(0)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        F_score_list=[]
        st=time.time()
        # task_list=[]

        # for idx,speciesi in enumerate(init_species):
        #     task_list.append((speciesi,alpha))
        
        max_loss=0
        min_loss=1000000000
        max_time=0
        min_time=1000000000
        # 计算当前每个个体的时间和loss
        for speciesi in init_species:
            loss_spi=0
            edge_time=0
            if tuple(speciesi) in species_map:
                loss_spi=species_map[tuple(speciesi)]['loss']
                edge_time=species_map[tuple(speciesi)]['edge_time']
            else:
                get_model(collaboration,speciesi)
                loss_spi=get_loss(collaboration)
                # loss_spi=random.randint(1,10)*0.1
                cloud_flops,edge_flops=count_flops(collaboration)
                edge_flops=random.randint(100000,200000)
                species_map[tuple(speciesi)]={}
                species_map[tuple(speciesi)]['loss']=loss_spi
                species_map[tuple(speciesi)]['edge_time']=edge_flops/EDGE_DEVICE
                edge_time=edge_flops/EDGE_DEVICE

            max_time=max(max_time,edge_time)
            min_time=min(min_time,edge_time)
            max_loss=max(max_loss,loss_spi)
            min_loss=min(min_loss,loss_spi)
        # 计算每个个体的F得分

        for speciesi in init_species:
            if (speciesi,alpha) in F_map:
                F_score_list.append(F_map[(speciesi,alpha)])
                continue
            species_map[tuple(speciesi)]['loss']=(species_map[tuple(speciesi)]['loss']-min_loss)/(max_loss-min_loss)
            species_map[tuple(speciesi)]['edge_time']=(species_map[tuple(speciesi)]['edge_time']-min_time)/(max_time-min_time)
            F_map[(speciesi,alpha)]=count_F(species_map[tuple(speciesi)],alpha)
            F_score_list.append(F_map[(speciesi,alpha)])
        #去重后分布调整
        unique_dict=dict(zip(init_species,F_score_list))
        unique_species=list(unique_dict.keys())
        unique_F=list(unique_dict.values())
        sum_cut_list=[sum(sublist) for sublist in unique_species]
        max_cut=max(sum_cut_list)
        len_u=len(unique_species)
        for i in range(len_u):
            cut_i=sum(unique_species[i])
            if(ifexpand(max_cut=max_cut,cut_i=cut_i,alpha=(1-alpha))):
                unique_species.append(unique_species[i])
                unique_F.append(unique_F[i])

        # 淘汰个体
        cb=list(zip(unique_species,unique_F))
        cbsorted=sorted(cb,key=lambda x:x[1],reverse=True)
        cbsorted=cbsorted[:init_size]
        init_species=[x[0] for x in cbsorted]
        F_score_list=[x[1] for x in cbsorted]

        # 构造归一化积分函数
        sums=sum(F_score_list)
        for i,_ in enumerate(F_score_list):
            F_score_list[i]/=sums
        F_Integral=[]
        for i in range(len(F_score_list)):
            F_Integral.append(0)
            for j in range(i+1):
                F_Integral[i]+=F_score_list[j]
        maxx=len(init_species)

        #随机选择个体作为父代并进行交叉遗传
        for _ in range(maxx):
            r=random.random()
            father=[]
            for i,p in enumerate(F_Integral):
                if len(father)>1:
                    break
                if r<p:
                    father.append(init_species[i])
            
            if(len(father)<=1):
                continue
            
            r=random.random()
            son1=father[0][:int(len(father[0])*r)]+father[1][int(len(father[0])*r):]
            son2=father[1][:int(len(father[0])*r)]+father[0][int(len(father[0])*r):]
            init_species.append(son1)
            init_species.append(son2)
        
        # 记录当前迭代的数据
        iteration_end = time.time()
        iteration_time = iteration_end - iteration_start
        
        # 记录当前迭代的种群和适应度
        current_population = []
        for i, species in enumerate(init_species):
            current_population.append({
                'species': list(species),
                'fitness': F_score_list[i] if i < len(F_score_list) else 0
            })
        
        iteration_data = {
            'epoch': _1 + 1,
            'alpha': alpha,
            'time': iteration_time,
            'population_size': len(init_species),
            'population': current_population,
            'best_fitness': max(F_score_list) if F_score_list else 0,
            'avg_fitness': sum(F_score_list) / len(F_score_list) if F_score_list else 0
        }
        
        warm_log['iterations'].append(iteration_data)
        warm_log['total_time'] += iteration_time
        
        print(f"Warm epoch {_1+1}/{warm_epoch}: alpha={alpha:.1f}, time={iteration_time:.3f}s, "
              f"pop_size={len(init_species)}, best_fitness={max(F_score_list) if F_score_list else 0:.4f}")
    
    print(f"Warm phase completed. Total time: {warm_log['total_time']:.3f}s")
    
    # 保存warm阶段的日志
    with open('./warm_asto_log.pkl', 'wb') as f:
        pickle.dump(warm_log, f)
    
    alpha_fit={}
    alphas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for speciesi in init_species:
        max_F=-1000
        for alphai in alphas:
            if speciesi in species_map:
                if (max_F<count_F(species_map[tuple(speciesi)],alphai)):
                    alpha_fit[speciesi]=alphai
    return alpha_fit,species_map

def asto_v2(generate_epoch,alpha,init_species_,species_map_):
    print(f"start{alpha}")
    alpha_cp=[]
    warm_res={}
    init_species=init_species_
    species_map=species_map_
    F_map={}
    init_size=30
    
    # 记录每轮迭代的数据
    v2_log = {
        'alpha': alpha,
        'iterations': [],
        'total_time': 0
    }
    for i in range(init_size-len(init_species)):
        listi=[round(random.randint(0, 8) * 0.1, 1) for j,_ in enumerate(range(28))]
        init_species.append(tuple(listi))
    for _1 in range(generate_epoch):
        # 记录当前迭代开始时间
        iteration_start = time.time()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        F_score_list=[]
        st=time.time()
        
        max_loss=0
        min_loss=1000000000
        max_time=0
        min_time=1000000000
        # 计算当前每个个体的时间和loss
        for speciesi in init_species:
            loss_spi=0
            edge_time=0
            if tuple(speciesi) in species_map:
                loss_spi=species_map[tuple(speciesi)]['loss']
                edge_time=species_map[tuple(speciesi)]['edge_time']
            else:
                get_model(collaboration,speciesi)
                loss_spi=get_loss(collaboration)
                # loss_spi = random.randint(1,10)*0.1
                cloud_flops,edge_flops=count_flops(collaboration)
                edge_flops=random.randint(100000,200000)
                species_map[tuple(speciesi)]={}
                species_map[tuple(speciesi)]['loss']=loss_spi
                species_map[tuple(speciesi)]['edge_time']=edge_flops/EDGE_DEVICE
                edge_time=edge_flops/EDGE_DEVICE

            max_time=max(max_time,edge_time)
            min_time=min(min_time,edge_time)
            max_loss=max(max_loss,loss_spi)
            min_loss=min(min_loss,loss_spi)
        # 计算每个个体的F得分

        for speciesi in init_species:
            if (speciesi,alpha) in F_map:
                F_score_list.append(F_map[(speciesi,alpha)])
                continue
            species_map[tuple(speciesi)]['loss']=(species_map[tuple(speciesi)]['loss']-min_loss)/(max_loss-min_loss)
            species_map[tuple(speciesi)]['edge_time']=(species_map[tuple(speciesi)]['edge_time']-min_time)/(max_time-min_time)
            F_map[(speciesi,alpha)]=count_F(species_map[tuple(speciesi)],alpha)
            F_score_list.append(F_map[(speciesi,alpha)])
        #去重后分布调整
        unique_dict=dict(zip(init_species,F_score_list))
        unique_species=list(unique_dict.keys())
        unique_F=list(unique_dict.values())
        sum_cut_list=[sum(sublist) for sublist in unique_species]
        max_cut=max(sum_cut_list)
        len_u=len(unique_species)
        for i in range(len_u):
            cut_i=sum(unique_species[i])
            if(ifexpand(max_cut=max_cut,cut_i=cut_i,alpha=(1-alpha))):
                unique_species.append(unique_species[i])
                unique_F.append(unique_F[i])

        # 淘汰个体
        cb=list(zip(unique_species,unique_F))
        cbsorted=sorted(cb,key=lambda x:x[1],reverse=True)
        cbsorted=cbsorted[:init_size]
        init_species=[x[0] for x in cbsorted]
        F_score_list=[x[1] for x in cbsorted]

        # 构造归一化积分函数
        sums=sum(F_score_list)
        for i,_ in enumerate(F_score_list):
            F_score_list[i]/=sums
        F_Integral=[]
        for i in range(len(F_score_list)):
            F_Integral.append(0)
            for j in range(i+1):
                F_Integral[i]+=F_score_list[j]
        maxx=len(init_species)

        #随机选择个体作为父代并进行交叉遗传
        for _ in range(maxx):
            r=random.random()
            father=[]
            for i,p in enumerate(F_Integral):
                if len(father)>1:
                    break
                if r<p:
                    father.append(init_species[i])
            
            if(len(father)<=1):
                continue
            
            r=random.random()
            son1=father[0][:int(len(father[0])*r)]+father[1][int(len(father[0])*r):]
            son2=father[1][:int(len(father[0])*r)]+father[0][int(len(father[0])*r):]
            init_species.append(son1)
            init_species.append(son2)
        
        # 记录当前迭代的数据
        iteration_end = time.time()
        iteration_time = iteration_end - iteration_start
        
        # 记录当前迭代的种群和适应度
        current_population = []
        for i, species in enumerate(init_species):
            current_population.append({
                'species': list(species),
                'fitness': F_score_list[i] if i < len(F_score_list) else 0
            })
        
        iteration_data = {
            'epoch': _1 + 1,
            'time': iteration_time,
            'population_size': len(init_species),
            'population': current_population,
            'best_fitness': max(F_score_list) if F_score_list else 0,
            'avg_fitness': sum(F_score_list) / len(F_score_list) if F_score_list else 0
        }
        
        v2_log['iterations'].append(iteration_data)
        v2_log['total_time'] += iteration_time
        
        print(f"Generate epoch {_1+1}/{generate_epoch} (alpha={alpha:.1f}): time={iteration_time:.3f}s, "
              f"pop_size={len(init_species)}, best_fitness={max(F_score_list) if F_score_list else 0:.4f}")
    
    print(f"Generate phase completed for alpha={alpha:.1f}. Total time: {v2_log['total_time']:.3f}s")
    
    # 保存v2阶段的日志
    with open(f'./asto_v2_alpha_{alpha:.1f}_log.pkl', 'wb') as f:
        pickle.dump(v2_log, f)
    
    ans=0
    max_F=-1000
    for speicei in init_species:
        if speicei in species_map:
            F=count_F(species_map[tuple(speciesi)],alpha)
            if(F>max_F):
                ans=speicei
    return ans
            

def asto(warm_epoch,generate_epoch):
    """
    asto算法主函数，包含完整的日志记录功能
    
    Args:
        warm_epoch: 热身阶段的迭代次数
        generate_epoch: 生成阶段的迭代次数
    
    Returns:
        dict: 包含所有阶段日志的完整记录
    """
    print("="*50)
    print("Starting ASTO Algorithm with Logging")
    print(f"Warm epochs: {warm_epoch}")
    print(f"Generate epochs: {generate_epoch}")
    print("="*50)
    
    # 记录整个算法的开始时间
    total_start_time = time.time()
    
    # 执行热身阶段
    alpha_fit,species_map=warm_asto(warm_epoch)
    
    # 记录生成阶段的结果
    generate_results = {}
    alpha=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    for alphai in alpha:
        print(f"\n--- Processing alpha = {alphai:.1f} ---")
        init_spi=[]
        for key,value in alpha_fit.items():
            if value==alphai:
                init_spi.append(key)
        
        if not init_spi:
            print(f"No species found for alpha = {alphai:.1f}, skipping...")
            continue
            
        ans=asto_v2(generate_epoch,alphai,init_spi,species_map)
        generate_results[alphai] = {
            'initial_species': init_spi,
            'best_solution': ans
        }
        print(f"Best solution for alpha {alphai:.1f}:", ans)
    
    # 记录总时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 创建完整的日志记录
    complete_log = {
        'algorithm': 'ASTO',
        'parameters': {
            'warm_epochs': warm_epoch,
            'generate_epochs': generate_epoch,
        },
        'total_time': total_time,
        'warm_results': alpha_fit,
        'generate_results': generate_results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    
    # 保存完整的日志
    with open('./asto_complete_log.pkl', 'wb') as f:
        pickle.dump(complete_log, f)
    
    print("\n" + "="*50)
    print("ASTO Algorithm Completed")
    print(f"Total execution time: {total_time:.3f}s")
    print("Log files saved:")
    print("  - ./warm_asto_log.pkl (warm phase details)")
    print("  - ./asto_v2_alpha_X.X_log.pkl (generate phase details for each alpha)")
    print("  - ./asto_complete_log.pkl (complete algorithm summary)")
    print("="*50)
    
    return complete_log


def analyze_asto_logs():
    """
    分析ASTO算法的日志数据
    """
    try:
        # 读取完整日志
        with open('./asto_complete_log.pkl', 'rb') as f:
            complete_log = pickle.load(f)
        
        print("ASTO Algorithm Analysis")
        print("="*50)
        print(f"Algorithm: {complete_log['algorithm']}")
        print(f"Execution time: {complete_log['timestamp']}")
        print(f"Total runtime: {complete_log['total_time']:.3f}s")
        print(f"Warm epochs: {complete_log['parameters']['warm_epochs']}")
        print(f"Generate epochs: {complete_log['parameters']['generate_epochs']}")
        
        # 分析热身阶段
        try:
            with open('./warm_asto_log.pkl', 'rb') as f:
                warm_log = pickle.load(f)
            
            print("\nWarm Phase Analysis:")
            print(f"  Total warm time: {warm_log['total_time']:.3f}s")
            print(f"  Number of iterations: {len(warm_log['iterations'])}")
            print(f"  Average time per iteration: {warm_log['total_time']/len(warm_log['iterations']):.3f}s")
            
            # 分析每轮的最佳适应度变化
            best_fitness_history = [iter_data['best_fitness'] for iter_data in warm_log['iterations']]
            print(f"  Best fitness progression: {best_fitness_history[:5]}...{best_fitness_history[-5:]}")
            
        except FileNotFoundError:
            print("Warm phase log not found")
        
        # 分析生成阶段
        print("\nGenerate Phase Analysis:")
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for alpha in alpha_values:
            try:
                with open(f'./asto_v2_alpha_{alpha:.1f}_log.pkl', 'rb') as f:
                    v2_log = pickle.load(f)
                
                print(f"  Alpha {alpha:.1f}:")
                print(f"    Total time: {v2_log['total_time']:.3f}s")
                print(f"    Iterations: {len(v2_log['iterations'])}")
                if v2_log['iterations']:
                    final_best = v2_log['iterations'][-1]['best_fitness']
                    print(f"    Final best fitness: {final_best:.4f}")
                    
            except FileNotFoundError:
                print(f"  Alpha {alpha:.1f}: No log found")
        
        return complete_log
        
    except FileNotFoundError:
        print("Complete log file not found. Please run the ASTO algorithm first.")
        return None


def save_detailed_population_history():
    """
    保存详细的种群历史数据到CSV文件，便于后续分析
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available. Install with: pip install pandas")
        return
    
    try:
        # 读取热身阶段数据
        with open('./warm_asto_log.pkl', 'rb') as f:
            warm_log = pickle.load(f)
        
        # 创建热身阶段的详细数据
        warm_data = []
        for iter_data in warm_log['iterations']:
            for i, pop in enumerate(iter_data['population']):
                warm_data.append({
                    'phase': 'warm',
                    'epoch': iter_data['epoch'],
                    'alpha': iter_data['alpha'],
                    'individual_id': i,
                    'species': str(pop['species']),
                    'fitness': pop['fitness'],
                    'iteration_time': iter_data['time']
                })
        
        warm_df = pd.DataFrame(warm_data)
        warm_df.to_csv('./warm_phase_detailed.csv', index=False)
        print("Warm phase detailed data saved to: ./warm_phase_detailed.csv")
        
        # 处理生成阶段数据
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for alpha in alpha_values:
            try:
                with open(f'./asto_v2_alpha_{alpha:.1f}_log.pkl', 'rb') as f:
                    v2_log = pickle.load(f)
                
                generate_data = []
                for iter_data in v2_log['iterations']:
                    for i, pop in enumerate(iter_data['population']):
                        generate_data.append({
                            'phase': 'generate',
                            'alpha': alpha,
                            'epoch': iter_data['epoch'],
                            'individual_id': i,
                            'species': str(pop['species']),
                            'fitness': pop['fitness'],
                            'iteration_time': iter_data['time']
                        })
                
                if generate_data:
                    generate_df = pd.DataFrame(generate_data)
                    generate_df.to_csv(f'./generate_alpha_{alpha:.1f}_detailed.csv', index=False)
                    print(f"Generate phase (alpha={alpha:.1f}) data saved to: ./generate_alpha_{alpha:.1f}_detailed.csv")
                    
            except FileNotFoundError:
                continue
        
        print("All detailed population history saved successfully!")
        
    except Exception as e:
        print(f"Error saving detailed data: {e}")



if __name__=="__main__":
    
    # rates=[0.0 for _ in range(28)]
    # get_model(collaboration)
    #     # collaboration.generate("BJTU is a university that")
    # loss=get_loss(collaboration)
    # print(loss)
    # input()
    # collaboration.edge.clear()
    # input()
    # print(cloud_flops)

    asto(2,2)

    