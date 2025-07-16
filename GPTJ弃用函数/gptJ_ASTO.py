import random
import time
import torch
from torch import nn
import gc
from detection.Loader.mymodel_file.gptJ_edge import gptJ_edge
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer
from detection.Loader.mymodel_file.gptJ_cloud import gptJ_cloud
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from torch.profiler import profile, record_function, ProfilerActivity
#################################################
generate_epoch=10
ALPHASTEP=0.1
model_name='AI-ModelScope/gpt-j-6b'
device_cloud='cuda:0'
device_edge='cuda:0'
original_edge = gptJ_edge(model_name=model_name).to(device_edge)
num_layers    = original_edge.num_layers
del original_edge

gc.collect()


#################################################
class SVD_GPTJ_Edge_Model(nn.Module):
    """包含所有SVD层的完整edge模型，兼容原始edge模型接口"""
    def __init__(self, original_edge, svd_reduce_rate, device='cpu', svd_device='cpu',No_init=False):
        super().__init__()
        self.device = device
        self.svd_device = svd_device
        self.num_layers = original_edge.num_layers
        self.max_ctx = original_edge.max_ctx
        self.v_cache = [None] * self.num_layers
        
        print(f"🔄 开始SVD分解处理，压缩率: {svd_reduce_rate}")
        print(f"📊 总共需要处理 {self.num_layers} 层...")
        print(f"⚡ SVD分解设备: {svd_device}, 运行设备: {device}")
        
        # 用SVD压缩的层替换原始edge层
        self.svd_layers = nn.ModuleList()
        if(not No_init):
            for i in range(self.num_layers):
                print(f"  处理第 {i+1}/{self.num_layers} 层: ", end="")
                original_edge_layer = original_edge.layers[i]
                    # 奇数层跳过压缩
                if isinstance(svd_reduce_rate, list):

                    svd_layer = SVDED_GPTJ_EDGE_Layer(
                        gptj_edge_layer=original_edge_layer,
                        reduce_rate=svd_reduce_rate[i],
                        device=device,
                        svd_device=svd_device
                    )
                else:
                    svd_layer = SVDED_GPTJ_EDGE_Layer(
                        gptj_edge_layer=original_edge_layer,
                        reduce_rate=svd_reduce_rate,
                        device=device,
                        svd_device=svd_device
                    )
                print("跳过压缩 (奇数层)")
                self.svd_layers.append(svd_layer)
        
        print(f"🎉 所有层的SVD分解处理完成！")
    
    def forward_no_cache(self,x,layer_idx,attn_weights):
        output=self.svd_layers[layer_idx].forward_no_cache(
            x,  attn_weights
        )
        return output
    
    def forward_cache(self, x, layer_idx, attn_weights):
        
        self.v_cache[layer_idx], output_x = self.svd_layers[layer_idx].forward_cache(
            x, self.v_cache[layer_idx], attn_weights
        )
        if self.v_cache[layer_idx] is not None and self.v_cache[layer_idx].size(1) > self.max_ctx:
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, -self.max_ctx:, :]
        
        return self.v_cache[layer_idx], output_x


class GPTJPipeline(nn.Module):

    def performance_init(self):
        self.cloud_flops_per_s = 1e13  # Cloud hardware FLOPs/s
        self.edge_flops_per_s = 1e10    # Edge hardware FLOPs/s
        self.bandwidth = 20971520  # in B/s  20MB

        # Initialize accumulated statistics
        self.total_flops_cloud = 0
        self.total_flops_edge = 0
        self.total_data_transfer = 0  # in MB

    def reset_stats(self):
        """Reset all accumulated stats."""
        self.total_flops_cloud = 0
        self.total_flops_edge = 0
        self.total_data_transfer = 0

    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cpu', svd_reduce_rate=0.5, use_compile=True,edge=None):
        super(GPTJPipeline, self).__init__()
        self.performance_init()
        
       
        # 使用 ModelScope 下载模型
        print(f"📥 使用ModelScope下载模型 {model_name}...")
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir='./gpt-j-6b'
        )
        print(f"✅ 模型下载完成，路径: {model_dir}")
        
        # 使用本地模型路径加载 tokenizer
        print(f"🔤 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 设置 pad_token 为 eos_token（GPT-J 没有 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"☁️  加载云端模型到 {device_cloud}...")
        self.cloud = gptJ_cloud(model_name=model_dir).to(device_cloud)
        print(f"🖥️  加载边缘模型到CPU...")
        # 强制 edge 放在 CPU
        original_edge = gptJ_edge(model_name=model_dir).to('cpu')
        self.embed = self.cloud.model.transformer.wte
        self.ln_f = self.cloud.model.transformer.ln_f
        self.lm_head = self.cloud.model.lm_head
        self.num_layers = len(self.cloud.q_weights)
        
        print(f"📊 模型共有 {self.num_layers} 层")
        
        # SVD压缩参数
        self.svd_reduce_rate = svd_reduce_rate
        self.use_compile = use_compile
        
        # 创建整个SVD edge模型
        print(f"🔧 创建SVD边缘模型...")
        # 如果有GPU，先在GPU上进行SVD分解，然后移到CPU
        svd_device = device_cloud if torch.cuda.is_available() else 'cpu'
        print(f"🔧 SVD分解将在 {svd_device} 上进行...")
        
        if(svd_reduce_rate!=-1):
            self.edge = SVD_GPTJ_Edge_Model(
                original_edge=original_edge,
                svd_reduce_rate=svd_reduce_rate,
                device='cpu',  # 最终运行在CPU上
                svd_device=svd_device  # 但SVD分解在GPU上进行
            )
        else:
            self.edge=self.edge = SVD_GPTJ_Edge_Model(
                original_edge=original_edge,
                svd_reduce_rate=svd_reduce_rate,
                device='cpu',  # 最终运行在CPU上
                svd_device=svd_device,  # 但SVD分解在GPU上进行
                No_init=True
            )
        
        print(f"✅ GPTJPipeline初始化完成！")
        print(f"🎯 准备开始推理，SVD压缩率: {self.svd_reduce_rate}")

    def forward_one_batch(self):
        input_ids = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=(1, 1),
            dtype=torch.long,
            device='cuda:0'
        )
        self.forward(input_ids)


    def forward(self, input_ids):
        outputs = input_ids

        # Reset caches for a fresh generation
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.edge.v_cache[i] = None

        # Statistics variables
        layer_calls = 0

        # Context window size
        max_ctx = self.cloud.max_ctx
        x = self.embed(outputs)
        # Process the input sequence step by step for causal language modeling
        # for token_idx in range(outputs.size(1) - 1):  # Exclude the last token for target generation
            # For each token in the sequence, we use the preceding tokens for input
            # cur_input = outputs[:, token_idx].unsqueeze(-1)  # Take current token as input
            

        for layer_idx in range(self.num_layers):
            # Use cache-enabled forward so attention spans all previous tokens
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            self.total_data_transfer+=x.numel() * x.element_size()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,             # 记录输入输出的 shape
                with_flops=True,
            )as pf:
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
            layer_events = pf.key_averages()
            layer_flops = sum(evt.flops for evt in layer_events if evt.flops is not None)

            self.total_flops_cloud+=layer_flops

            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            
            x_cpu = x.to('cuda:0')
            attn_cpu = attn_weights.to('cuda:0')
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,             # 记录输入输出的 shape
                with_flops=True,
            )as pf:
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
            layer_events = pf.key_averages()
            layer_flops = sum(evt.flops for evt in layer_events if evt.flops is not None)
            self.total_flops_edge+=layer_flops

            elements = attn_cpu.numel() * attn_cpu.element_size()  # B
            self.total_data_transfer+=elements
            x = x_cpu.to(self.embed.weight.device)
            
            layer_calls += 1

        # Final normalization and LM head to get logits
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        """
        调用 forward 方法生成文本
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()

        # 开始生成文本
        outputs = input_ids.copy()

        for token_idx in range(max_length):
            # 当前token输入到模型
            cur_input = torch.tensor([[outputs[-1]]]).to(self.embed.weight.device)
            logits = self.forward(cur_input)  # 调用forward方法

            # 使用 top-k + 温度采样代替贪心采样
            next_logits = logits[:, -1, :] / temperature
            topk_vals, topk_idx = torch.topk(next_logits, k=top_k, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            next_id = topk_idx[0, torch.multinomial(probs, num_samples=1).item()].item()
            
            outputs.append(next_id)
            
            # 如果遇到结束符，提前停止
            if next_id == self.tokenizer.eos_token_id:
                print(f"  遇到结束符，提前结束生成")
                break

        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=True)


pipeline = GPTJPipeline(
        model_name=model_name,
        device_cloud=device_cloud,
        device_edge=device_edge,
        svd_reduce_rate=-1,  #-1 即不做初始化
)
dataloader=load_and_tokenize_dataset("./minipile_cache",pipeline.tokenizer,1)

def model_reduce(rates_list):
    edge_model = pipeline.edge  # 直接复用对象
    temp = nn.ModuleList()

    # 加载模型缓存并添加到 svd_layers
    for i, rate in enumerate(rates_list):
        cache_path = f"svd_models/svd_layer_{i}_rate_{rate}.pt"
        print(f"正在加载：{cache_path}")
        mod = torch.load(cache_path, map_location='cuda:0', weights_only=False)
        temp.append(mod)

    edge_model.svd_layers = temp
    edge_model.v_cache = [None] * num_layers
    return edge_model

# def F()
from detection.MINI_PIPE_EVAL import evaluate_minipile_gptj,load_and_tokenize_dataset
def taski(speciesi): 
    F_score_list=0
    cloud_flops=0
    edge_flops=0
    net_element=0

    cloud_time=0
    edge_time=0
    net_time=0

    model=model_reduce(speciesi)
    eval_result = evaluate_minipile_gptj(model, batch_size=1, Dataloader=dataloader)
    model.forward_one_batch()
    cloud_flops=model.total_flops_cloud
    edge_flops=model.total_flops_edge
    net_element=model.total_data_transfer
    
    cloud_time=cloud_flops/model.cloud_flops_per_s
    edge_time=edge_flops/model.edge_flops_per_s
    net_time=net_element/model.bandwidth
    
    ans={
        "cloud_time":cloud_time,
        "edge_time":edge_time,
        "net_time":net_time,
        "cloud_flops":cloud_flops,
        "edge_flops":edge_flops,
        "net_element":net_time,
        "loss":eval_result['avg_loss']
    }
        
    
    torch.cuda.empty_cache()
    
    return ans

def search_GA_warm(self,number_of_layer_to_reduce,step=0.1):
        init_species=[]
        species_map={}
        alpha_fit={}
        rates= [round(i * 0.1, 1) for i in range(0,10)]
        for i in range(50):
            temp=[random.choice(rates) for _ in range(num_layers)]
            indices_to_zero = random.sample(range(num_layers), number_of_layer_to_reduce)

            for idx in indices_to_zero:
                temp[idx] = 0.0
            
            init_species.append(temp)
        lass_F=0
        generate_epoch=30
        # F_score_list=[0 for _ in range(len(init_species))]
        scnt=0
        alpha_cp=[]
        while(generate_epoch):
            alpha=random.randint(0,int(1/ALPHASTEP)-1)*ALPHASTEP
            if(len(alpha_cp)<1//ALPHASTEP):
                alpha=len(alpha_cp)*ALPHASTEP
                alpha_cp.append(0)
            F_score_list=[]
            st=time.time()

            task_group = []
            
            task_list=[]
            torch.cuda.empty_cache()

            for idx,speciesi in enumerate(init_species):
                if((speciesi,alpha) in species_map):
                    continue
                task_list.append((speciesi,alpha))
            
            i=0
            torch.cuda.empty_cache()

        # 将任务分配给进程池中的进程
            print("开始分配进程,总任务量:",len(task_list))
            if((len(task_list)//numworker)==0):
                if(len(task_list)==0):
                    print("搜索结束")
                    if(len(alpha_cp)>=1//CONFIG.ALPHASTEP):
                        break
                    else:
                        continue
                
                else:
                    pool.starmap(self.taski, [ 
                        (task_list,q,gpu_usage,lock) 
                            ])
            else:
                pool.starmap(self.taski, [(task_list[i:min(len(task_list), i + len(task_list) // numworker)],q,gpu_usage,lock) for i in range(0, len(task_list), len(task_list) // numworker)])
            
            
            print("All tasks are completed.")
            ed=time.time()
            print("Generate_time:",ed-st)
            
            F_score_list.clear()
            while not q.empty():
                task_group.append(q.get())
            
            # 创建缓存
            for groupi in task_group:
                schemes=groupi[0]
                F_alpha_scores=groupi[1]
                latencies=groupi[2]
                losses=groupi[3]
                accs=groupi[4]
                net_latencies=groupi[5]
                for i in range(len(schemes)):
                    for j in range(0,int(1/CONFIG.ALPHASTEP)):
                        species_map[(schemes[i],F_alpha_scores[i][j][1])]=(
                            F_alpha_scores[i][j][0],
                            latencies[i],
                            losses[i],
                            accs[i],
                            net_latencies[i]
                        )

            for specisei in init_species:
                F_score_list.append(species_map[(specisei,alpha)][0])
            
            #分布调整
            unique_dict=dict(zip(init_species,F_score_list))
            unique_species=list(unique_dict.keys())
            unique_F=list(unique_dict.values())
            sum_cut_list=[sum(sublist) for sublist in unique_species]
            max_cut=max(sum_cut_list)
            len_u=len(unique_species)
            for i in range(len_u):
                cut_i=sum(unique_species[i])
                if(self.ifexpand(max_cut=max_cut,cut_i=cut_i,alpha=(1-alpha))):
                    unique_species.append(unique_species[i])
                    unique_F.append(unique_F[i])
            
            #淘汰个体
            cb=list(zip(unique_species,unique_F))

            cbsorted=sorted(cb,key=lambda x:x[1],reverse=True)
            cbsorted=cbsorted[:self.init_size]
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
        

            scnt+=1
            generate_epoch-=1
            print("solutions:",scnt,end='\r')


        pool.close()
        pool.join()
        manager.shutdown()

        for spi,f in species_map.items():
            print(f"F:{f[0]},latency:{f[1]},loss:{f[2]},acc:{f[3]},net_latency:{f[4]},alpha:{spi[1]}")
        return species_map
    
    

def searcer_GA_V2(self,init_specise,alpha_step):
    task_number_change=[]
    f_change=[]
    species=[tuple(x) for x in init_specise]
    species_map={(x[0],round(x[1],1)):y for x,y in init_specise.items()}
    # species_map={}
    avege_alpha={}
    for f_a in init_specise:
        she=f_a[0]
        alpha=f_a[1]
        f=init_specise[f_a][0]
        if alpha in avege_alpha:
            avege_alpha[round(alpha,1)].append(f)
        else:
            avege_alpha[round(alpha,1)]=[]
            avege_alpha[round(alpha,1)].append(f)

    for alpha in avege_alpha:
        alpha=round(alpha,1)
        avege_alpha[alpha]=sum(avege_alpha[alpha])/len(avege_alpha[alpha])
    
    lass_F=0
    
    # F_score_list=[0 for _ in range(len(init_species))]
    scnt=0
    numworker=CONFIG.WORKERNUMBER
    pool = multiprocessing.Pool(processes=numworker)
    manager=Manager()
    q=manager.Queue()
    # lock=manager.Namespace()
    lock=manager.Lock()
    # lock.lock=False
    gpu_usage=manager.list()
    for i in range(CONFIG.GPU_AVAILABLE[0],CONFIG.GPU_AVAILABLE[1]):
        gpu_usage.append(0)
    for i in CONFIG.UNAVAILABLE:
        gpu_usage[i]=1000000

    ffg=0
    for alpha in np.arange(0,1,alpha_step):
        alpha=round(alpha,1)
        print(f"alpha:{alpha}")
        generate_epoch=CONFIG.ASTOEPOCH
        init_species=[]
        for x in species:
            if init_specise[x][0]>=avege_alpha[alpha]:
                init_species.append(x[0]) 
        while(generate_epoch):
            F_score_list=[]
            st=time.time()
            task_group = []
            threads=[]
            task_list=[]
            torch.cuda.empty_cache()

            for idx,speciesi in enumerate(init_species):
                if((speciesi,alpha) in species_map):
                    continue
                task_list.append((speciesi,alpha))
            
            i=0
            torch.cuda.empty_cache()
            if(alpha==0.5):
                task_number_change.append(len(task_list))
        # 将任务分配给进程池中的进程
            print("开始分配进程,总任务量:",len(task_list))
            if((len(task_list)//numworker)==0):
                if(len(task_list)==0):
                    print("无新任务")
                    # break
                
                else:
                    pool.starmap(self.taski, [ 
                        (task_list,q,gpu_usage,lock) 
                            ])
            else:
                pool.starmap(self.taski, [ 
                    (task_list[i:min(len(task_list), i + len(task_list) // numworker)],q,gpu_usage,lock) 
                                    for i in range(0, len(task_list), len(task_list) // numworker)])
            
            
            print("All tasks are completed.")
            ed=time.time()
            print("Generate_time:",ed-st)
            
            F_score_list.clear()
            while not q.empty():
                task_group.append(q.get())
            
            # 创建缓存
            for groupi in task_group:
                schemes=groupi[0]
                F_alpha_scores=groupi[1]
                latencies=groupi[2]
                losses=groupi[3]
                accs=groupi[4]
                net_latencies=groupi[5]
                for i in range(len(schemes)):
                    for j in range(0,int(1/CONFIG.ALPHASTEP)):
                        species_map[(schemes[i],round(F_alpha_scores[i][j][1],1))]=(
                            F_alpha_scores[i][j][0],
                            latencies[i],
                            losses[i],
                            accs[i],
                            net_latencies[i]
                        )

            for specisei in init_species:
                F_score_list.append(species_map[(specisei,alpha)][0])
            
            #分布调整
            unique_dict=dict(zip(init_species,F_score_list))
            unique_species=list(unique_dict.keys())
            unique_F=list(unique_dict.values())
            sum_cut_list=[sum(sublist) for sublist in unique_species]
            max_cut=max(sum_cut_list)
            len_u=len(unique_species)
            for i in range(len_u):
                cut_i=sum(unique_species[i])
                if(self.ifexpand(max_cut=max_cut,cut_i=cut_i,alpha=(1-alpha))):
                    unique_species.append(unique_species[i])
                    unique_F.append(unique_F[i])
            
            #淘汰个体
            cb=list(zip(unique_species,unique_F))

            cbsorted=sorted(cb,key=lambda x:x[1],reverse=True)
            cbsorted=cbsorted[:self.init_size]
            init_species=[x[0] for x in cbsorted]
            F_score_list=[x[1] for x in cbsorted]
            if(alpha==0.5):
                f_change.append(max(F_score_list))
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
        

            scnt+=1
            generate_epoch-=1
            print("solutions:",scnt,end='\r')
        print("alpha:",alpha,"搜索结束,正在计算最优划分方案")
        ffg=1
        max_F=-10
        best_sp=None
        for ii in init_species:
            if((ii,alpha) not in species_map):
                continue
            if(species_map[(ii,alpha)][0]>max_F):
                max_F=species_map[(ii,alpha)][0]
                best_sp=ii
        if(best_sp==None):
            print("alpha:",alpha,"未初始化的划分方案")
            continue
        print("alpha:",alpha,"最优划分方案:",best_sp)
        print("alpha:",alpha,"最优F值:",max_F)
        print("alpha:",alpha,"最优latency:",species_map[(best_sp,alpha)][1])
        print("alpha:",alpha,"最优loss:",species_map[(best_sp,alpha)][2])
        print("alpha:",alpha,"最优acc:",species_map[(best_sp,alpha)][3])
                
        with open(CONFIG.SAVE_PATH_SCHEME, "a", encoding="utf-8") as f:
            f.write("alpha: " + str(alpha) + "\n")
            f.write(" 最优划分方案: " + str(best_sp) + "\n")
            f.write(" 最优F值: " + str(max_F) + "\n")
            f.write(" 最优latency: " + str(species_map[(best_sp,alpha)][1]) + "\n")
            f.write(" 最优loss: " + str(species_map[(best_sp,alpha)][2]) + "\n")
            f.write(" 最优acc: " + str(species_map[(best_sp,alpha)][3]) + "\n")


    pool.close()
    pool.join()
    manager.shutdown()

    
    return species_map,task_number_change,f_change
