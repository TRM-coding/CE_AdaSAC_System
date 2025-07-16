import torch
import torch.nn as nn
import time
import psutil
import os
import tracemalloc
from transformers import AutoTokenizer
from modelscope.utils.hub import snapshot_download
from detection.Loader.mymodel_file.gptJ_cloud import gptJ_cloud
from detection.Loader.mymodel_file.gptJ_edge import gptJ_edge
from detection.SVD_model import SVDED_GPTJ_EDGE_Layer
from detection.MINI_PIPE_EVAL import evaluate_minipile_gptj,load_and_tokenize_dataset

class PerformanceMonitor:
    """性能监控类，用于记录CPU时间和内存使用情况"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.reset_stats()
        
    def reset_stats(self):
        """重置统计数据"""
        self.cloud_gpu_times = []
        self.edge_cpu_times = []
        self.network_times = []
        self.memory_snapshots = []
        self.token_count = 0
        self.layer_calls = 0
        
        # 详细的计时统计
        self.cloud_total_time = 0.0
        self.edge_total_time = 0.0
        self.network_total_time = 0.0
        
        # 内存统计
        self.initial_memory = self.get_memory_mb()
        self.peak_memory = self.initial_memory
        
    def get_memory_mb(self):
        """获取当前内存使用量(MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_percent(self):
        """获取CPU使用率"""
        return self.process.cpu_percent()
    
    def start_memory_tracking(self):
        """开始内存跟踪"""
        tracemalloc.start()
        
    def stop_memory_tracking(self):
        """停止内存跟踪并返回统计信息"""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return current / 1024 / 1024, peak / 1024 / 1024  # 转换为MB
        return 0, 0
    
    def record_cloud_time(self, time_taken):
        """记录云端GPU时间"""
        self.cloud_gpu_times.append(time_taken)
        self.cloud_total_time += time_taken
        
    def record_edge_time(self, time_taken):
        """记录边缘CPU时间"""
        self.edge_cpu_times.append(time_taken)
        self.edge_total_time += time_taken
        
    def record_network_time(self, time_taken):
        """记录网络传输时间"""
        self.network_times.append(time_taken)
        self.network_total_time += time_taken
        
    def record_memory_snapshot(self, phase=""):
        """记录内存快照"""
        current_memory = self.get_memory_mb()
        self.memory_snapshots.append({
            'phase': phase,
            'memory_mb': current_memory,
            'timestamp': time.time()
        })
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def increment_counters(self):
        """增加计数器"""
        self.layer_calls += 1
        
    def increment_token_count(self):
        """增加token计数"""
        self.token_count += 1
        
    def print_detailed_report(self):
        """打印详细的性能报告"""
        print(f"\n{'='*70}")
        print(f"🔍 详细性能分析报告")
        print(f"{'='*70}")
        
        # 基本统计
        print(f"📊 基本统计信息:")
        print(f"   🔢 处理的Token数量: {self.token_count}")
        print(f"   🔢 总层调用次数: {self.layer_calls}")
        print(f"   🔢 平均每token层调用: {self.layer_calls/max(1, self.token_count):.1f}")
        
        # 时间统计
        print(f"\n⏱️  时间统计 (总计):")
        print(f"   ☁️  GPU云端总时间: {self.cloud_total_time:.4f}s")
        print(f"   🖥️  CPU边缘总时间: {self.edge_total_time:.4f}s")
        print(f"   🌐 网络传输总时间: {self.network_total_time:.4f}s")
        print(f"   🔄 总处理时间: {self.cloud_total_time + self.edge_total_time + self.network_total_time:.4f}s")
        
        # 平均时间统计
        if self.token_count > 0:
            print(f"\n⏱️  平均每Token时间:")
            print(f"   ☁️  GPU云端平均: {self.cloud_total_time/self.token_count:.4f}s")
            print(f"   🖥️  CPU边缘平均: {self.edge_total_time/self.token_count:.4f}s")
            print(f"   🌐 网络传输平均: {self.network_total_time/self.token_count:.4f}s")
            print(f"   🔄 总平均: {(self.cloud_total_time + self.edge_total_time + self.network_total_time)/self.token_count:.4f}s")
        
        # 内存统计
        current_memory = self.get_memory_mb()
        memory_diff = current_memory - self.initial_memory
        print(f"\n💾 内存使用统计:")
        print(f"   📈 初始内存: {self.initial_memory:.2f}MB")
        print(f"   📊 当前内存: {current_memory:.2f}MB")
        print(f"   📈 峰值内存: {self.peak_memory:.2f}MB")
        print(f"   📊 内存变化: {memory_diff:+.2f}MB")
        
        # CPU使用率
        cpu_percent = self.get_cpu_percent()
        print(f"   🔥 当前CPU使用率: {cpu_percent:.1f}%")
        
        # 获取内存跟踪信息
        if hasattr(self, '_tracemalloc_peak'):
            print(f"   🔍 内存跟踪峰值: {self._tracemalloc_peak:.2f}MB")
        
        # 时间分布分析
        if len(self.cloud_gpu_times) > 0:
            print(f"\n📈 GPU时间分布:")
            print(f"   最小: {min(self.cloud_gpu_times):.4f}s")
            print(f"   最大: {max(self.cloud_gpu_times):.4f}s")
            print(f"   平均: {sum(self.cloud_gpu_times)/len(self.cloud_gpu_times):.4f}s")
            
        if len(self.edge_cpu_times) > 0:
            print(f"\n📈 CPU时间分布:")
            print(f"   最小: {min(self.edge_cpu_times):.4f}s")
            print(f"   最大: {max(self.edge_cpu_times):.4f}s")
            print(f"   平均: {sum(self.edge_cpu_times)/len(self.edge_cpu_times):.4f}s")
        
        # 性能比较
        if self.cloud_total_time > 0 and self.edge_total_time > 0:
            ratio = self.edge_total_time / self.cloud_total_time
            print(f"\n🔍 性能比较:")
            print(f"   CPU/GPU时间比: {ratio:.2f}x")
            if ratio > 1:
                print(f"   💡 CPU比GPU慢 {ratio:.1f} 倍")
            else:
                print(f"   💡 CPU比GPU快 {1/ratio:.1f} 倍")
        
        print(f"{'='*70}")
        
    def print_memory_timeline(self):
        """打印内存使用时间线"""
        if len(self.memory_snapshots) > 0:
            print(f"\n📊 内存使用时间线:")
            for i, snapshot in enumerate(self.memory_snapshots):
                print(f"   {i+1}. {snapshot['phase']}: {snapshot['memory_mb']:.2f}MB")
                
    def get_summary_stats(self):
        """返回摘要统计信息"""
        return {
            'cloud_total_time': self.cloud_total_time,
            'edge_total_time': self.edge_total_time,
            'network_total_time': self.network_total_time,
            'token_count': self.token_count,
            'layer_calls': self.layer_calls,
            'memory_usage_mb': self.get_memory_mb(),
            'memory_peak_mb': self.peak_memory,
            'memory_diff_mb': self.get_memory_mb() - self.initial_memory
        }

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
        """
        兼容原始edge模型的forward_cache接口
        Args:
            x: 输入tensor
            layer_idx: 层索引
            attn_weights: 注意力权重
        Returns:
            tuple: (v_cache, output_x) - 与原始edge模型相同的返回格式
        """
        # 使用SVD压缩的层进行前向传播
        # tim1=time.time()
        self.v_cache[layer_idx], output_x = self.svd_layers[layer_idx].forward_cache(
            x, self.v_cache[layer_idx], attn_weights
        )
        # tim2=time.time()
        # print(f"layer_{layer_idx}_forward_time:",tim2-tim1)
        
        # 应用sliding window到缓存
        if self.v_cache[layer_idx] is not None and self.v_cache[layer_idx].size(1) > self.max_ctx:
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, -self.max_ctx:, :]
        # tim3=time.time()
        # print(f"layer_{layer_idx}_memory_time:",tim3-tim2)

        return self.v_cache[layer_idx], output_x

class GPTJPipeline(nn.Module):
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cpu', svd_reduce_rate=0.5, use_compile=True,edge=None):
        super(GPTJPipeline, self).__init__()
        print(f"🚀 初始化GPTJPipeline...")
        print(f"📋 配置信息:")
        print(f"   - 模型: {model_name}")
        print(f"   - 云端设备: {device_cloud}")
        print(f"   - 边缘设备: {device_edge}")
        print(f"   - SVD压缩率: {svd_reduce_rate}")
        
        # 初始化性能监控器
        self.performance_monitor = PerformanceMonitor()
        
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


    def forward(self, input_ids):
        # 1. 生成 padding mask: pad_token_id 位置为 0，其它为 1
        #    假设 self.config.pad_token_id 已经被设置
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()  # [B, T]

        # Reset caches for a fresh generation
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.edge.v_cache[i] = None

        # Statistics variables
        cloud_time = edge_time = net_time = 0.0
        layer_calls = 0
        bandwidth = 10  # MB/s

        # Embedding
        x = self.embed(input_ids)  # [B, T, D]

        # 层级迭代
        for layer_idx in range(self.num_layers):
            # Cloud forward：传入 attention_mask，用于内部做 pad+causal 屏蔽
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            t0 = time.time()
            _, _, attn_weights = self.cloud.forward_cache(x, layer_idx, attention_mask)
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
            cloud_time += time.time() - t0

            # Edge forward（保持不变）
            x_cpu = x.to('cuda:0')
            attn_cpu = attn_weights.to('cuda:0')
            t1 = time.time()
            _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
            edge_time += time.time() - t1

            # 网络开销估算
            elements = attn_cpu.numel() * attn_cpu.element_size()
            net_time += elements / bandwidth / 1024 / 1024
            x = x_cpu.to(self.embed.weight.device)
            elements = x.numel() * x.element_size()
            net_time += elements / bandwidth / 1024 / 1024

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


def load_svd_cache(
    model_dir: str = "svd_models",
    rates: list[float] = None,
    map_location: str = "cpu"
) -> dict[float, nn.ModuleList]:
    """
    从磁盘加载所有已保存的 SVD 分解层，返回 rate->ModuleList 的映射。

    Args:
        model_dir (str): 存放 .pt 文件的目录。
        rates (List[float], optional): 需要加载的压缩率列表，默认 [0.0,0.1,...,0.9]。
        map_location (str): torch.load 的 map_location 参数，默认为 "cpu"。

    Returns:
        Dict[float, ModuleList]: key 是压缩率，value 是加载后的 ModuleList。
    """
    if rates is None:
        rates = [round(i * 0.1, 1) for i in range(10)]
    svd_cache: dict[float, nn.ModuleList] = {}
    for rate in rates:
        fname = f"svd_layers_rate_{rate}.pt"
        fpath = os.path.join(model_dir, fname)
        if not os.path.isfile(fpath):
            print(f"⚠️ 文件不存在，跳过：{fpath}")
            continue
        # 加载
        ml = torch.load(fpath, map_location=map_location)
        if not isinstance(ml, nn.ModuleList):
            raise ValueError(f"文件 {fname} 中的对象不是 ModuleList")
        svd_cache[rate] = ml
        print(f"✔ 已加载：{fname}")
    return svd_cache

import gc
import random
import os
if __name__ == "__main__":
    # 基本配置
    model_name    = 'AI-ModelScope/gpt-j-6b'
    device_cloud  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_edge   = 'cuda:0'
    svd_device    = device_cloud if torch.cuda.is_available() else 'cpu'
    rates         = [round(i * 0.1, 1) for i in range(0,10)]   # [0.0, 0.1, …, 0.9]

    # 1. 下载并加载原始 edge 模型
    original_edge = gptJ_edge(model_name=model_name).to(device_edge)
    num_layers    = original_edge.num_layers

    del original_edge
    
    gc.collect()
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Allocated memory: {torch.cuda.memory_allocated(device_edge)}")
    print(f"Reserved memory: {torch.cuda.memory_reserved(device_edge)}")

    # 2. 预分解：对每层、每个 rate 进行一次 SVD，缓存到内存 & 磁盘
    svd_cache: dict = {}   # rate -> nn.ModuleList
    os.makedirs("svd_models", exist_ok=True)
    # with torch.no_grad():
    #     for i, layer in enumerate(original_edge.layers):
    #         svd_cache.setdefault(i, {})  # 初始化该层的缓存字典
    #         print(f"▶️ 处理第 {i} 层 SVD 分解")
            
    #         for rate in rates:
    #             cache_path = f"svd_models/svd_layer_{i}_rate_{rate}.pt"
    #             print(f"  🔄 压缩率={rate}")
    #             svd_layer=None
    #             # 如果已有缓存，尝试加载
    #             if rate in svd_cache[i]:
    #                 print(f"    ⚡ 缓存命中，跳过计算，已在 svd_cache 中")
    #             elif os.path.exists(cache_path):
    #                 try:
    #                     mod = torch.load(cache_path, map_location='cpu',weights_only=False)
    #                     svd_cache[i][rate] = mod
    #                     print(f"    ⚡ 从磁盘加载缓存：{cache_path}")
    #                 except Exception as e:
    #                     print(f"    ⚠️ 加载缓存失败：{e}")
    #                     gc.collect()
    #             else:
    #                 # 未缓存则执行分解
    #                 print(f"    ▶️ 执行分解，使用设备: {device_edge}")
    #                 svd_layer = SVDED_GPTJ_EDGE_Layer(
    #                     gptj_edge_layer=layer,
    #                     reduce_rate=rate,
    #                     device=device_edge,
    #                     svd_device=svd_device
    #                 )
    #                 # 分解完成后，移动到CPU并清理显存
    #                 svded_layer=svd_layer.to('cpu')
    #                 torch.cuda.empty_cache()
    #                 gc.collect()
    #                 svd_cache[i][rate] = svded_layer
    #                 # 保存到磁盘
    #                 torch.save(svded_layer, cache_path)
                    
    #             if svd_layer is not None:
    #                 # svd_layer.clear()
    #                 del svd_layer
    #             gc.collect()
    #             torch.cuda.synchronize()
    #             torch.cuda.empty_cache()
    #             print(f"    ✔ 已保存：{cache_path}")
    #         # 此层所有 rate 完成后，强制回收本层临时变量
    #         # del layer
    #         gc.collect()
    #         torch.cuda.empty_cache()

    print("正常退出")
    # 3. 随机生成 1000 个裁剪方案
    schemes=[]
    for i in range(2,9):
        rates=[round(j * 0.1, 1) for j in range(0,i+1)]
        for k in range(20):
            temp=[random.choice(rates) for _ in range(num_layers)]
            schemes.append(temp)
    # schemes = [
    #     [random.choice(rates) for _ in range(num_layers)]
    #     for _ in range(1000)
    # ]
    # schemes[0]=[0.0 for _ in range(num_layers)]

    # 4. 对每个方案，快速构建 edge_model 并评估
    #    这里假设已有一个 eval_pipeline(pipeline, scheme_index) 函数，
    #    它会把 pipeline.edge 换成基于该方案的 SVD_GPTJ_Edge_Model，
    #    然后对 MINIPIPE 数据集做一次 eval。
    # from MINI_PIPE_TEST import evaluate_on_minipipe  # 假设已有评估函数

    # # 初始化云端 pipeline，只加载一次大模型
    pipeline = GPTJPipeline(
        model_name=model_name,
        device_cloud=device_cloud,
        device_edge=device_edge,
        svd_reduce_rate=-1,  # 占位，无实际用到
        
    )


    evaluation_results = {}

    dataloader=load_and_tokenize_dataset("./minipile_cache",pipeline.tokenizer,1)

    import pickle
    with torch.no_grad():
        for idx, scheme in enumerate(schemes):
            torch.cuda.empty_cache()
            # 4.1 构建仅替换 svd_layers 的 edge 模型
            edge_model = pipeline.edge  # 直接复用对象
            temp = nn.ModuleList()

            # 加载模型缓存并添加到 svd_layers
            for i, rate in enumerate(scheme):
                cache_path = f"svd_models/svd_layer_{i}_rate_{rate}.pt"
                print(f"正在加载：{cache_path}")
                mod = torch.load(cache_path, map_location='cuda:0', weights_only=False)
                temp.append(mod)

            edge_model.svd_layers = temp
            edge_model.v_cache = [None] * num_layers
            
            # 4.2 将 pipeline.edge 指向新模型并评估
            pipeline.edge = edge_model.to('cuda:0')
            pipeline.cloud = pipeline.cloud.to('cuda:0')
            print(f"\n===== 方案 {idx + 1}/200 =====")
            
            # 调用 evaluate_minipile_gptj 函数并获取结果
            eval_result = evaluate_minipile_gptj(pipeline, batch_size=1, Dataloader=dataloader)
            
            # 存储当前方案的评估结果与 scheme
            evaluation_results[idx] = (scheme, eval_result)
            
            # 每次评估完后将整个 evaluation_results 存储到文件
            with open('evaluation_results.pkl', 'wb') as f:
                pickle.dump(evaluation_results, f)
            
            # 释放内存
            pipeline.edge.to('cpu')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 可选：打印每个方案的评估结果
            print(f"方案 {idx + 1} 评估结果: avg_loss = {eval_result['avg_loss']}, perplexity = {eval_result['perplexity']}")
            
            # 清理缓存
            del temp
            gc.collect()

    print("\n🎉 全部 1000 个方案评估完成！")
