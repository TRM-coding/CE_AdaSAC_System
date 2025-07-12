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
    def __init__(self, original_edge, svd_reduce_rate, device='cpu', svd_device='cpu'):
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
        for i in range(self.num_layers):
            print(f"  处理第 {i+1}/{self.num_layers} 层: ", end="")
            original_edge_layer = original_edge.layers[i]
            if(i>=4):
                self.svd_layers.append(original_edge_layer)
                continue
            if(i%2):
                # 奇数层跳过压缩
                svd_layer = SVDED_GPTJ_EDGE_Layer(
                    gptj_edge_layer=original_edge_layer,
                    reduce_rate=svd_reduce_rate,
                    device=device,
                    svd_device=svd_device
                )
                print("跳过压缩 (奇数层)")
                self.svd_layers.append(svd_layer)
            else:
                # 偶数层进行SVD压缩
                print(f"正在进行SVD分解 (压缩率: {svd_reduce_rate})...")
                
                svd_start_time = time.time()
                svd_layer = SVDED_GPTJ_EDGE_Layer(
                    gptj_edge_layer=original_edge_layer,
                    reduce_rate=svd_reduce_rate,
                    device=device,
                    svd_device=svd_device
                )
                svd_end_time = time.time()
                print(f"    ✅ 第 {i+1} 层SVD分解完成 (耗时: {svd_end_time - svd_start_time:.2f}秒)")
                self.svd_layers.append(svd_layer)
        
        print(f"🎉 所有层的SVD分解处理完成！")
    
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

class GPTJPipeline:
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b', device_cloud='cuda:0', device_edge='cpu', svd_reduce_rate=0.5, use_compile=True):
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
        self.cloud       = gptJ_cloud(model_name=model_dir).to(device_cloud)
        print(f"🖥️  加载边缘模型到CPU...")
        # 强制 edge 放在 CPU
        original_edge    = gptJ_edge (model_name=model_dir).to('cpu')
        self.embed       = self.cloud.model.transformer.wte
        self.ln_f        = self.cloud.model.transformer.ln_f
        self.lm_head     = self.cloud.model.lm_head
        self.num_layers  = len(self.cloud.q_weights)
        
        print(f"📊 模型共有 {self.num_layers} 层")
        
        # SVD压缩参数
        self.svd_reduce_rate = svd_reduce_rate
        self.use_compile = use_compile
        
        # 创建整个SVD edge模型
        print(f"🔧 创建SVD边缘模型...")
        # 如果有GPU，先在GPU上进行SVD分解，然后移到CPU
        svd_device = device_cloud if torch.cuda.is_available() else 'cpu'
        print(f"🔧 SVD分解将在 {svd_device} 上进行...")
        
        self.edge = SVD_GPTJ_Edge_Model(
            original_edge=original_edge,
            svd_reduce_rate=svd_reduce_rate,
            device='cpu',  # 最终运行在CPU上
            svd_device=svd_device  # 但SVD分解在GPU上进行
        )
        
        print(f"✅ GPTJPipeline初始化完成！")
        print(f"🎯 准备开始推理，SVD压缩率: {self.svd_reduce_rate}")
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        print(f"🔄 开始文本生成...")
        print(f"📝 输入提示: '{prompt}'")
        print(f"⚙️  生成参数: max_length={max_length}, temperature={temperature}, top_k={top_k}")
        
        # 重置性能监控器并开始跟踪
        self.performance_monitor.reset_stats()
        self.performance_monitor.start_memory_tracking()
        self.performance_monitor.record_memory_snapshot("生成开始")
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].tolist()
        outputs   = input_ids.copy()

        # reset caches for a fresh generation
        print(f"🗂️  清空缓存...")
        for i in range(self.num_layers):
            self.cloud.k_cache[i] = None
            self.edge.v_cache[i] = None

        # 统计变量
        bandwidth = 10  # MB/s

        # 上下文窗口大小
        max_ctx = self.cloud.max_ctx

        print(f"🔥 预热阶段：处理 {len(input_ids)} 个提示token...")
        self.performance_monitor.record_memory_snapshot("预热阶段开始")
        
        # 预热缓存：将 prompt 中每个 token 走一次 forward_cache
        for pos, token_id in enumerate(input_ids):
            print(f"  处理提示token {pos+1}/{len(input_ids)}")
            self.performance_monitor.increment_token_count()
            
            # clamp 位置，防止越界
            pos_clamped = pos if pos < max_ctx else max_ctx - 1
            cur_id = torch.tensor([[token_id]]).to(self.embed.weight.device)
            
            # GPT-J 没有位置embedding，直接使用 token embedding
            x = self.embed(cur_id)
            
            # 逐层处理 - 云端计算QK，边端计算V
            for layer_idx in range(self.num_layers):
                # 云端计算：在GPU上计算QK和注意力权重，保持K缓存在GPU
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                cloud_time = time.time() - t0
                self.performance_monitor.record_cloud_time(cloud_time)
                
                # 网络传输：只传输注意力权重到边端
                attn_cpu = attn_weights.to('cpu')
                elements = attn_cpu.numel() * attn_cpu.element_size()  # B
                net_time = elements / bandwidth / 1024 / 1024  # s
                self.performance_monitor.record_network_time(net_time)
                
                # 边端计算：在CPU上计算V和输出，保持V缓存在CPU
                x_cpu = x.to('cpu')
                t1 = time.time()
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                edge_time = time.time() - t1
                print(f"edge_time_{layer_idx}:",edge_time)
                self.performance_monitor.record_edge_time(edge_time)
                
                # 网络传输：将处理后的x传回云端
                x = x_cpu.to(self.embed.weight.device)
                elements = x.numel() * x.element_size()  # B
                net_time = elements / bandwidth / 1024 / 1024
                self.performance_monitor.record_network_time(net_time)
                
                self.performance_monitor.increment_counters()
            
            self.performance_monitor.record_memory_snapshot(f"Token {pos+1} 处理完成")

        print(f"🎯 生成阶段：开始生成新token...")
        self.performance_monitor.record_memory_snapshot("生成阶段开始")
        
        # 真实生成阶段
        for token_idx in range(max_length):
            if token_idx % 5 == 0:  # 每5个token显示一次进度
                print(f"  生成进度: {token_idx}/{max_length}")
                self.performance_monitor.record_memory_snapshot(f"生成Token {token_idx}")
                
            self.performance_monitor.increment_token_count()
            cur_id = torch.tensor([[outputs[-1]]]).to(self.embed.weight.device)
            x = self.embed(cur_id)
            
            # 逐层处理 - 云端计算QK，边端计算V
            for layer_idx in range(self.num_layers):
                # 云端计算：在GPU上计算QK和注意力权重，保持K缓存在GPU
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                t0 = time.time()
                _, _, attn_weights = self.cloud.forward_cache(x, layer_idx)
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                cloud_time = time.time() - t0
                self.performance_monitor.record_cloud_time(cloud_time)

                # 网络传输：只传输注意力权重到边端
                attn_cpu = attn_weights.to('cpu')
                elements = attn_cpu.numel() * attn_cpu.element_size()  # B
                net_time = elements / bandwidth / 1024 / 1024
                self.performance_monitor.record_network_time(net_time)
                
                # 边端计算：在CPU上计算V和输出，保持V缓存在CPU
                x_cpu = x.to('cpu')
                t1 = time.time()
                _, x_cpu = self.edge.forward_cache(x_cpu, layer_idx, attn_cpu)
                edge_time = time.time() - t1
                self.performance_monitor.record_edge_time(edge_time)
                
                # 网络传输：将处理后的x传回云端
                x = x_cpu.to(self.embed.weight.device)
                elements = x.numel() * x.element_size()  # B
                net_time = elements / bandwidth / 1024 / 1024
                self.performance_monitor.record_network_time(net_time)
                
                self.performance_monitor.increment_counters()
            
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
                print(f"  遇到结束符，提前结束生成")
                break

        self.performance_monitor.record_memory_snapshot("生成完成")
        
        # 停止内存跟踪
        tracemalloc_current, tracemalloc_peak = self.performance_monitor.stop_memory_tracking()
        self.performance_monitor._tracemalloc_peak = tracemalloc_peak
        
        # 打印详细性能报告
        self.performance_monitor.print_detailed_report()
        self.performance_monitor.print_memory_timeline()
            
        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=True)

if __name__ == "__main__":
    # 把 intra-op（单个算子内部并行）线程数调到 56
    # torch.set_num_threads(56)
    # 把 inter-op（不同算子之间并行）线程数也调大
    # torch.set_num_interop_threads(56)
    print("num_threads: %d" % torch.get_num_threads())
    # torch.set_num_interop_threads(8)
    model_name = 'AI-ModelScope/gpt-j-6b'
    device_cloud = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_edge = 'cpu'
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"🎮 检测到CUDA设备，将使用GPU进行云端计算")
        print(f"🔧 GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  未检测到CUDA设备，将使用CPU进行云端计算")
    
    # 测试不同的SVD压缩率
    svd_rates = [0.8]
    
    for svd_rate in svd_rates:
        print(f"\n{'='*60}")
        print(f"🧪 测试SVD压缩率: {svd_rate}")
        print(f"{'='*60}")
        
        try:
            print(f"🔧 初始化性能监控...")
            pipeline = GPTJPipeline(
                model_name=model_name, 
                device_cloud=device_cloud, 
                device_edge=device_edge,
                svd_reduce_rate=svd_rate
            )
            
            prompt = "Once upon a time"
            print(f"\n💬 提示词: '{prompt}'")
            
            print(f"⏱️  开始生成文本...")
            overall_start_time = time.time()
            generated_text = pipeline.generate(prompt, max_length=20)
            overall_end_time = time.time()
            
            print(f"\n📝 生成的文本:")
            print(f"   {generated_text}")
            print(f"⏱️  总生成时间: {overall_end_time - overall_start_time:.2f}秒")
            
            # 获取性能摘要
            stats = pipeline.performance_monitor.get_summary_stats()
            print(f"\n📈 性能摘要:")
            print(f"   🔢 处理Token数: {stats['token_count']}")
            print(f"   ☁️  GPU总时间: {stats['cloud_total_time']:.4f}s")
            print(f"   🖥️  CPU总时间: {stats['edge_total_time']:.4f}s")
            print(f"   🌐 网络总时间: {stats['network_total_time']:.4f}s")
            print(f"   💾 内存变化: {stats['memory_diff_mb']:+.2f}MB")
            print(f"   📊 峰值内存: {stats['memory_peak_mb']:.2f}MB")
            
            # 保存性能数据到文件
            with open(f"performance_svd_{svd_rate}.txt", "w") as f:
                f.write(f"SVD压缩率: {svd_rate}\n")
                f.write(f"处理Token数: {stats['token_count']}\n")
                f.write(f"GPU总时间: {stats['cloud_total_time']:.4f}s\n")
                f.write(f"CPU总时间: {stats['edge_total_time']:.4f}s\n")
                f.write(f"网络总时间: {stats['network_total_time']:.4f}s\n")
                f.write(f"内存变化: {stats['memory_diff_mb']:+.2f}MB\n")
                f.write(f"峰值内存: {stats['memory_peak_mb']:.2f}MB\n")
                f.write(f"总生成时间: {overall_end_time - overall_start_time:.2f}秒\n")
                f.write(f"生成文本: {generated_text}\n")
            
            print(f"💾 性能数据已保存到 performance_svd_{svd_rate}.txt")
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}")
        print(f"🏁 SVD压缩率 {svd_rate} 测试完成")
        print(f"{'='*60}")
    
    print(f"\n🎉 所有测试完成！")
