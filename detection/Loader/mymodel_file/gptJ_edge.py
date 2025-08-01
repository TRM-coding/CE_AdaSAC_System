import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from modelscope.utils.hub import snapshot_download
import os

class gptJ_edge_layer(nn.Module):
    def __init__(self, block, config):
        super().__init__()
        hidden_size = config.n_embd
        self.flops=None
        # Get model dtype from the weight
        model_dtype = block.attn.v_proj.weight.dtype
        
        # V权重和bias
        self.v_weight = nn.Parameter(block.attn.v_proj.weight.clone(), requires_grad=False)
        v_bias = block.attn.v_proj.bias if block.attn.v_proj.bias is not None else torch.zeros(block.attn.v_proj.weight.size(0), dtype=model_dtype)
        self.v_bias = nn.Parameter(v_bias.clone(), requires_grad=False)
        
        # ln_1 参数
        self.ln1_weight = nn.Parameter(block.ln_1.weight.clone(), requires_grad=False)
        self.ln1_bias = nn.Parameter(block.ln_1.bias.clone(), requires_grad=False)
        self.ln1_eps = block.ln_1.eps
        
        # attention 输出投影
        self.out_proj_weight = nn.Parameter(block.attn.out_proj.weight.clone(), requires_grad=False)
        out_proj_bias = block.attn.out_proj.bias if block.attn.out_proj.bias is not None else torch.zeros(block.attn.out_proj.weight.size(0), dtype=model_dtype)
        self.out_proj_bias = nn.Parameter(out_proj_bias.clone(), requires_grad=False)
        
        # MLP 部分 (GPT-J uses parallel attention and MLP)
        self.fc_in_weight = nn.Parameter(block.mlp.fc_in.weight.clone(), requires_grad=False)
        fc_in_bias = block.mlp.fc_in.bias if block.mlp.fc_in.bias is not None else torch.zeros(block.mlp.fc_in.weight.size(0), dtype=model_dtype)
        self.fc_in_bias = nn.Parameter(fc_in_bias.clone(), requires_grad=False)
        self.fc_out_weight = nn.Parameter(block.mlp.fc_out.weight.clone(), requires_grad=False)
        fc_out_bias = block.mlp.fc_out.bias if block.mlp.fc_out.bias is not None else torch.zeros(block.mlp.fc_out.weight.size(0), dtype=model_dtype)
        self.fc_out_bias = nn.Parameter(fc_out_bias.clone(), requires_grad=False)
        
        # 保存配置信息
        self.num_heads = config.n_head
        self.head_dim = hidden_size // config.n_head
        
    # def forward_cache(self, x, v_cache, attn_weights):
    #     # Ensure input has correct dtype
    #     x = x.to(self.v_weight.dtype)
    #     if attn_weights is not None:
    #         attn_weights = attn_weights.to(self.v_weight.dtype)
        
    #     # ln_1 (GPT-J applies LayerNorm to input)
    #     m1 = x.mean(-1, keepdim=True)
    #     v1 = x.var(-1, keepdim=True, unbiased=False)
    #     x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
    #     # V 投影
    #     v_new = torch.matmul(x1, self.v_weight.T) + self.v_bias
        
    #     # 更新缓存
    #     if v_cache is not None:
    #         v_cache = v_cache.to(v_new.dtype)
    #         v_all = torch.cat([v_cache, v_new], dim=1)
    #     else:
    #         v_all = v_new
        
    #     # reshape & attention
    #     b, seq_k, _ = v_all.shape
    #     v_h = v_all.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
    #     ctx = torch.matmul(attn_weights, v_h)
    #     b2, h2, seq_q, hd = ctx.shape
    #     ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
    #     # attention 输出投影
    #     attn_out = torch.matmul(ctx, self.out_proj_weight.T) + self.out_proj_bias
        
    #     # MLP (parallel to attention in GPT-J)
    #     mlp_hidden = torch.nn.functional.gelu(torch.matmul(x1, self.fc_in_weight.T) + self.fc_in_bias)
    #     mlp_out = torch.matmul(mlp_hidden, self.fc_out_weight.T) + self.fc_out_bias
        
    #     # 残差连接 (GPT-J adds both attention and MLP outputs to input)
    #     x_out = x + attn_out + mlp_out
        
    #     return v_all, x_out
    
    def forward_no_cache(self, x, attn_weights):
        # Ensure input has correct dtype
        x = x.to(self.v_weight.dtype)
        if attn_weights is not None:
            attn_weights = attn_weights.to(self.v_weight.dtype)
        
        # ln_1
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影（仅当前 token）
        v_new = torch.matmul(x1, self.v_weight.T) + self.v_bias
        
        # reshape & attention
        b, seq_k, _ = v_new.shape
        v_h = v_new.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # attention 输出投影
        attn_out = torch.matmul(ctx, self.out_proj_weight.T) + self.out_proj_bias
        
        # MLP (parallel to attention in GPT-J)
        mlp_hidden = torch.nn.functional.gelu(torch.matmul(x1, self.fc_in_weight.T) + self.fc_in_bias)
        mlp_out = torch.matmul(mlp_hidden, self.fc_out_weight.T) + self.fc_out_bias
        
        # 残差连接
        x_out = x + attn_out + mlp_out
        
        return v_new, x_out

class gptJ_edge(nn.Module):
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b',svd=False,dtype=torch.float16):
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
        # 使用本地路径加载预训练模型
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to loading config first
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                weights_only=False
            )
        for block in self.model.transformer.h:
            layer = gptJ_edge_layer(block, self.model.config)
            self.layers.append(layer)
        
        # 创建层列表
        
        
        
        self.num_layers = 28
        # GPT-J 使用 n_positions 而不是 n_ctx
        # self.max_ctx = getattr(self.model.config, 'n_positions', 2048)
        # self.v_cache = [None] * self.num_layers
        
        # # 保存配置信息
        # self.num_heads = self.model.config.n_head
        # hidden_size = self.model.config.n_embd
        # self.head_dim = hidden_size // self.num_heads
    
    # def forward_cache(self, x, layer_idx, attn_weights):
    #     # 使用指定层进行前向        
    #     v_all, x_out = self.layers[layer_idx].forward_cache(x, self.v_cache[layer_idx], attn_weights)
        
    #     # 更新缓存，应用sliding window
    #     if v_all.size(1) > self.max_ctx:
    #         v_all = v_all[:, -self.max_ctx:, :]
    #     self.v_cache[layer_idx] = v_all
        
    #     return v_all, x_out
    
    def forward_no_cache(self, x, layer_idx, attn_weights):
        return self.layers[layer_idx].forward_no_cache(x, attn_weights)
    
    # 在 gptJ_edge_layer 类中添加 clear 方法

    def clear(self):
        """清理层以节省内存"""
        # 删除所有张量属性
        attrs_to_clear = [
            'v_weight', 'v_bias', 'ln1_weight', 'ln1_bias',
            'out_proj_weight', 'out_proj_bias', 'fc_in_weight', 
            'fc_in_bias', 'fc_out_weight', 'fc_out_bias'
        ]
        
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
