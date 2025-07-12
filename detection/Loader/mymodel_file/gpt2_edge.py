import torch
from torch import nn
from transformers import GPT2Model, AutoModelForCausalLM

class gpt2_edge_layer(nn.Module):
    def __init__(self, block, config):
        super().__init__()
        # 从原始block中提取权重和参数
        hidden_size = config.hidden_size
        
        # V权重和bias
        w = block.attn.c_attn.weight    # shape: [hidden, 3*hidden]
        b = block.attn.c_attn.bias      # shape: [3*hidden]
        self.v_weight = nn.Parameter(w[:, 2*hidden_size:], requires_grad=False)
        self.v_bias = nn.Parameter(b[2*hidden_size:], requires_grad=False)
        
        # ln_1 参数
        self.ln1_weight = nn.Parameter(block.ln_1.weight, requires_grad=False)
        self.ln1_bias = nn.Parameter(block.ln_1.bias, requires_grad=False)
        self.ln1_eps = block.ln_1.eps
        
        # attention 输出投影
        self.proj_weight = nn.Parameter(block.attn.c_proj.weight, requires_grad=False)
        self.proj_bias = nn.Parameter(block.attn.c_proj.bias, requires_grad=False)
        
        # ln_2 + MLP
        self.ln2_weight = nn.Parameter(block.ln_2.weight, requires_grad=False)
        self.ln2_bias = nn.Parameter(block.ln_2.bias, requires_grad=False)
        self.ln2_eps = block.ln_2.eps
        self.fc_weight = nn.Parameter(block.mlp.c_fc.weight, requires_grad=False)
        self.fc_bias = nn.Parameter(block.mlp.c_fc.bias, requires_grad=False)
        self.fc_proj_w = nn.Parameter(block.mlp.c_proj.weight, requires_grad=False)
        self.fc_proj_b = nn.Parameter(block.mlp.c_proj.bias, requires_grad=False)
        
        # 保存配置信息
        self.num_heads = config.n_head
        self.head_dim = hidden_size // config.n_head
        
    def forward_cache(self, x, v_cache, attn_weights):
        """
        x: Tensor [batch_size, seq_len, hidden]
        v_cache: 当前层的V缓存
        attn_weights: Tensor [batch_size, num_heads, seq_q, seq_k]
        返回: (更新的v_cache, 层输出)
        """
        # ln_1
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影
        v_new = torch.matmul(x1, self.v_weight) + self.v_bias
        
        # 更新缓存
        v_all = torch.cat([v_cache, v_new], dim=1) if v_cache is not None else v_new
        
        # reshape & attn
        b, seq_k, _ = v_all.shape
        v_h = v_all.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # c_proj + 残差
        attn_out = torch.matmul(ctx, self.proj_weight) + self.proj_bias
        x2 = attn_out + x
        
        # ln_2 + MLP
        m2 = x2.mean(-1, keepdim=True)
        v2 = x2.var(-1, keepdim=True, unbiased=False)
        x2n = (x2 - m2) / torch.sqrt(v2 + self.ln2_eps) * self.ln2_weight + self.ln2_bias
        
        # ffn
        hidden = torch.nn.functional.gelu(torch.matmul(x2n, self.fc_weight) + self.fc_bias)
        ffn_out = torch.matmul(hidden, self.fc_proj_w) + self.fc_proj_b
        x_out = x2 + ffn_out
        
        return v_all, x_out
    
    def forward_no_cache(self, x, attn_weights):
        """
        无缓存版：仅使用当前 x 计算 V，与 attn_weights 对应。
        """
        # ln_1
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影（仅当前 token）
        v_new = torch.matmul(x1, self.v_weight) + self.v_bias
        
        # reshape & attn
        b, seq_k, _ = v_new.shape
        v_h = v_new.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # c_proj + 残差
        attn_out = torch.matmul(ctx, self.proj_weight) + self.proj_bias
        x2 = attn_out + x
        
        # ln_2 + MLP
        m2 = x2.mean(-1, keepdim=True)
        v2 = x2.var(-1, keepdim=True, unbiased=False)
        x2n = (x2 - m2) / torch.sqrt(v2 + self.ln2_eps) * self.ln2_weight + self.ln2_bias
        
        # ffn
        hidden = torch.nn.functional.gelu(torch.matmul(x2n, self.fc_weight) + self.fc_bias)
        ffn_out = torch.matmul(hidden, self.fc_proj_w) + self.fc_proj_b
        x_out = x2 + ffn_out
        
        return v_new, x_out

class gpt2_edge(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        # 离线加载预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            "/hy-tmp/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
            local_files_only=True,
        )
        
        # 创建层列表
        self.layers = nn.ModuleList()
        for block in self.model.transformer.h:
            layer = gpt2_edge_layer(block, self.model.config)
            self.layers.append(layer)
        
        self.num_layers = len(self.layers)
        # 最大上下文长度
        self.max_ctx = self.model.config.n_ctx
        self.v_cache = [None] * self.num_layers
        
        # 保存配置信息
        self.num_heads = self.model.config.n_head
        hidden_size = self.model.config.hidden_size
        self.head_dim = hidden_size // self.num_heads
    
    def forward_cache(self, x, layer_idx, attn_weights):
        """
        x: Tensor [batch_size, seq_len, hidden]
        layer_idx: int
        attn_weights: Tensor [batch_size, num_heads, seq_q, seq_k]
        """
        # 使用指定层进行前向传播
        v_all, x_out = self.layers[layer_idx].forward_cache(x, self.v_cache[layer_idx], attn_weights)
        
        # 更新缓存，应用sliding window
        if v_all.size(1) > self.max_ctx:
            v_all = v_all[:, -self.max_ctx:, :]
        self.v_cache[layer_idx] = v_all
        
        return v_all, x_out
    
    def forward_no_cache(self, x, layer_idx, attn_weights):
        """
        无缓存版：仅使用当前 x 计算 V，与 attn_weights 对应。
        """
        return self.layers[layer_idx].forward_no_cache(x, attn_weights)