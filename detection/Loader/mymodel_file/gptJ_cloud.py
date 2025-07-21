import torch 
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from modelscope.utils.hub import snapshot_download
import math
import os

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class gptJ_cloud(nn.Module):
    def __init__(self, model_name='AI-ModelScope/gpt-j-6b'):
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
        
        # 使用本地路径加载预训练模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                weights_only=False
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
        
        # 获取配置信息 - GPT-J 使用不同的属性名
        self.num_heads = self.model.config.n_head
        self.head_dim = self.model.config.n_embd // self.num_heads
        self.rotary_dim = getattr(self.model.config, 'rotary_dim', self.head_dim)
        
        # 提取 Q/K 权重
        self.q_weights = nn.ParameterList()
        self.k_weights = nn.ParameterList()
        self.q_bias = nn.ParameterList()
        self.k_bias = nn.ParameterList()
        
        # LayerNorm 参数
        self.ln1_weights = nn.ParameterList()
        self.ln1_bias = nn.ParameterList()
        self.ln1_eps = []
        # model_dtype=torch.float16
        for block in self.model.transformer.h:
            # Get model dtype
            model_dtype = block.attn.q_proj.weight.dtype
            # block=block.half()
            
            # GPT-J 使用 q_proj, k_proj, v_proj 分离的投影
            self.q_weights.append(nn.Parameter(block.attn.q_proj.weight.clone(), requires_grad=False))
            self.k_weights.append(nn.Parameter(block.attn.k_proj.weight.clone(), requires_grad=False))
            
            # 处理bias，有些模型可能没有bias
            q_bias = block.attn.q_proj.bias if block.attn.q_proj.bias is not None else torch.zeros(block.attn.q_proj.weight.size(0), dtype=model_dtype)
            k_bias = block.attn.k_proj.bias if block.attn.k_proj.bias is not None else torch.zeros(block.attn.k_proj.weight.size(0), dtype=model_dtype)
            self.q_bias.append(nn.Parameter(q_bias.clone(), requires_grad=False))
            self.k_bias.append(nn.Parameter(k_bias.clone(), requires_grad=False))
            
            self.ln1_weights.append(nn.Parameter(block.ln_1.weight.clone(), requires_grad=False))
            self.ln1_bias.append(nn.Parameter(block.ln_1.bias.clone(), requires_grad=False))
            self.ln1_eps.append(block.ln_1.eps)
        
        self.num_layers = len(self.q_weights)
        # GPT-J 使用 n_positions 而不是 n_ctx
        self.max_ctx = getattr(self.model.config, 'n_positions', 2048)
        self.k_cache = [None] * self.num_layers
        
        # 获取模型设备
        self.device = next(self.model.parameters()).device
        
        # 创建旋转位置编码并移动到正确设备
        self.rotary_emb = self._create_rotary_emb()
    
    def _create_rotary_emb(self):
        """Create rotary positional embedding."""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2,device=self.device).half() / self.rotary_dim))
        return inv_freq
    
    def _get_rotary_emb(self, seq_len, device):
        """Get rotary embedding for given sequence length."""
        # 确保所有张量在同一设备上
        t = torch.arange(seq_len, device=device, dtype=self.rotary_emb.dtype)
        rotary_emb = self.rotary_emb.to(device)
        freqs = torch.einsum('i,j->ij', t, rotary_emb)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    # def forward_cache(self, x, layer_idx):
    #     """
    #     x: Tensor [batch_size, seq_len, hidden]
    #     layer_idx: int
    #     """
    #     # LayerNorm
    #     eps = self.ln1_eps[layer_idx]
    #     w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
    #     mean = x.mean(-1, keepdim=True)
    #     var = x.var(-1, keepdim=True, unbiased=False)
    #     x_norm = (x - mean) / torch.sqrt(var + eps) * w1 + b1
        
    #     # Q/K 投影 - 确保bias在正确的设备上
    #     q = torch.matmul(x_norm, self.q_weights[layer_idx].T) + self.q_bias[layer_idx]
    #     k_new = torch.matmul(x_norm, self.k_weights[layer_idx].T) + self.k_bias[layer_idx]
        
    #     # 更新K缓存
    #     k_cache = self.k_cache[layer_idx]
    #     k_all = torch.cat([k_cache, k_new], dim=1) if k_cache is not None else k_new
        
    #     # Sliding window
    #     if k_all.size(1) > self.max_ctx:
    #         k_all = k_all[:, -self.max_ctx:, :]
    #     self.k_cache[layer_idx] = k_all
        
    #     # Reshape for multi-head attention
    #     b, seq_q, _ = q.shape
    #     b, seq_k, _ = k_all.shape
    #     q = q.view(b, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
    #     k_all = k_all.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        
    #     # Apply rotary embeddings
    #     cos_q, sin_q = self._get_rotary_emb(seq_q, q.device)
    #     cos_k, sin_k = self._get_rotary_emb(seq_k, k_all.device)
        
    #     # Only apply rotary to rotary_dim portion
    #     q_rot = q[..., :self.rotary_dim]
    #     q_pass = q[..., self.rotary_dim:]
    #     k_rot = k_all[..., :self.rotary_dim]
    #     k_pass = k_all[..., self.rotary_dim:]
        
    #     q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_q[-seq_q:], sin_q[-seq_q:])
    #     q = torch.cat([q_rot, q_pass], dim=-1)
    #     k_all = torch.cat([k_rot, k_pass], dim=-1)
        
    #     # Compute attention weights
    #     attn_scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #     attn_weights = torch.softmax(attn_scores, dim=-1)
        
    #     return q, k_all, attn_weights
    
    # def forward_no_cache(self, x, layer_idx):
    #     """No cache version for comparison."""
    #     # LayerNorm
    #     eps = self.ln1_eps[layer_idx]
    #     w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
    #     mean = x.mean(-1, keepdim=True)
    #     var = x.var(-1, keepdim=True, unbiased=False)
    #     x_norm = (x - mean) / torch.sqrt(var + eps) * w1 + b1
        
    #     # Q/K 投影 - 确保bias在正确的设备上
    #     q = torch.matmul(x_norm, self.q_weights[layer_idx].T) + self.q_bias[layer_idx]
    #     k = torch.matmul(x_norm, self.k_weights[layer_idx].T) + self.k_bias[layer_idx]
        
    #     # Reshape for multi-head attention
    #     b, seq, _ = q.shape
    #     q = q.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
    #     k = k.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
    #     # Apply rotary embeddings
    #     cos, sin = self._get_rotary_emb(seq, q.device)
        
    #     q_rot = q[..., :self.rotary_dim]
    #     q_pass = q[..., self.rotary_dim:]
    #     k_rot = k[..., :self.rotary_dim]
    #     k_pass = k[..., self.rotary_dim:]
        
    #     q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    #     q = torch.cat([q_rot, q_pass], dim=-1)
    #     k = torch.cat([k_rot, k_pass], dim=-1)
        
    #     # Compute attention weights
    #     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #     attn_weights = torch.softmax(scores, dim=-1)
        
    #     return q, k, attn_weights

    def forward_no_cache(self, x, layer_idx, position_ids=None, attention_mask=None):
        """No cache version with proper attention mask and rotary embedding."""
        # LayerNorm
        eps = self.ln1_eps[layer_idx]
        w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps) * w1 + b1
        
        # Q/K 投影
        q = torch.matmul(x_norm, self.q_weights[layer_idx].T) + self.q_bias[layer_idx]
        k = torch.matmul(x_norm, self.k_weights[layer_idx].T) + self.k_bias[layer_idx]
        
        # Reshape for multi-head attention
        b, seq, _ = q.shape
        q = q.view(b, seq, self.num_heads, self.head_dim)
        k = k.view(b, seq, self.num_heads, self.head_dim)
        
        # 处理位置编码（参考GPT-J源码）
        if position_ids is None:
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0).expand(b, -1)
        
        # **关键：实现正确的rotary embedding**
        embed_positions = self.model.transformer.h[layer_idx].attn.embed_positions.to(x.device)
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
        
        # 重复embed_positions以匹配batch_size
        embed_positions_expanded = embed_positions.unsqueeze(0).repeat(b, 1, 1)  # [b, max_pos, pos_dim]
        
        # 使用position_ids索引对应的位置编码
        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions_expanded, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        
        # 应用rotary embedding（参考GPT-J源码）
        if self.rotary_dim is not None:
            k_rot = k[:, :, :, :self.rotary_dim]
            k_pass = k[:, :, :, self.rotary_dim:]
            
            q_rot = q[:, :, :, :self.rotary_dim]  
            q_pass = q[:, :, :, self.rotary_dim:]
            
            # 应用rotary位置编码
            def apply_rotary_pos_emb_local(tensor, sin, cos):
                # tensor: [b, seq, heads, rotary_dim]
                # sin, cos: [b, seq, rotary_dim//2]
                sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)  # [b, seq, 1, rotary_dim]
                cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
                
                def rotate_every_two(x):
                    x1 = x[:, :, :, ::2]
                    x2 = x[:, :, :, 1::2]
                    x = torch.stack((-x2, x1), dim=-1)
                    return x.flatten(-2)
                
                return (tensor * cos) + (rotate_every_two(tensor) * sin)
            
            k_rot = apply_rotary_pos_emb_local(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb_local(q_rot, sin, cos)
            
            k = torch.cat([k_rot, k_pass], dim=-1)
            q = torch.cat([q_rot, q_pass], dim=-1)
        else:
            # 如果没有rotary_dim限制，应用到整个向量
            def apply_rotary_pos_emb_local(tensor, sin, cos):
                sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
                cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
                
                def rotate_every_two(x):
                    x1 = x[:, :, :, ::2]
                    x2 = x[:, :, :, 1::2]
                    x = torch.stack((-x2, x1), dim=-1)
                    return x.flatten(-2)
                
                return (tensor * cos) + (rotate_every_two(tensor) * sin)
            
            k = apply_rotary_pos_emb_local(k, sin, cos)
            q = apply_rotary_pos_emb_local(q, sin, cos)
        
        # 转置到正确的维度 [b, num_heads, seq, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        
        # 计算attention分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # **创建完整的attention mask（参考GPT-J源码）**
        dtype = scores.dtype
        min_dtype = torch.finfo(dtype).min
        
        # 1. 因果掩码
        if seq > 1:
            causal_mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, min_dtype)
        
        # 2. Padding掩码
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] 其中1表示有效token，0表示padding
            # 扩展为4D: [batch_size, num_heads, seq_len, seq_len]
            
            # 创建padding mask：如果key位置是padding（0），则mask掉
            # attention_mask: [b, seq] -> [b, 1, 1, seq]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 广播到 [b, num_heads, seq, seq]
            expanded_mask = expanded_mask.expand(b, self.num_heads, seq, seq)
            
            # 如果key位置是padding，则设置为min_dtype
            padding_mask = expanded_mask == 0
            scores = scores.masked_fill(padding_mask, min_dtype)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        return q, k, attn_weights