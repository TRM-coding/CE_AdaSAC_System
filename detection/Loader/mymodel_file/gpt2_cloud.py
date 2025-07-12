import torch 
from torch import nn
from transformers import GPT2Model, AutoModelForCausalLM
import math

class gpt2_cloud(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        # 离线加载预训练模型
        # self.model = GPT2Model.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "/hy-tmp/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
            local_files_only=True,
        )
        # 增加 head 信息
        self.num_heads = self.model.config.n_head
        self.head_dim = self.model.config.hidden_size // self.num_heads

        # use ParameterList to register buffers for .to()
        self.q_weights = nn.ParameterList()
        self.k_weights = nn.ParameterList()
        self.q_bias    = nn.ParameterList()
        self.k_bias    = nn.ParameterList()

        # 第一层 LayerNorm 参数
        self.ln1_weights = nn.ParameterList()
        self.ln1_bias    = nn.ParameterList()
        self.ln1_eps     = []

        for block in self.model.transformer.h:
            w = block.attn.c_attn.weight    # shape: [hidden, 3*hidden]
            b = block.attn.c_attn.bias      # shape: [3*hidden]
            hidden = self.model.config.hidden_size
            self.q_weights.append(nn.Parameter(w[:, :hidden], requires_grad=False))
            self.k_weights.append(nn.Parameter(w[:, hidden:2*hidden], requires_grad=False))
            self.q_bias.append(nn.Parameter(b[:hidden], requires_grad=False))
            self.k_bias.append(nn.Parameter(b[hidden:2*hidden], requires_grad=False))
            self.ln1_weights.append(nn.Parameter(block.ln_1.weight, requires_grad=False))
            self.ln1_bias.append   (nn.Parameter(block.ln_1.bias,   requires_grad=False))
            self.ln1_eps.append(block.ln_1.eps)
        self.num_layers = len(self.q_weights)
        # 最大上下文长度
        self.max_ctx    = self.model.config.n_ctx
        # 内部维护的 K 缓存
        self.k_cache = [None] * self.num_layers

    def forward_cache(self, x, layer_idx):
        """
        x: Tensor [batch_size, seq_len, hidden]
        layer_idx: int
        """
        # 先做 ln_1
        eps = self.ln1_eps[layer_idx]
        w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps) * w1 + b1
        # 用归一化后 x_norm 计算 q/k
        # 取出历史缓存
        k_cache = self.k_cache[layer_idx]
        # 使用转置使得 weight 形状匹配: [hidden, hidden]
        q     = torch.matmul(x_norm, self.q_weights[layer_idx]) + self.q_bias[layer_idx]
        k_new = torch.matmul(x_norm, self.k_weights[layer_idx]) + self.k_bias[layer_idx]
        # 拼接并更新缓存
        k_all = torch.cat([k_cache, k_new], dim=1) if k_cache is not None else k_new
        # sliding window: 保留最近 max_ctx 个 token 的 K
        if k_all.size(1) > self.max_ctx:
            k_all = k_all[:, -self.max_ctx:, :]
        self.k_cache[layer_idx] = k_all
        # 按 head 切分并计算注意力权重
        b, seq, _ = q.shape
        q = q.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k_all = k_all.view(b, k_all.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores  = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return q, k_all, attn_weights

    def forward_no_cache(self, x, layer_idx):
        """
        x: Tensor [batch_size, seq_len, hidden]
        layer_idx: int
        不使用缓存，每次重新计算 Q 和 K
        """
        # 无缓存版也做 ln_1
        eps = self.ln1_eps[layer_idx]
        w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps) * w1 + b1
        # 重新计算 q/k
        # 线性投影
        q = torch.matmul(x_norm, self.q_weights[layer_idx]) + self.q_bias[layer_idx]
        k = torch.matmul(x_norm, self.k_weights[layer_idx]) + self.k_bias[layer_idx]
        # 按 head 切分
        b, seq, _ = q.shape
        q = q.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        return q, k, attn_weights