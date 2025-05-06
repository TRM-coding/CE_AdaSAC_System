import torch
from torch import nn
from transformers import GPT2Model, AutoModelForCausalLM

class gpt2_edge(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        # 离线加载预训练模型
        # self.model = GPT2Model.from_pretrained(model_name, local_files_only=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            "/home/tianruiming/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
            local_files_only=True,
        )
        # 加载 GPT2 并提取每层的 V 权重和 bias
        # use ParameterList to register buffers for .to()
        self.v_weights = nn.ParameterList()
        self.v_bias    = nn.ParameterList()

        # ln_1 参数
        self.ln1_weights = nn.ParameterList()
        self.ln1_bias    = nn.ParameterList()
        self.ln1_eps     = []
        # attention 输出投影
        self.proj_weights = nn.ParameterList()
        self.proj_bias    = nn.ParameterList()
        # ln_2 + MLP
        self.ln2_weights  = nn.ParameterList()
        self.ln2_bias     = nn.ParameterList()
        self.ln2_eps      = []
        self.fc_weights   = nn.ParameterList()
        self.fc_bias      = nn.ParameterList()
        self.fc_proj_w    = nn.ParameterList()
        self.fc_proj_b    = nn.ParameterList()

        for block in self.model.transformer.h:
            w = block.attn.c_attn.weight    # shape: [hidden, 3*hidden]
            b = block.attn.c_attn.bias      # shape: [3*hidden]
            hidden = self.model.config.hidden_size
            # take the V part from columns
            self.v_weights.append(nn.Parameter(w[:, 2*hidden:], requires_grad=False))
            self.v_bias.append(nn.Parameter(b[2*hidden:], requires_grad=False))

            # ln1
            self.ln1_weights.append(nn.Parameter(block.ln_1.weight, requires_grad=False))
            self.ln1_bias.append   (nn.Parameter(block.ln_1.bias,   requires_grad=False))
            self.ln1_eps.append(block.ln_1.eps)
            # c_proj
            self.proj_weights.append(nn.Parameter(block.attn.c_proj.weight, requires_grad=False))
            self.proj_bias   .append(nn.Parameter(block.attn.c_proj.bias,   requires_grad=False))
            # ln2
            self.ln2_weights.append(nn.Parameter(block.ln_2.weight, requires_grad=False))
            self.ln2_bias.append   (nn.Parameter(block.ln_2.bias,   requires_grad=False))
            self.ln2_eps.append(block.ln_2.eps)
            # MLP
            self.fc_weights.append(nn.Parameter(block.mlp.c_fc.weight, requires_grad=False))
            self.fc_bias   .append(nn.Parameter(block.mlp.c_fc.bias,   requires_grad=False))
            self.fc_proj_w .append(nn.Parameter(block.mlp.c_proj.weight, requires_grad=False))
            self.fc_proj_b .append(nn.Parameter(block.mlp.c_proj.bias,   requires_grad=False))

        self.num_layers = len(self.v_weights)
        # 最大上下文长度
        self.max_ctx    = self.model.config.n_ctx
        self.v_cache = [None] * self.num_layers

        # 新增：record head 信息
        self.num_heads  = self.model.config.n_head
        hidden_size     = self.model.config.hidden_size
        self.head_dim   = hidden_size // self.num_heads


    def forward_cache(self, x, layer_idx, attn_weights):
        """
        x: Tensor [batch_size, seq_len, hidden]
        layer_idx: int
        attn_weights: Tensor [batch_size, num_heads, seq_q, seq_k]
        """
        # ln_1
        eps1 = self.ln1_eps[layer_idx]
        w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
        m1 = x.mean(-1, keepdim=True); v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1)/torch.sqrt(v1+eps1)*w1 + b1
        # V 投影
        v_new = torch.matmul(x1, self.v_weights[layer_idx]) + self.v_bias[layer_idx]
        # sliding window 同前
        v_all = torch.cat([self.v_cache[layer_idx], v_new], dim=1) if self.v_cache[layer_idx] is not None else v_new
        if v_all.size(1) > self.max_ctx: v_all = v_all[:, -self.max_ctx:, :]
        self.v_cache[layer_idx] = v_all
        # reshape & attn
        b, seq_k, _ = v_all.shape
        v_h = v_all.view(b, seq_k, self.num_heads, self.head_dim).transpose(1,2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1,2).contiguous().view(b2, seq_q, h2*hd)
        # c_proj + 残差
        wpr, bpr = self.proj_weights[layer_idx], self.proj_bias[layer_idx]
        attn_out = torch.matmul(ctx, wpr) + bpr
        x2 = attn_out + x
        # ln_2 + MLP
        eps2 = self.ln2_eps[layer_idx]
        w2, b2 = self.ln2_weights[layer_idx], self.ln2_bias[layer_idx]
        m2 = x2.mean(-1, keepdim=True); v2 = x2.var(-1,keepdim=True,unbiased=False)
        x2n = (x2 - m2)/torch.sqrt(v2+eps2)*w2 + b2
        # ffn
        fw, fb = self.fc_weights[layer_idx], self.fc_bias[layer_idx]
        pw, pb = self.fc_proj_w[layer_idx], self.fc_proj_b[layer_idx]
        hidden = torch.nn.functional.gelu(torch.matmul(x2n, fw) + fb)
        ffn_out= torch.matmul(hidden, pw) + pb
        x_out  = x2 + ffn_out
        return v_all, x_out

    def forward_no_cache(self, x, layer_idx, attn_weights):
        """
        无缓存版：仅使用当前 x 计算 V，与 attn_weights 对应。
        """
        # ln_1
        eps1 = self.ln1_eps[layer_idx]
        w1, b1 = self.ln1_weights[layer_idx], self.ln1_bias[layer_idx]
        m1 = x.mean(-1, keepdim=True); v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1)/torch.sqrt(v1+eps1)*w1 + b1
        # V 投影（仅当前 token）
        v_new = torch.matmul(x1, self.v_weights[layer_idx]) + self.v_bias[layer_idx]
        # reshape & attn
        b, seq_k, _ = v_new.shape
        v_h = v_new.view(b, seq_k, self.num_heads, self.head_dim).transpose(1,2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1,2).contiguous().view(b2, seq_q, h2*hd)
        # c_proj + 残差
        wpr, bpr = self.proj_weights[layer_idx], self.proj_bias[layer_idx]
        attn_out = torch.matmul(ctx, wpr) + bpr
        x2 = attn_out + x
        # ln_2 + MLP
        eps2 = self.ln2_eps[layer_idx]
        w2, b2 = self.ln2_weights[layer_idx], self.ln2_bias[layer_idx]
        m2 = x2.mean(-1, keepdim=True); v2 = x2.var(-1,keepdim=True,unbiased=False)
        x2n = (x2 - m2)/torch.sqrt(v2+eps2)*w2 + b2
        fw, fb = self.fc_weights[layer_idx], self.fc_bias[layer_idx]
        pw, pb = self.fc_proj_w[layer_idx], self.fc_proj_b[layer_idx]
        hidden = torch.nn.functional.gelu(torch.matmul(x2n, fw) + fb)
        ffn_out= torch.matmul(hidden, pw) + pb
        x_out  = x2 + ffn_out
        return v_new, x_out
