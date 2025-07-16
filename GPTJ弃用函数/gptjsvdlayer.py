class SVDED_GPTJ_EDGE_Layer(nn.Module):

    def clear(self):
        # 删除所有实例属性
        for name in list(self.__dict__.keys()):
            if hasattr(self,name):
                delattr(self, name)

    def __init__(self, gptj_edge_layer, reduce_rate, device, svd_device='cpu'):
        super(SVDED_GPTJ_EDGE_Layer, self).__init__()
        self.device = device
        self.svd_device = svd_device  # 新增SVD分解设备参数
        self.reduce_rate = reduce_rate
        self.original_layer = gptj_edge_layer
        
        # 保存配置信息
        self.num_heads = gptj_edge_layer.num_heads
        self.head_dim = gptj_edge_layer.head_dim
        self.ln1_eps = gptj_edge_layer.ln1_eps
        
        # 保存LayerNorm参数(不进行SVD) - GPT-J只有一个LayerNorm
        self.ln1_weight = gptj_edge_layer.ln1_weight
        self.ln1_bias = gptj_edge_layer.ln1_bias
        
        # 对所有线性层进行SVD分解，保存U和V矩阵用于fused操作
        self.v_svd = self._create_svd_linear(gptj_edge_layer.v_weight, gptj_edge_layer.v_bias)
        self.out_proj_svd = self._create_svd_linear(gptj_edge_layer.out_proj_weight, gptj_edge_layer.out_proj_bias)
        self.fc_in_svd = self._create_svd_linear(gptj_edge_layer.fc_in_weight, gptj_edge_layer.fc_in_bias)
        self.fc_out_svd = self._create_svd_linear(gptj_edge_layer.fc_out_weight, gptj_edge_layer.fc_out_bias)
    
    def _create_svd_linear(self, weight, bias):
        """创建SVD分解的线性层，返回U、V矩阵和bias"""
        # 保存原始数据类型
        original_dtype = weight.dtype
        
        # 将权重移到SVD分解设备上
        weight_svd = weight.to(self.svd_device).t()
        
        # 如果是半精度，转换为float32进行SVD分解
        if weight_svd.dtype == torch.float16:
            weight_svd = weight_svd.float()
        
        # 在指定设备上进行SVD分解
        U, S, V = torch.linalg.svd(weight_svd)
        
        # 按奇异值排序
        sort_index = torch.argsort(S)
        U = U[:, sort_index]
        S = S[sort_index]
        V = V[sort_index, :]
        
        # 计算保留的维度
        r = int(len(S) * self.reduce_rate)
        r = max(0, min(r, len(S)))
        
        U = U[:, r:]
        V = V[r:, :]
        S = torch.diag(S[r:])
        
        # 将奇异值融入V
        for i in range(min(V.shape[0], V.shape[1])):
            V[i] = V[i] * S[i][i]
        
        # 转换回原始数据类型并移到最终设备
        U_tensor = U.contiguous().to(dtype=original_dtype, device=self.device)
        V_tensor = V.contiguous().to(dtype=original_dtype, device=self.device)
        bias_tensor = bias.to(dtype=original_dtype, device=self.device).contiguous() if bias is not None else torch.empty(0).to(dtype=original_dtype, device=self.device)
        torch.cuda.empty_cache()
        return {
            'U': U_tensor,
            'V': V_tensor,
            'bias': bias_tensor
        }
    
    def _apply_svd_linear(self, x, svd_data):
        """应用SVD分解的线性变换，使用fused操作"""
        # if fused_svd_op is not None:
        #     # 使用C++融合操作进行加速
        #     return fused_svd_op.fused_svd_matmul(x, svd_data['U'], svd_data['V'], svd_data['bias'])
        # else:
        #     # 优化的PyTorch实现
        #     # 避免创建中间tensor，使用torch.nn.functional代替手动matmul
        #     intermediate = F.linear(x, svd_data['U'].t())
        #     result = F.linear(intermediate, svd_data['V'].t(), 
        #                      svd_data['bias'] if svd_data['bias'].numel() > 0 else None)
        #     return result
        # tm1=time.time()
        intermediate = torch.matmul(x, svd_data['U'])
        # intermediate = F.linear(x,svd_data['U'].t())
        # print("U_time:",time.time()-tm1)
        # tm2=time.time()
        # result = torch.matmul(intermediate, svd_data['V'])
        result=F.linear(intermediate,svd_data['V'].t(),bias=svd_data['bias'])
        # print("V_time:",time.time()-tm2)
        # tm3=time.time()
        if svd_data['bias'].numel() > 0:
            result = result + svd_data['bias']
        # print("bias_time:",time.time()-tm3)
        torch.cuda.empty_cache()
        return result
    
    def forward_cache(self, x, v_cache, attn_weights):
        """
        x: Tensor [batch_size, seq_len, hidden]
        v_cache: 当前层的V缓存
        attn_weights: Tensor [batch_size, num_heads, seq_q, seq_k]
        返回: (更新的v_cache, 层输出)
        """
        # 确保 attn_weights 的数据类型与边缘端模型一致（从云端传来可能是float16）
        if attn_weights.dtype != x.dtype:
            attn_weights = attn_weights.to(dtype=x.dtype)
        
        # ln_1 (GPT-J只有一个LayerNorm)
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影 (使用SVD)
        v_new = self._apply_svd_linear(x1, self.v_svd)
        
        # 更新缓存
        v_all = torch.cat([v_cache, v_new], dim=1) if v_cache is not None else v_new
        
        # reshape & attn
        b, seq_k, _ = v_all.shape
        v_h = v_all.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        # ctx = F.linear(attn_weights,v_h.t())


        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # attention 输出投影 (使用SVD)
        attn_out = self._apply_svd_linear(ctx, self.out_proj_svd,)
        
        # MLP (parallel to attention in GPT-J, 使用SVD)
        mlp_hidden = self._apply_svd_linear(x1, self.fc_in_svd)
        mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
        mlp_out = self._apply_svd_linear(mlp_hidden, self.fc_out_svd)
        
        # 残差连接 (GPT-J adds both attention and MLP outputs to input)
        x_out = x + attn_out + mlp_out
        
        torch.cuda.empty_cache()
        return v_all, x_out
    
    def forward_no_cache(self, x, attn_weights):
        """
        无缓存版：仅使用当前 x 计算 V，与 attn_weights 对应。
        """
        # 确保 attn_weights 的数据类型与边缘端模型一致（从云端传来可能是float16）
        if attn_weights.dtype != x.dtype:
            attn_weights = attn_weights.to(dtype=x.dtype)
        
        # ln_1 (GPT-J只有一个LayerNorm)
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影（仅当前 token）(使用SVD)
        v_new = self._apply_svd_linear(x1, self.v_svd)
        
        # reshape & attn
        b, seq_k, _ = v_new.shape
        v_h = v_new.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # attention 输出投影 (使用SVD)
        attn_out = self._apply_svd_linear(ctx, self.out_proj_svd)
        
        # MLP (parallel to attention in GPT-J, 使用SVD)
        mlp_hidden = self._apply_svd_linear(x1, self.fc_in_svd)
        mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
        mlp_out = self._apply_svd_linear(mlp_hidden, self.fc_out_svd)
        
        # 残差连接 (GPT-J adds both attention and MLP outputs to input)
        x_out = x + attn_out + mlp_out
        
        return  x_out