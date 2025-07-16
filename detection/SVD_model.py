import torch
from torch import nn
from torch import vmap
import time
from thop import profile
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import time

# Load the fused C++ extension
fused_svd_op = None
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fused_svd_op = load(
        name="fused_svd_matmul",
        sources=[os.path.join(current_dir, "fused_svd_matmul.cpp")],
        verbose=True,
    )
except Exception as e:
    print(f"Warning: Could not load fused SVD extension: {e}")
    fused_svd_op = None

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

class SVDED_GPT2_EDGE_Layer(nn.Module):
    def __init__(self, gpt2_edge_layer, reduce_rate, device, svd_device='cpu'):
        super(SVDED_GPT2_EDGE_Layer, self).__init__()
        self.device = device
        self.svd_device = svd_device  # 新增SVD分解设备参数
        self.reduce_rate = reduce_rate
        self.original_layer = gpt2_edge_layer
        
        # 保存配置信息
        self.num_heads = gpt2_edge_layer.num_heads
        self.head_dim = gpt2_edge_layer.head_dim
        self.ln1_eps = gpt2_edge_layer.ln1_eps
        self.ln2_eps = gpt2_edge_layer.ln2_eps
        
        # 保存LayerNorm参数(不进行SVD)
        self.ln1_weight = gpt2_edge_layer.ln1_weight
        self.ln1_bias = gpt2_edge_layer.ln1_bias
        self.ln2_weight = gpt2_edge_layer.ln2_weight
        self.ln2_bias = gpt2_edge_layer.ln2_bias
        
        # 对所有线性层进行SVD分解，保存U和V矩阵用于fused操作
        self.v_svd = self._create_svd_linear(gpt2_edge_layer.v_weight, gpt2_edge_layer.v_bias)
        self.proj_svd = self._create_svd_linear(gpt2_edge_layer.proj_weight, gpt2_edge_layer.proj_bias)
        self.fc_svd = self._create_svd_linear(gpt2_edge_layer.fc_weight, gpt2_edge_layer.fc_bias)
        self.fc_proj_svd = self._create_svd_linear(gpt2_edge_layer.fc_proj_w, gpt2_edge_layer.fc_proj_b)
    
    def _create_svd_linear(self, weight, bias):
        """创建SVD分解的线性层，返回U、V矩阵和bias"""
        # 保存原始数据类型
        original_dtype = weight.dtype
        
        # 将权重移到SVD分解设备上
        weight_svd = weight.to(self.svd_device)
        
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
        U_tensor = U.t().contiguous().to(dtype=original_dtype, device=self.device)
        V_tensor = V.t().contiguous().to(dtype=original_dtype, device=self.device)
        bias_tensor = bias.to(dtype=original_dtype, device=self.device).contiguous() if bias is not None else torch.empty(0).to(dtype=original_dtype, device=self.device)
        
        return {
            'U': U_tensor.t(),
            'V': V_tensor.t(),
            'bias': bias_tensor
        }
    
    def _apply_svd_linear(self, x, svd_data, original_weight, original_bias):
        """应用SVD分解的线性变换，使用fused操作"""
        # if fused_svd_op is not None:
        #     # 使用fused C++ 操作
        #     return fused_svd_op.fused_svd_matmul(x, svd_data['U'].t(), svd_data['V'], svd_data['bias'])
        # else:
        #     # 回退到原始的两步操作
        #     intermediate = torch.matmul(x, svd_data['U'].t())
        #     result = torch.matmul(intermediate, svd_data['V'].t())
        #     if svd_data['bias'].numel() > 0:
        #         result = result + svd_data['bias']
        #     return result


        # intermediate = torch.matmul(x, svd_data['U'])                      # -> (out_dim, r)
        # result       = F.linear(intermediate, svd_data['V'], svd_data['bias'])

        intermediate = torch.matmul(x, svd_data['U'])
        result = torch.matmul(intermediate, svd_data['V'])
        if svd_data['bias'].numel() > 0:
            result = result + svd_data['bias']
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
        
        # ln_1
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影 (使用SVD)
        v_new = self._apply_svd_linear(x1, self.v_svd, self.original_layer.v_weight, self.original_layer.v_bias)
        
        # 更新缓存
        v_all = torch.cat([v_cache, v_new], dim=1) if v_cache is not None else v_new
        
        # reshape & attn
        b, seq_k, _ = v_all.shape
        v_h = v_all.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # c_proj + 残差 (使用SVD)
        attn_out = self._apply_svd_linear(ctx, self.proj_svd, self.original_layer.proj_weight, self.original_layer.proj_bias)
        x2 = attn_out + x
        
        # ln_2 + MLP
        m2 = x2.mean(-1, keepdim=True)
        v2 = x2.var(-1, keepdim=True, unbiased=False)
        x2n = (x2 - m2) / torch.sqrt(v2 + self.ln2_eps) * self.ln2_weight + self.ln2_bias
        
        # ffn (使用SVD)
        hidden = self._apply_svd_linear(x2n, self.fc_svd, self.original_layer.fc_weight, self.original_layer.fc_bias)
        hidden = torch.nn.functional.gelu(hidden)
        ffn_out = self._apply_svd_linear(hidden, self.fc_proj_svd, self.original_layer.fc_proj_w, self.original_layer.fc_proj_b)
        x_out = x2 + ffn_out
        
        return v_all, x_out
    
    def forward_no_cache(self, x, attn_weights):
        """
        无缓存版：仅使用当前 x 计算 V，与 attn_weights 对应。
        """
        # 确保 attn_weights 的数据类型与边缘端模型一致（从云端传来可能是float16）
        if attn_weights.dtype != x.dtype:
            attn_weights = attn_weights.to(dtype=x.dtype)
        
        # ln_1
        m1 = x.mean(-1, keepdim=True)
        v1 = x.var(-1, keepdim=True, unbiased=False)
        x1 = (x - m1) / torch.sqrt(v1 + self.ln1_eps) * self.ln1_weight + self.ln1_bias
        
        # V 投影（仅当前 token）(使用SVD)
        v_new = self._apply_svd_linear(x1, self.v_svd, self.original_layer.v_weight, self.original_layer.v_bias)
        
        # reshape & attn
        b, seq_k, _ = v_new.shape
        v_h = v_new.view(b, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.matmul(attn_weights, v_h)
        b2, h2, seq_q, hd = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(b2, seq_q, h2 * hd)
        
        # c_proj + 残差 (使用SVD)
        attn_out = self._apply_svd_linear(ctx, self.proj_svd, self.original_layer.proj_weight, self.original_layer.proj_bias)
        x2 = attn_out + x
        
        # ln_2 + MLP
        m2 = x2.mean(-1, keepdim=True)
        v2 = x2.var(-1, keepdim=True, unbiased=False)
        x2n = (x2 - m2) / torch.sqrt(v2 + self.ln2_eps) * self.ln2_weight + self.ln2_bias
        
        # ffn (使用SVD)
        hidden = self._apply_svd_linear(x2n, self.fc_svd, self.original_layer.fc_weight, self.original_layer.fc_bias)
        hidden = torch.nn.functional.gelu(hidden)
        ffn_out = self._apply_svd_linear(hidden, self.fc_proj_svd, self.original_layer.fc_proj_w, self.original_layer.fc_proj_b)
        x_out = x2 + ffn_out
        
        return v_new, x_out

class Bias_conv(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.bias=nn.Parameter(b.view(b.shape[0],1,1)) 
    def forward(self,x):
        return x+self.bias
    
class Bias_linear(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.bias=nn.Parameter(b) 
    def forward(self,x):
        return x+self.bias

class SVDED_Linear(nn.Module):
    def __init__(self,origin_layer,reduce_rate,device):
        super(SVDED_Linear,self).__init__()
        self.device=device
        self.origin_layer=origin_layer
        # self.origion_conv=nn.Conv1d(in_channels=self.origin_layer.in_features, out_channels=self.origin_layer.out_features, kernel_size=1)
        # self.origion_conv.weight.data = self.origin_layer.weight.data.unsqueeze(2)
        # self.origion_conv.bias.data = self.origin_layer.bias.data
        self.weight=origin_layer.weight
        if(origin_layer.bias is not None):
            self.b=origin_layer.bias
        else:
            self.b=None
        self.reduce_rate=reduce_rate
        self.U,self.V=self.svd()
        return
    
    def forward_origin(self,x):
        return self.origin_layer(x)
    
    def forward_svd(self,x):
        o1=self.U(x)
        o2=self.V(o1)
        return o2
    
    def forward(self,x):
        if(self.reduce_rate==0):
            return self.forward_origin(x)
        else:
            return self.forward_svd(x)

    def svd(self):
        
        U,S,V=torch.linalg.svd(self.weight.t())

        sort_index=torch.argsort(S)
        U=U[:,sort_index]
        S=S[sort_index]
        V=V[sort_index,:]

        r=int(len(S)*self.reduce_rate)
        if(r<=0):
            r=len(S)
        U=U[:,r:]
        V=V[r:,:]
        S=torch.diag(S[r:])

        for i in range(min(V.shape[0],V.shape[1])):
            V[i]=V[i]*S[i][i]

        newlinear1=torch.nn.Linear(U.shape[0],U.shape[1],bias=False).to(self.device)
        newlinear2=None
        newlinear1.weight=nn.Parameter(U.t())
        
        if(self.b is not None):
            newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
            newlinear2.weight=nn.Parameter(V.t())
            newlinear2.bias=nn.Parameter(self.b)
        else :
            newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
            newlinear2.weight=nn.Parameter(V.t())
            # newlinear2.bias=None
        return newlinear1,newlinear2


class SVDED_Conv(nn.Module):
    def __init__(self,origin_layer,reduce_rate,device):
        super(SVDED_Conv,self).__init__()
        self.device=device
        self.conv_layer=origin_layer
        self.conv_layer_padding=origin_layer.padding
        self.conv_layer_stride=origin_layer.stride
        self.conv_layer_kernel_size=origin_layer.kernel_size
        self.weight=origin_layer.weight.view(origin_layer.out_channels,-1)
        self.compile=0
        self.temp_conv=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False).to(self.device)
        if(origin_layer.bias is not None):
            self.b=origin_layer.bias
        else:
            self.b=None
        self.reduce_rate=reduce_rate
        self.newconv1,self.newlinear2,self.bias=self.svd()
        
        return
    
    
    def forward_origin(self,x):
        if(x.shape[0]!=1):
            return self.conv_layer(x)
        else:
            # sp=time.perf_counter()
            op=self.conv_layer(x)
            # ed=time.perf_counter()
        return op
        

    def linear2(self,x):
        st=time.perf_counter()
        if(len(x.shape)!=3):
            x=x.view(x.shape[-3],x.shape[-2],x.shape[-1])
     
        output1=self.newconv1(x)
      
        output2=output1.view(output1.shape[0],-1)
        ed=time.perf_counter()
        print("CONV1:",ed-st)

        st=time.perf_counter()
        weight=output2
        output2=output2.view(1,1,output2.shape[0],output2.shape[1])
        output2=output2.permute(3,0,1,2)
        weight=output2

        output3=F.conv2d(self.newlinear2,weight)
        ed=time.perf_counter()
        print("CODE:conv2_time:",ed-st)
        st=time.perf_counter()
        output3_=output3.permute(0,3,2,1)
         
        # output3_=torch.matmul(self.newlinear2,output2)  # 打开这里使用linear计算

        output3_=output3_.view(output3_.shape[2],output3_.shape[3])
        output_H=(x.shape[1]+2*self.conv_layer_padding[0]-self.conv_layer_kernel_size[0])//self.conv_layer_stride[0]+1
        output_W=(x.shape[2]+2*self.conv_layer_padding[1]-self.conv_layer_kernel_size[1])//self.conv_layer_stride[1]+1
        output_res=output3_.view(output3_.shape[0],output_H,output_W)
        if(self.bias is not None):
            b=Bias_conv(self.bias)
            output_res=b(output_res)
        ed=time.perf_counter()
        print("CODE:permute_time:",ed-st)
        print("----------------------------------------------")
        return output_res
    
    def linear2_val(self,x):
        # print("linear2_val")
        if(len(x.shape)!=3):
            x=x.view(1,x.shape[-3],x.shape[-2],x.shape[-1])
        output1=self.newconv1(x)
        # ed=time.perf_counter()
        # print("CODE:conv1_time:",ed-st)
        # print("CODE:conv1_flops:",flops)
        # st=time.perf_counter()
        
        
        # output2=output1.view(output1.shape[1],output1.shape[2],-1)
        # weight=output2
        # output2=output2.view(1,1,output2.shape[0],output2.shape[1])
        # output2=output2.permute(3,0,1,2)
        # weight=output2

        # output3=F.conv2d(self.newlinear2,weight)
        # # ed=time.perf_counter()
        # # print("CODE:conv2_time:",ed-st)
        # # flops,_=profile(F.conv2d,inputs=(weight,))
        # # print("CODE:conv2_flops:",flops)
        # # st=time.perf_counter()
        # output3_=output3.permute(0,3,2,1)
        # 

        output2=output1.view(output1.shape[0],-1)
        output3_=torch.matmul(self.newlinear2,output2)  # 打开这里使用linear计算

        output3_=output3_.view(output3_.shape[2],output3_.shape[3])
        output_H=(x.shape[1]+2*self.conv_layer_padding[0]-self.conv_layer_kernel_size[0])//self.conv_layer_stride[0]+1
        output_W=(x.shape[2]+2*self.conv_layer_padding[1]-self.conv_layer_kernel_size[1])//self.conv_layer_stride[1]+1
        output_res=output3_.view(output3_.shape[0],output_H,output_W)
        if(self.bias is not None):
            b=Bias_conv(self.bias)
            output_res=b(output_res)
        # ed=time.perf_counter()
        # print("CODE:permute_time:",ed-st)
        # print("----------------------------------------------")
        # print("CODE:out_time:",ed2-ed)
        return output_res
    
    def forward_svd(self,x):
        # 记着在有批量数据的时候开vamp
        #start_time=time.perf_counter()
        if(x.shape[0]!=1):
            output_res=vmap(self.linear2_val, in_dims=(0))(x)
            return output_res
        else:
            output_res=self.linear2(x)
        return output_res

        
    
    def forward(self,x):
        if(self.reduce_rate==0):
            op=self.forward_origin(x)
            return op
        else:
            op=self.forward_svd(x)
            
            return op

    def svd(self):
        U,S,V=torch.linalg.svd(self.weight.t())
        
        sort_index=torch.argsort(S)
        U=U[:,sort_index]
        S=S[sort_index]
        V=V[sort_index,:]

        r=int(len(S)*self.reduce_rate)
        U=U[:,r:]
        V=V[r:,:]
        S=torch.diag(S[r:])

        for i in range(min(V.shape[0],V.shape[1])):
            V[i]=V[i]*S[i][i]
        U=U.contiguous()
        V=V.contiguous()
        conv_1=nn.Conv2d(
            in_channels=self.conv_layer.in_channels,
            out_channels=U.shape[1],
            kernel_size=self.conv_layer.kernel_size,
            stride=self.conv_layer.stride,
            padding=self.conv_layer.padding,
            dilation=self.conv_layer.dilation,
            groups=self.conv_layer.groups,
            bias=False
        ).to(self.device)
        conv_1.weight=nn.Parameter(U.t().contiguous().view(U.shape[1],self.conv_layer.in_channels,*self.conv_layer.kernel_size).contiguous())
        newlinear2=None
        biass=None
        if(self.b is not None):
            newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=True).to(self.device)   
            weight=V.contiguous().t().contiguous()     
            weight=weight.view(1,1,weight.shape[0],weight.shape[1])
            newlinear2.weight=nn.Parameter(weight)
            newlinear2.bias=nn.Parameter(self.b.contiguous())
            biass=self.b.contiguous()
        else :
            newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
            weight=V.contiguous().t().contiguous()
            weight=weight.view(1,1,weight.shape[0],weight.shape[1])
            newlinear2.weight=nn.Parameter(weight)
        newlinear2.requires_grad_=False
        conv_1.weight.requires_grad=False
        
        return conv_1,newlinear2.weight,biass

class SVDED_CONV1D(nn.Module):
    def __init__(self):
        super(SVDED_CONV1D,self).__init__()
        return