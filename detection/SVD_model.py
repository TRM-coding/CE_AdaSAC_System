import torch
from torch import nn
from torch import vmap
import time
from thop import profile
import torch.nn.functional as F

class SVDED_GPT2_EDGE_Layer()
    

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