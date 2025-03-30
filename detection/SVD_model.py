import torch
from torch import nn
from torch import vmap
class Bias(nn.Module):
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
        self.weight=origin_layer.weight
        if(origin_layer.bias is not None):
            self.b=origin_layer.bias
        else:
            self.b=None
        self.reduce_rate=reduce_rate
        self.U,self.V,self.bias=self.svd()
        return
    
    def forward_origin(self,x):
        return self.origin_layer(x)
    
    def forward_svd(self,x):
        o1=self.U(x)
        o2=self.V(o1)
        if(self.bias is not None):
            o3=self.bias(o2)
            return o3
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
        newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
        
        newlinear1.weight=nn.Parameter(U.t())
        newlinear2.weight=nn.Parameter(V.t())
        if(self.b is not None):
            newbias=Bias(self.b).to(self.device)
        else :
            newbias=None
        
        return newlinear1,newlinear2,newbias


class SVDED_Conv(nn.Module):
    def __init__(self,origin_layer,reduce_rate,device):
        super(SVDED_Conv,self).__init__()
        self.device=device
        self.conv_layer=origin_layer
        self.weight=origin_layer.weight.view(origin_layer.out_channels,-1)
        if(origin_layer.bias is not None):
            self.b=origin_layer.bias
        else:
            self.b=None
        self.reduce_rate=reduce_rate
        self.newconv1,self.newlinear2,self.bias=self.svd()
        return
    
    def linear2(self,w,x):
        x=x.view(x.shape[0],-1)
        return w@x
    
    def forward_origin(self,x):
        return self.conv_layer(x)
    
    def forward_svd(self,x):
        output1=self.newconv1(x)
        output2=output1.view(output1.shape[0],output1.shape[1],-1)
        output3=torch.matmul(self.newlinear2,output2)

        if(self.bias is not None):
            output3=self.bias(output3)
        output_permute=output3

        output_H=(x.shape[2]+2*self.conv_layer.padding[0]-self.conv_layer.kernel_size[0])//self.conv_layer.stride[0]+1
        output_W=(x.shape[3]+2*self.conv_layer.padding[1]-self.conv_layer.kernel_size[1])//self.conv_layer.stride[1]+1
        output_res=output_permute.view(output_permute.shape[0],output_permute.shape[1],output_H,output_W)
        return output_res
    
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
        U=U[:,r:]
        V=V[r:,:]
        S=torch.diag(S[r:])

        for i in range(min(V.shape[0],V.shape[1])):
            V[i]=V[i]*S[i][i]
        
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
        conv_1.weight=nn.Parameter(U.contiguous().t().view(U.shape[1],self.conv_layer.in_channels,*self.conv_layer.kernel_size))

        # newlinear1=torch.nn.Linear(U.shape[0],U.shape[1],bias=False).to(self.device)
        newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
        
        # newlinear1.weight=nn.Parameter(U.t())
        newlinear2.weight=nn.Parameter(V.t())
        if(self.b is not None):
            newbias=Bias(self.b).to(self.device)
        else :
            newbias=None
        
        return conv_1,newlinear2.weight,newbias
