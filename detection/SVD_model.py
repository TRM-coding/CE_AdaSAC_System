import torch
from torch import nn

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
        self.weight=origin_layer.weight
        if(origin_layer.bias is not None):
            self.b=origin_layer.bias
        else:
            self.b=None
        self.reduce_rate=reduce_rate
        self.U,self.V,self.bias=self.svd()
        return
    
    def forward(self,x):
        o1=self.U(x)
        o2=self.V(o1)
        if(self.bias is not None):
            o3=self.bias(o2)
            return o3
        return o2

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
        self.newlinear1,self.newlinear2,self.bias=self.svd()
        return
    
    def forward(self,x):
        x_unfold=nn.Unfold(
            kernel_size=self.conv_layer.kernel_size,
            stride=self.conv_layer.stride,
            padding=self.conv_layer.padding
        )(x)
        x_permute=x_unfold.permute(0,2,1)
        output1=self.newlinear1(x_permute)
        output2=self.newlinear2(output1)
        output3=output2
        if(self.bias is not None):
            output3=self.bias(output2)
            

        output_permute=output3.permute(0,2,1)

        output_H=int((x.shape[2]+2*self.conv_layer.padding[0]-self.conv_layer.kernel_size[0])/self.conv_layer.stride[0]+1)
        output_W=int((x.shape[3]+2*self.conv_layer.padding[1]-self.conv_layer.kernel_size[1])/self.conv_layer.stride[1]+1)
        output_res=output_permute.view(output_permute.shape[0],output_permute.shape[1],output_H,output_W)
        return output_res

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

        newlinear1=torch.nn.Linear(U.shape[0],U.shape[1],bias=False).to(self.device)
        newlinear2=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
        
        newlinear1.weight=nn.Parameter(U.t())
        newlinear2.weight=nn.Parameter(V.t())
        if(self.b is not None):
            newbias=Bias(self.b).to(self.device)
        else :
            newbias=None
        
        return newlinear1,newlinear2,newbias