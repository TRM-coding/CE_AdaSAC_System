import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        
        self.layer1 =nn.Linear(28*28,1024),
        self.layer2 =nn.LeakyReLU(negative_slope=0.1),
        self.layer3 =nn.Linear(1024,128),
        self.layer4 =nn.LeakyReLU(negative_slope=0.1),
        self.layer5 =nn.Linear(128,64),
        self.layer6 =nn.LeakyReLU(negative_slope=0.1),
        self.layer7 =nn.Linear(64,32),
        self.layer8 =nn.LeakyReLU(negative_slope=0.1),
        self.layer9 =nn.Linear(32,10)
        
        self.init_param()
    
    def init_param(self):
        for name,param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self,x):
        x_flat=x.view(-1,28*28)
        op1=self.layer1(x_flat)
        op2=self.layer2(op1)
        op3=self.layer3(op2)
        op4=self.layer4(op3)
        op5=self.layer5(op4)
        op6=self.layer6(op5)
        op7=self.layer7(op6)
        op8=self.layer8(op7)
        op9=self.layer9(op8)
        return op9
    

class Bias(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.bias=nn.Parameter(b)
    
    def forward(self,x):
        return x+self.bias