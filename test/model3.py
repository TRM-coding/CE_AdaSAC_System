import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.layers=nn.ModuleList([
            nn.Linear(28*28,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)
        ])
        self.init_param()
    
    def init_param(self):
        for name,param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self,x):
        x_flat=x.view(-1,28*28)
        for layer in self.layers:
            x_flat=layer(x_flat)

        return x_flat

class Bias(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.bias=nn.Parameter(b)
    
    def forward(self,x):
        return x+self.bias