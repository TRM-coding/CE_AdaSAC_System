import torch
from torch import nn
import copy
class Conv(nn.Module):
    def __init__(self):
        super(Conv,self).__init__()
        self.layer_list=nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5
            ),
            nn.Conv2d(
                in_channels=6,
                out_channels=32,
                kernel_size=3
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1
            ),
            nn.Linear(30976,10)
        ])
    
    def forward(self,x):
        output=copy.deepcopy(x)
        for layer in self.layer_list:
            if(isinstance(layer,nn.Linear)):
                output=output.view(output.size(0),-1)
            output=layer.forward(output)
        return output