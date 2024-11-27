import torch
from torch import nn
import copy
class Conv(nn.Module):
    def __init__(self):
        super(Conv,self).__init__()
        self.layer1 =nn.Conv2d(in_channels=1,out_channels=64,kernel_size=5)
        self.layer2 =nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3)
        self.layer3 =nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1)
        self.layer4 =nn.Flatten()
        self.layer5 =nn.Linear(7744,10)
    
    def forward(self,x):
        output1=self.layer1(x)
        output2=self.layer2(output1)
        output3=self.layer3(output2)
        output4=self.layer4(output3)
        output5=self.layer5(output4)
        return output5