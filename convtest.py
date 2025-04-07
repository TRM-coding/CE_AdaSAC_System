import torch
from torch import nn

from thop import profile

model_a=nn.Linear(5,7)
input=torch.randn(2,5)
flops, _ = profile(model_a, inputs=(input,))
op=model_a(input)
print(flops)
print(op.shape)

model_b=nn.Conv2d(1,7,(2,1),stride=1,padding=1)
input=torch.randn(1,1,2,5)
flops, _ = profile(model_b, inputs=(input,))
op=model_b(input)
print(flops)
print(op.shape)