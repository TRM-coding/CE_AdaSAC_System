import torch
from torchvision import models

densnet = models.AlexNet(pretrained=False)
print(densnet)
print(torch.hub.get_dir())
