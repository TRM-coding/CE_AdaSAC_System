import torch
from torchvision import models

densnet = models.AlexNet(weights=False)
print(densnet)
print(torch.hub.get_dir())
