import torch 
import torch.nn as nn 
from torchvision import models
from mymodel_file.MobileNetv3 import MobileNetv3

class MobileNetv3Loader:
    def __init__(self):
        self.model = MobileNetv3()    
        self.pre_trained = models.mobilenet_v3_large(pretrained=True)
        # print(self.pre_trained)
        

mobileNetv3 = MobileNetv3Loader()
    
        
        