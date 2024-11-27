import torch
import torch.nn as nn
from torchvision import models
from mymodel_file.Resnet50 import Resnet50
class Resnet50Loader:
    def __init__(self):
        self.model=Resnet50()
        self.pre_trained=models.resnet50(pretrained=True)
        # print(self.pre_trained)
        self.layer_name=[f'bt{i}' for i in range(1,17)]
        self.pre_layer_map={name:layer for name,layer in self.pre_trained.named_children()}

    
    def load(self):
        idx=0
        setattr(self.model,'conv1',self.pre_layer_map['conv1'])
        setattr(self.model,'bn1',self.pre_layer_map['bn1'])
        setattr(self.model,'relu1',self.pre_layer_map['relu'])
        setattr(self.model,'maxpool1',self.pre_layer_map['maxpool'])
        for name,layeri in self.pre_layer_map['layer1'].named_children():
            setattr(self.model,self.layer_name[idx],layeri)
            idx+=1
        for name,layeri in self.pre_layer_map['layer2'].named_children():
            setattr(self.model,self.layer_name[idx],layeri)
            idx+=1
        for name,layeri in self.pre_layer_map['layer3'].named_children():
            setattr(self.model,self.layer_name[idx],layeri)
            idx+=1
        for name,layeri in self.pre_layer_map['layer4'].named_children():
            setattr(self.model,self.layer_name[idx],layeri)
            idx+=1
        setattr(self.model,'avgpool',self.pre_layer_map['avgpool'])
        setattr(self.model,'linear',self.pre_layer_map['fc'])
        return self.model
if __name__=='__main__':
    rs=Resnet50Loader()
    rs5=rs.load()
    for name,params in rs5.named_parameters():
        print(name,params)
        print()