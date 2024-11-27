import torch
import torch.nn as nn
from torchvision import models
from .mymodel_file.Resnet50 import Resnet50
import itertools
class Resnet50Loader:
    def __init__(self):
        self.model=Resnet50()
        self.pre_trained=models.resnet50(pretrained=True)
        # print(self.pre_trained)
        self.layer_name=[f'bt{i}' for i in range(1,17)]
        self.pre_layer_map={name:layer for name,layer in self.pre_trained.named_children()}

    def rand(self):
        return self.model

    def load(self):
        idx=0
        setattr(self.model,'conv1',self.pre_layer_map['conv1'])
        setattr(self.model,'bn1',self.pre_layer_map['bn1'])
        setattr(self.model,'relu1',self.pre_layer_map['relu'])
        setattr(self.model,'maxpool1',self.pre_layer_map['maxpool'])
        for name,layeri in self.pre_layer_map['layer1'].named_children():
            idj=0
            layeri_list=[]
            for name,layer in layeri.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    layeri_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        layeri_list.append(layerpp)
            my_model_i=getattr(self.model,self.layer_name[idx])
            for name,layerp in my_model_i.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(my_model_i,name,layeri_list[idj])
                idj+=1
            idx+=1
        for name,layeri in self.pre_layer_map['layer2'].named_children():
            idj=0
            layeri_list=[]
            for name,layer in layeri.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    layeri_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        layeri_list.append(layerpp)
            my_model_i=getattr(self.model,self.layer_name[idx])
            for name,layerp in my_model_i.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(my_model_i,name,layeri_list[idj])
                idj+=1
            idx+=1
        for name,layeri in self.pre_layer_map['layer3'].named_children():
            idj=0
            layeri_list=[]
            for name,layer in layeri.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    layeri_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        layeri_list.append(layerpp)
            my_model_i=getattr(self.model,self.layer_name[idx])
            for name,layerp in my_model_i.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(my_model_i,name,layeri_list[idj])
                idj+=1
            idx+=1
        for name,layeri in self.pre_layer_map['layer4'].named_children():
            idj=0
            layeri_list=[]
            for name,layer in layeri.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    layeri_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        layeri_list.append(layerpp)
            my_model_i=getattr(self.model,self.layer_name[idx])
            for name,layerp in my_model_i.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(my_model_i,name,layeri_list[idj])
                idj+=1
            idx+=1
        setattr(self.model,'avgpool',self.pre_layer_map['avgpool'])
        setattr(self.model,'linear',self.pre_layer_map['fc'])
        return self.model
if __name__=='__main__':
    rs=Resnet50Loader()
    rs5=rs.load()
    ip=torch.randn(1,3,224,224)
    rs5(ip)
    for name,layer in rs5.named_children():
        print(name,layer)
        print()