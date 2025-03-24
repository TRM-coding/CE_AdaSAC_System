import torch
import torch.nn as nn
from torchvision import models
from .mymodel_file.Resnet50 import Resnet50
import itertools
class Resnet50Loader:
    def __init__(self):
        self.model=Resnet50()
        self.origin=Resnet50()
        self.pre_trained=models.resnet50(weights=True)
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
        for name,Bottlenecki in self.pre_layer_map['layer1'].named_children():
            idj=0
            Bottlenecki_list=[]
            for name,layer in Bottlenecki.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    Bottlenecki_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        Bottlenecki_list.append(layerpp)
            Myres=getattr(self.model,self.layer_name[idx])
            for name,layerp in Myres.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(Myres,name,Bottlenecki_list[idj])
                idj+=1
            idx+=1
            setattr(Myres,'multed',True)
        for name,Bottlenecki in self.pre_layer_map['layer2'].named_children():
            idj=0
            Bottlenecki_list=[]
            for name,layer in Bottlenecki.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    Bottlenecki_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        Bottlenecki_list.append(layerpp)
            Myres=getattr(self.model,self.layer_name[idx])
            for name,layerp in Myres.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(Myres,name,Bottlenecki_list[idj])
                idj+=1
            idx+=1
            setattr(Myres,'multed',True)
        for name,Bottlenecki in self.pre_layer_map['layer3'].named_children():
            idj=0
            Bottlenecki_list=[]
            for name,layer in Bottlenecki.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    Bottlenecki_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        Bottlenecki_list.append(layerpp)
            Myres=getattr(self.model,self.layer_name[idx])
            for name,layerp in Myres.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(Myres,name,Bottlenecki_list[idj])
                idj+=1
            idx+=1
            setattr(Myres,'multed',True)
        for name,Bottlenecki in self.pre_layer_map['layer4'].named_children():
            idj=0
            Bottlenecki_list=[]
            for name,layer in Bottlenecki.named_children():
                if not isinstance(layer,nn.Sequential):
                    if isinstance(layer,nn.ReLU):
                        continue
                    Bottlenecki_list.append(layer)
                else:
                    for _,layerpp in layer.named_children():
                        Bottlenecki_list.append(layerpp)
            Myres=getattr(self.model,self.layer_name[idx])
            for name,layerp in Myres.named_children():
                if(isinstance(layerp,nn.ReLU)):
                    continue
                setattr(Myres,name,Bottlenecki_list[idj])
                idj+=1
            idx+=1
            setattr(Myres,'multed',True)
        setattr(self.model,'avgpool',self.pre_layer_map['avgpool'])
        setattr(self.model,'linear',self.pre_layer_map['fc'])
        return self.model
if __name__=='__main__':
    rs=Resnet50Loader()
    rs5=rs.origin
    # print(rs5)
    # for _,layer in rs5.named_children():
    #     if(hasattr(layer,'multed')):
    #         print(layer.multed)
    # ip=torch.randn(1,3,224,224)
    # rs5(ip)
    for name,layer in rs5.named_parameters():
        print(name,layer)
        print()