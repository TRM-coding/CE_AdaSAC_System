from detection.Loader.mymodel_file.Alexnet import AlexNet
from torchvision import models
import torch
class AlexnetLoader:
    def __init__(self):
        self.model=AlexNet()
        self.pre_trained=models.alexnet(weights=True)
        print(self.pre_trained)
        self.pre_layer_map={name:layer for name,layer in self.pre_trained.named_children()}
        self.layer_names=[name for name,_ in self.model.named_children()]
    def load(self):
        idx=0
        for layer in self.pre_layer_map['features']:
            setattr(self.model,self.layer_names[idx],layer)
            idx+=1
        setattr(self.model,self.layer_names[idx],self.pre_layer_map['avgpool'])
        idx+=1
        for layer in self.pre_layer_map['classifier']:
            if(self.layer_names[idx]=='flatten'):
                idx+=1
            setattr(self.model,self.layer_names[idx],layer)
            idx+=1
        return self.model
    

if __name__=='__main__':
    ax=AlexnetLoader()
    model=ax.load()
    for name,params in model.named_parameters():
        print(name,params)
        print()