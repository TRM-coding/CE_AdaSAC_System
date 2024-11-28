import torch 
import torch.nn as nn 
from torchvision import models
from mymodel_file.MBNv3 import MobileNetv3
from mymodel_file.MBNv3 import InvertedResidual
from mymodel_file.MBNv3 import Conv2dNormActivation
# class MobileNetv3Loader:
#     def __init__(self):
#         self.model = MobileNetv3()    
        self.pre_trained = models.mobilenet_v3_large(pretrained=True)
#         print(self.pre_trained)
#         self.pre_layer_map = {name:layer for name,layer in self.pre_trained.named_children()}
#         # print(self.pre_layer_map)
#         self.layer_names=[name for name,_ in self.model.named_children()]
#         # print(self.layer_names)
    
#     def load(self):
#         idx=0
#         # print(len(self.pre_layer_map['features']))
#         # print(len(self.layer_names))
#         for layer in self.pre_layer_map['features']:
#             setattr(self.model, self.layer_names[idx], layer)
#             # print(layer)
#             # print()
#             idx+=1
            
#         setattr(self.model, self.layer_names[idx], self.pre_layer_map['avgpool'])
#         idx += 1
        
#         for layer in self.pre_layer_map['classifier']:
#             setattr(self.model, self.layer_names[idx], layer)
#             # print(layer)
#             # print()
#             idx+=1
#         return self.model

        

# mobileNetv3 = MobileNetv3Loader()
# mbn3 = mobileNetv3.load()
# for name, params in mbn3.named_children():
#     print(name)
    
# # output = mbn3(torch.randn(1,3,224,224))
# # print(output)

res1 = InvertedResidual([
            Conv2dNormActivation(16,16,3,1,1,16,False,"relu"),
            Conv2dNormActivation(16,16,1,1,0,1,False),
        ])