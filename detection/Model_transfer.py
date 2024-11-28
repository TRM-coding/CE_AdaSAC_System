import torch
import detection.SVD_model as SVD_model
import copy
from torch import nn

class Model_transfer():
    def __init__(self,model,device,no_wight=True):
        self.model=model.to(device)
        # if(not no_wight):
        #     self.model.load_state_dict(torch.load(model_path))
        # self.model=torch.load(model_path)
        self.device=device
        return
    
    def recrusively_update_based_on_number(self,model,reduce_rate,reduce_number,already_reduced=0):
        for name,child in model.named_children():
            if(already_reduced>=reduce_number):
                return
            if(len(list(child.children()))==0):
                if(isinstance(child,torch.nn.Conv2d)):
                    # print("Conv2d")
                    setattr(model,name,SVD_model.SVDED_Conv(child,reduce_rate,self.device))
                    already_reduced+=1
                elif(isinstance(child,torch.nn.Linear)):
                    # print("Linear")
                    setattr(model,name,SVD_model.SVDED_Linear(child,reduce_rate,self.device))
                    already_reduced+=1
            else:
                self.recrusively_update_based_on_number(child,reduce_rate,reduce_number,already_reduced)   
        return model
    
    def get_svd_model_based_on_rate(self,reduce_rate,reduce_number):
        svded_model=copy.deepcopy(self.model)
        self.recrusively_update_based_on_number(svded_model,reduce_rate,reduce_number)
        return svded_model
    
    def recrusively_update_based_on_list(self,model,reduce_list,already_reduced=0):
        for name,child in model.named_children():
            if(already_reduced>=len(reduce_list)):
                return
            if(len(list(child.children()))==0):
                if(isinstance(child,torch.nn.Conv2d)):
                    # print("Conv2d")
                    setattr(model,name,SVD_model.SVDED_Conv(child,reduce_list[already_reduced],self.device))
                    already_reduced+=1
                elif(isinstance(child,torch.nn.Linear)):
                    # print("Linear")
                    setattr(model,name,SVD_model.SVDED_Linear(child,reduce_list[already_reduced],self.device))
                    already_reduced+=1
            else:
                self.recrusively_update_based_on_list(child,reduce_list,already_reduced)   
        return model
    
    def layer_svd(self,layer,reduce_rate):
        svded_layer=copy.deepcopy(layer)
        if(isinstance(layer,torch.nn.Conv2d)):
            return SVD_model.SVDED_Conv(svded_layer,reduce_rate,self.device)
        elif(isinstance(layer,torch.nn.Linear)):
            return SVD_model.SVDED_Linear(svded_layer,reduce_rate,self.device)
        else: 
            return svded_layer

    def get_svd_model_based_on_list(self,reduce_list:list):
        svded_model=copy.deepcopy(self.model)
        self.recrusively_update_based_on_list(svded_model,reduce_list)
        return svded_model
    
    

    