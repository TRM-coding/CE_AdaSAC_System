import torch
import copy
from torch import nn

class Bias(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.bias=nn.Parameter(b)
    
    def forward(self,x):
        return x+self.bias

class SVD():
    def __init__(self,model_path,detection_input,detection_label,device):
        self.detection_input=detection_input
        self.detection_label=detection_label
        self.model=torch.load(model_path)
        self.device=device

    def based_on_reduce_rate(self,reduce_rate=0,reduce_bound_of_layer=1,sorted=True):
        model=copy.deepcopy(self.model)
        
        for k , layer in enumerate(model.layers):
            if(k>reduce_bound_of_layer):
                break
            if(isinstance(layer,torch.nn.Linear)):
                w=layer.weight
                b=layer.bias
                U,S,V=torch.linalg.svd(w.t())
                if(sorted):
                    sort_index=torch.argsort(S)
                    U=U[:,sort_index]
                    S=S[sort_index]
                    V=V[sort_index,:]
                r=len(S)*reduce_rate
                U=U[:,r:]
                V=V[r:,:]
                S=torch.diag(S[r:])

                newlinear1=torch.nn.Linear(U.shape[0],U.shape[1],bias=False).to(self.device)
                newlinear2=torch.nn.Linear(S.shape[0],S.shape[1],bias=False).to(self.device)
                newlinear3=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
                newlinear1.weight=nn.Parameter(U.t())
                newlinear2.weight=nn.Parameter(S.t())
                newlinear3.weight=nn.Parameter(V.t())
                newbias=Bias(b).to(self.device)

                svded=nn.Sequential(newlinear1,newlinear2,newlinear3,newbias)
                model.layers[k]=svded

        return model
    
    def loss_evaluation(self,model,inputs,outputs_label):
        model.eval()
        model_output=model(inputs)
        loss_function=nn.CrossEntropyLoss()
        evaled_loss=loss_function(model_output,outputs_label).item()
        return evaled_loss

    def acc_evaluation(self,model,inputs,labels):
        model.eval()
        model_output=model(inputs)
        model_label=model_output.argmax(dim=1)
        acc=(model_label==labels).sum().item()/len(inputs)
        return acc