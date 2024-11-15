import torch
import copy
from torch import nn

class Bias(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.bias=nn.Parameter(b)
    
    def forward(self,x):
        return x+self.bias
    
class NewModel(nn.Module):
    def __init__(self):
        super(NewModel,self).__init__()
        self.layer_list=nn.ModuleList([])
    def create_module_from_name(module_name, *args, **kwargs):
  
        if hasattr(nn, module_name):
            module_class = getattr(nn, module_name)
            return module_class(*args, **kwargs)
        
        else:
            raise ValueError(f"No such module: {module_name}")
    
    def forward(self,x):
        output=copy.deepcopy(x)
        output=output.view(-1,28*28)
        for layer in self.layer_list:
            output=layer.forward(output)
        return output

class model_transfer():
    def __init__(self):
        self.transfer_res=[]
        return    
    def jit_transfer(self,jit_model):
        x=0
        for i in jit_model.named_children():
            x=x+1
        if(x==0):
            self.transfer_res.append(jit_model)
        for name,child in jit_model.named_children():
            self.jit_transfer(child)


class SVD():
    def __init__(self,model_path,device):
        # self.model=torch.jit.load(model_path)
        # self.MT=model_transfer()
        # self.MT.jit_transfer(self.model)
        # self.model_list=self.MT.transfer_res
        self.model=torch.load(model_path)
        self.device=device

    def based_on_reduce_rate(self,reduce_rate=0,reduce_bound_of_layer=1,sorted=True):
        SVD_model=copy.deepcopy(self.model)
        cnt=0
        for k,layer in enumerate(SVD_model.layers):
            if(cnt>reduce_bound_of_layer):
                # SVD_model.layer_list.append(layer)
                continue
            if(isinstance(layer,nn.Linear)):
                cnt=cnt+1
                w=layer.weight
                b=layer.bias
                U,S,V=torch.linalg.svd(w.t())
                if(sorted):
                    sort_index=torch.argsort(S)
                    U=U[:,sort_index]
                    S=S[sort_index]
                    V=V[sort_index,:]
                else:
                    U=U[:,:len(S)]
                    V=V[:len(S),]
                r=int(len(S)*reduce_rate)
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
                # SVD_model.layer_list.append(newlinear1)
                # SVD_model.layer_list.append(newlinear2)
                # SVD_model.layer_list.append(newlinear3)
                svded=nn.Sequential(newlinear1,newlinear2,newlinear3,newbias)
                # jis_svded=torch.jit.script(svded)
                SVD_model.layers[k]=svded
                # setattr(layer,"Linear",jis_svded)

        
        return SVD_model
    
    def loss_evaluation(self,model,inputs,outputs_label):
        model.eval()
        model_output=model(inputs)
        loss_function=nn.CrossEntropyLoss()
        evaled_loss=loss_function(model_output,outputs_label.softmax(dim=1)).item()
        return evaled_loss

    def acc_evaluation(self,model,inputs,labels):
        model.eval()
        model_output=model(inputs)
        model_label=model_output.argmax(dim=1)
        acc=(model_label==labels).sum().item()/len(inputs)
        return acc