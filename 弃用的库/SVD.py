#该库已弃用




# import torch
# import copy
# from torch import nn

# class Bias(nn.Module):
#     def __init__(self,b):
#         super().__init__()
#         self.bias=nn.Parameter(b)
    
#     def forward(self,x):
#         return x+self.bias
    
# class NewModel(nn.Module):
#     def __init__(self):
#         super(NewModel,self).__init__()
#         self.layer_list=nn.ModuleList([])
#     def create_module_from_name(module_name, *args, **kwargs):
  
#         if hasattr(nn, module_name):
#             module_class = getattr(nn, module_name)
#             return module_class(*args, **kwargs)
        
#         else:
#             raise ValueError(f"No such module: {module_name}")
    
#     def forward(self,x):
#         output=copy.deepcopy(x)
#         output=output.view(-1,28*28)
#         for layer in self.layer_list:
#             output=layer.forward(output)
#         return output

# class model_transfer():
#     def __init__(self):
#         self.transfer_res=[]
#         return    
#     def jit_transfer(self,jit_model):
#         x=0
#         for i in jit_model.named_children():
#             x=x+1
#         if(x==0):
#             self.transfer_res.append(jit_model)
#         for name,child in jit_model.named_children():
#             self.jit_transfer(child)



# class SVD():

#     class Conv(nn.Module):
#         def __init__(self,conv_layer,SVD_obj,reduce_rate):
#             super(SVD.Conv,self).__init__()
#             self.conv_layer=conv_layer
#             self.weight=conv_layer.weight.view(conv_layer.out_channels,-1)
#             print(self.conv_layer.kernel_size)
#             self.bias=conv_layer.bias
#             self.SVD_obj=SVD_obj
#             self.svded=self.SVD_obj.SVD_Linear(reduce_rate=reduce_rate,weight=self.weight,bias=self.bias)
            
#         def forward(self,x):
#             x_unfold=nn.Unfold(
#                 kernel_size=self.conv_layer.kernel_size,
#                 stride=self.conv_layer.stride,
#                 padding=self.conv_layer.padding
#             )(x)
#             x_permute=x_unfold.permute(0,2,1)
#             output=self.svded(x_permute)

#             output_permute=output.permute(0,2,1)

#             output_H=int((x.shape[2]+2*self.conv_layer.padding[0]-self.conv_layer.kernel_size[0])/self.conv_layer.stride[0]+1)
#             output_W=int((x.shape[3]+2*self.conv_layer.padding[1]-self.conv_layer.kernel_size[1])/self.conv_layer.stride[1]+1)
#             output_res=output_permute.view(output_permute.shape[0],output_permute.shape[1],output_H,output_W)
#             return output_res
    

#     def __init__(self,model_path,device):
#         #// self.model=torch.jit.load(model_path)
#         #// self.MT=model_transfer()
#         #// self.MT.jit_transfer(self.model)
#         #// self.model_list=self.MT.transfer_res
#         self.model=torch.load(model_path)
#         self.device=device
#         return
    
#     def SVD_Conv(self,layer,reduce_rate):
#         Conv_layer=self.Conv(layer,self,reduce_rate)
#         return Conv_layer

#     def SVD_Linear(self,reduce_rate,weight=None,bias=None,layer=None,sorted=True):
#         w=None
#         b=None
#         if(layer==None):
#             w=weight
#             b=bias
#         else:
#             w=layer.weight
#             b=layer.bias
#         U,S,V=torch.linalg.svd(w.t())
#         if(sorted):
#             sort_index=torch.argsort(S)
#             U=U[:,sort_index]
#             S=S[sort_index]
#             V=V[sort_index,:]
#         else:
#             U=U[:,:len(S)]
#             V=V[:len(S),]
#         r=int(len(S)*reduce_rate)
#         U=U[:,r:]
#         V=V[r:,:]
#         S=torch.diag(S[r:])

#         for i in range(min(V.shape[0],V.shape[1])):
#             V[i]=V[i]*S[i][i]

#         newlinear1=torch.nn.Linear(U.shape[0],U.shape[1],bias=False).to(self.device)
#         #// newlinear2=torch.nn.Linear(S.shape[0],S.shape[1],bias=False).to(self.device)
#         newlinear3=torch.nn.Linear(V.shape[0],V.shape[1],bias=False).to(self.device)
#         newlinear1.weight=nn.Parameter(U.t())
#         #// newlinear2.weight=nn.Parameter(S.t())
#         newlinear3.weight=nn.Parameter(V.t())
#         newbias=Bias(b).to(self.device)
#         #// SVD_model.layer_list.append(newlinear1)
#         #// SVD_model.layer_list.append(newlinear2)
#         #// SVD_model.layer_list.append(newlinear3)
#         #// svded=nn.Sequential(newlinear1,newlinear2,newlinear3,newbias)
#         svded=nn.Sequential(newlinear1,newlinear3,newbias)
#         #// jis_svded=torch.jit.script(svded)
#         #// layers=svded
#         #// setattr(layer,"Linear",jis_svded)
#         return svded


#     def based_on_reduce_rate(self,reduce_rate=0,reduce_bound_of_layer_conv=-1,reduce_bound_of_layer_Linear=-1,sorted=True):
#         SVD_model=copy.deepcopy(self.model)
#         cnt_conv=0
#         cnt_linear=0
#         for k,layer in enumerate(SVD_model.layers):
            
#             if(isinstance(layer,nn.Linear)):
#                 if(cnt_linear<=reduce_bound_of_layer_Linear ):
#                 # //SVD_model.layer_list.append(layer)
                    
#                     cnt_linear=cnt_linear+1
#                     # print("Linear")

#                     svded_layer=self.SVD_Linear(layer=layer,reduce_rate=reduce_rate)
#                     SVD_model.layers[k]=svded_layer
#             if(isinstance(layer,nn.Conv2d)):
#                 if(cnt_conv<=reduce_bound_of_layer_conv ):
#                     cnt_conv=cnt_conv+1
#                     # print("Conv")

#                     svded_layer=self.SVD_Conv(layer,reduce_rate)
#                     SVD_model.layers[k]=svded_layer

#         return SVD_model
    
#     def based_on_reduce_rate_in_layers(self,reduce_rate=[],reduce_bound=-1,sorted=True):
#         SVD_model=copy.deepcopy(self.model)
#         cnt=0
#         for k,layer in enumerate(SVD_model.layers):
#             if(cnt>reduce_bound):
#                 return SVD_model
#             if(isinstance(layer,nn.Linear)):
#                 # print("Linear:",k)
#                 svded_layer=self.SVD_Linear(layer=layer,reduce_rate=reduce_rate[cnt])
#                 cnt=cnt+1
#                 SVD_model.layers[k]=svded_layer
#             if(isinstance(layer,nn.Conv2d)):
#                 # print("Conv")
#                 svded_layer=self.SVD_Conv(layer,reduce_rate[cnt])
#                 cnt=cnt+1
                
#                 SVD_model.layers[k]=svded_layer
    
#     def loss_evaluation(self,model,inputs,outputs_label):
#         model.eval()
#         model_output=model(inputs)
#         loss_function=nn.CrossEntropyLoss()
#         evaled_loss=loss_function(model_output,outputs_label.softmax(dim=1)).item()
#         return evaled_loss

#     def acc_evaluation(self,model,inputs,labels):
#         model.eval()
#         model_output=model(inputs)
#         model_label=model_output.argmax(dim=1)
#         acc=(model_label==labels).sum().item()/len(inputs)
#         return acc
    


