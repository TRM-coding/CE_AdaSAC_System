from argparse import Namespace
import torch
from torch import nn
# from . import model3

class train_based_self_detection():
    def __init__(self,model_path,learning_rate,epoch,device):
        self.Param=Namespace(
            lr=learning_rate,
            epoch=epoch
        )
        self.model_path=model_path
        self.model=torch.load(model_path)
        self.device=device
        self.model.to(device)
        self.model.eval()
        self.loss_list=[]
    
    def make_data(self,batch_size,input_size,output_size,randn_magnification,confidence):
        data=torch.randn(batch_size,input_size,input_size,requires_grad=True).to(self.device)
        output_lable=(torch.randn(batch_size,output_size)*randn_magnification).to(self.device)
        max_index=output_lable.argmax(dim=1)
        lable=max_index
        output_lable[torch.arange(batch_size),max_index]=confidence
        
        data=data.clone().detach().requires_grad_(True)
        loss_function=nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([data], lr=self.Param.lr)

        
        loss_list=[]
        for epoch in range(self.Param.epoch):
            optimizer.zero_grad()
            output=self.model(data)
            loss=loss_function(output,output_lable.softmax(dim=1))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        self.loss_list=loss_list

        return data,output_lable,lable

    def show_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_list)
        plt.show()

    