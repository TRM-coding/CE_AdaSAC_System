from argparse import Namespace
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
# from . import model3

class train_based_self_detection():
    def __init__(self,model_path,model,device,no_weight=True):
        # self.Param=Namespace(
        #     lr=learning_rate,
        #     epoch=epoch
        # )
        self.model_path=model_path
        self.model=model
        self.device=device
        self.model.to(device)
        self.model.eval()
        self.loss_list=[]
        if(not no_weight):
            self.model.load_state_dict(torch.load(model_path))
    
    def make_data(self,batch_size,epochs,learning_rate,channel,dim1,dim2,output_size,randn_magnification,confidence):
        data=torch.randn(batch_size,channel,dim1,dim2,requires_grad=True).to(self.device)
        output_lable=(torch.randn(batch_size,output_size)*randn_magnification).to(self.device)
        max_index=output_lable.argmax(dim=1)
        lable=max_index
        output_lable[torch.arange(batch_size),max_index]=confidence
        
        data=data.clone().detach().requires_grad_(True)
        loss_function=nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([data], lr=learning_rate)

        
        loss_list=[]
        
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            output=self.model(data)
            loss=loss_function(output,output_lable.softmax(dim=1))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # //print(loss.item(),end='\r')
        self.loss_list=loss_list

        return data,output_lable,lable

    def make_origin_data(self,input,label):
        inputs=input[0].to(self.device)
        labels=label[0].to(self.device)
        output=None
        with torch.no_grad():
            output=self.model(inputs)
        return inputs,output,labels
        

    class CustomLRScheduler(_LRScheduler):
        def __init__(self, optimizer: Optimizer, last_epoch: int = -1, warm_epoch=10,verbose: bool = False):
            self.last_loss = 0
            self.loss = 0
            self.warm_param=1e-3/int(warm_epoch)
            self.warm_epoch=warm_epoch
            self.current_lr=0
            super().__init__(optimizer, last_epoch, verbose)

        def get_lr(self):
            if self.last_loss is None or self.loss is None:
                return [base_lr for base_lr in self.base_lrs]

            # //print("current_lr",self.current_lr)
            # //print("loss_des:",abs(self.last_loss-self.loss) if abs(self.last_loss-self.loss)<1 else 1/abs(self.last_loss-self.loss))
            print("lr_update:",min(self.current_lr,self.base_lrs[0]* abs(self.last_loss-self.loss) if abs(self.last_loss-self.loss)<1 else 1/abs(self.last_loss-self.loss)))
            # //self.current_lr=min(self.current_lr,self.base_lrs[0]* abs(self.last_loss-self.loss) if abs(self.last_loss-self.loss)<1 else 1/abs(self.last_loss-self.loss))
            return [min(self.current_lr,base_lr* abs(self.last_loss-self.loss) if abs(self.last_loss-self.loss)<1 else 1/abs(self.last_loss-self.loss))  for base_lr in self.base_lrs]
            

        def step(self, epoch=0,last_loss=0, loss=0):
            if(epoch<=self.warm_epoch):
                self.current_lr+=self.warm_param
            if(abs(last_loss-loss)==0 and epoch!=0):
                # //print('stop')
                self.last_loss=self.loss=0
            else:
                self.last_loss=last_loss
                self.loss=loss
            super().step(epoch)

    def make_data_pid(self,batch_size,learning_rate,channel,dim1,dim2,output_size,randn_magnification,confidence):
        data=torch.rand(batch_size,channel,dim1,dim2,requires_grad=True).to(self.device)
        output_lable=(torch.rand(batch_size,output_size)*randn_magnification).to(self.device)
        max_index=output_lable.argmax(dim=1)
        lable=max_index
        output_lable[torch.arange(batch_size),max_index]=confidence
        
        data=data.clone().detach().requires_grad_(True)
        loss_function=nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([data], lr=learning_rate)
        scheduler = self.CustomLRScheduler(optimizer)

        loss_list=[]
        last_loss=0
        # for epoch in tqdm(range(self.Param.epoch)):
        epoch=0
        while(1):
            optimizer.zero_grad()
            output=self.model(data)
            loss=loss_function(output,output_lable.softmax(dim=1))
            loss.backward()
            scheduler.step(epoch,last_loss,loss.item())
            optimizer.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            # //print("current_lr:",current_lr)
            # //print()
            if(last_loss==loss.item() and epoch!=0):
                break
            last_loss=loss.item()
            loss_list.append(loss.item())
            epoch+=1
            if(epoch>=100):
                break
            # if(current_lr<1e-5):
            #     break
            # print(loss.item(),end='\r')
            
        self.loss_list=loss_list

        return data,output_lable,lable,self.loss_list[0],self.loss_list[-1]
    
    def make_data_less_than_acc(self,batch_size,learning_rate,channel,dim1,dim2,output_size,randn_magnification,confidence,target_acc):
        data=torch.randn(batch_size,channel,dim1,dim2,requires_grad=True).to(self.device)
        output_lable=(torch.randn(batch_size,output_size)*randn_magnification).to(self.device)
        max_index=output_lable.argmax(dim=1)
        lable=max_index
        output_lable[torch.arange(batch_size),max_index]=confidence
        
        data=data.clone().detach().requires_grad_(True)
        loss_function=nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([data], lr=learning_rate)
        # scheduler = self.CustomLRScheduler(optimizer)

        loss_list=[]
        last_loss=0
        # for epoch in tqdm(range(self.Param.epoch)):
        epoch=0
        while(1):
            optimizer.zero_grad()
            output=self.model(data)
            loss=loss_function(output,output_lable.softmax(dim=1))
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            epoch+=1
            max_index=output.argmax(dim=1)
            acc=(max_index==lable).sum().item()/lable.shape[0]
            if(acc>target_acc):
                break
            print("acc:",acc)
            print("loss:",loss.item()/batch_size)

            
        self.loss_list=loss_list
        return data,output_lable,lable,self.loss_list[0],self.loss_list[-1]

    def show_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_list)
        plt.show()
        loss_cf=[self.loss_list[i]-self.loss_list[i+1] for i in range(len(self.loss_list)-1)]
        plt.plot(loss_cf)
        plt.show()

    