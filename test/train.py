import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from argparse import Namespace
from model import MLP
from torch import nn
from dataloader import load_data
from tqdm import tqdm
param=Namespace(
    batch_size    = 60000,
    lr            = 1e-4,
    epoch         = 10000,
    
)


train_inputs,train_lables,test_inputs,test_lables=load_data(param.batch_size,param.batch_size,device=torch.device('cuda:5'))



mlp = MLP()
mlp.train()

device = torch.device('cuda:5')
mlp.to(device,non_blocking=True)
optimizer = torch.optim.SGD(mlp.parameters(), lr=param.lr)


loss_function = nn.CrossEntropyLoss()

for epoch in range(param.epoch):
    ls=0
    for i in range(len(train_inputs)): 
        inputi  = train_inputs[i]
        label  = train_lables[i]
        output = mlp(inputi)
        loss   = loss_function(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ls+=loss
        print("finished")
    print(f"epoch{epoch} loss:{loss}")

    



