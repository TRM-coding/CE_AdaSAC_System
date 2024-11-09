from torchvision import datasets,transforms
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))])



def load_data(train_batch_size,test_batch_size,device): 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print('load data set')
    train_dataset = datasets.MNIST(root='/home/tianruiming/Eckart-young-based-mlsys/data',train=True,download=False,transform=transform)
    test_dataset  = datasets.MNIST(root='/home/tianruiming/Eckart-young-based-mlsys/data',train=False,download=False,transform=transform)

    print('create loader')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    


    train_input = []
    train_lable = []
    test_input  = []
    test_lable  = []

    print('construct data')
    for images,labels in tqdm(train_loader):
        train_input.append(images.to(device))
        train_lable.append(labels.to(device))

    for images,labels in tqdm(test_loader):
        test_input.append(images.to(device))
        test_lable.append(labels.to(device))
    
    return train_input,train_lable,test_input,test_lable

    


        
    

        
    