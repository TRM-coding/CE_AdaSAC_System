model_path='./model/best_VGG16Net.pth'
device='cuda:5'

from detection.DataGenerator import train_based_self_detection
from alex import AlexNet
from detection.Loader.mymodel_file.VGG16Net import VGG16

model=VGG16()

maker=train_based_self_detection(
    model_path=model_path,
    device=device,
    model=model
)
input_data,output_label,label,highest_loss,lowest_loss= maker.make_data_pid(
    batch_size=50,
    learning_rate=1,
    channel=3,
    dim1=32,
    dim2=32,
    output_size=100,
    randn_magnification=10,
    confidence=1000
)


import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from argparse import Namespace
# from model import MLP

from torch import nn
from dataloader import load_data
from tqdm import tqdm
# train_inputs,train_lables,test_inputs,test_lables=load_data(100,100,device=torch.device(device))
# input_data,output_label,label= maker.make_origin_data(test_inputs,test_lables)

from matplotlib import pyplot as plt
plt.plot(maker.loss_list)

plt.savefig('./tables/pic2.png')

import detection.Model_transfer
import detection.Spliter
import importlib
importlib.reload(detection.Spliter)
importlib.reload(detection.Model_transfer)
import torch

searcher=detection.Spliter.Recrusively_reduce_search(
    model=model,
    model_path=model_path,
    input_data=input_data,
    output_label=output_label,
    label=label,
    device=device,
    highest_loss=highest_loss,
    lowest_loss=lowest_loss,
    # local_speed=2.72e10,   #Flops/s
    local_speed=9.6e9,   #Flops/s
    cloud_speed=1.7e13,    #Flops/s
    network_speed=1e7,     #B/s
    acc_cut_point=0.8,
)

acc_data=[]
loss_data=[]
best_partition=[]
reduce_step=0.1
searcher.init(reduce_step)

import time
start_time=time.time()

with torch.no_grad():

    for i in range(1,5):
        print(f'------------------------------------------------------------- 裁剪层数:{i}')
        searcher.search(
            number_of_layer_to_reduce=i,
            alpha=0.3,
            acc_data=acc_data,
            loss_data=loss_data,
            step=reduce_step,
            
        )
        # import gc
        # gc.collect()
# for i in best_partition:
#     print(i)
#     print()
end_time=time.time()
run_time=end_time-start_time
print("搜索时长:",run_time)
print("推理时长:",searcher.inference_time)
# print("flops评估时长:",searcher.flops_time)
# print(searcher.F_acc)
# print(searcher.F_latency)
print(searcher.best_scheme)
# print(searcher.loss_data)



from matplotlib import pyplot as plt
import numpy as np
plt.clf()
plt.plot(searcher.F_loss,label='F_loss')
plt.plot(searcher.F_latency,label='F_latency')
# plt.plot(searcher.loss_data,label='origin_loss')
# plt.plot(searcher.acc_data,label='origin_acc')
# plt.scatter(np.arange(len(searcher.F_list)),searcher.F_list)
plt.plot(searcher.F_list,label='F')
print("best_index:",searcher.F_list.index(max(searcher.F_list)))
# plt.scatter(np.arange(len(searcher.F_latency)),searcher.F_latency)
plt.legend()

plt.savefig('./tables/pic3.png')

edge_A=searcher.best_partition[0]
cloud=searcher.best_partition[1]
edge_B=searcher.best_partition[2]

print(edge_A)
print(cloud)
print(edge_B)

print("best_acc:",searcher.best_acc)
print("best_loss:",searcher.best_loss)
print("best_latency:",searcher.best_latency)
# print(searcher.loss_data)

torch.save(edge_A,'./p_model/edge_A.pth')
torch.save(cloud,'./p_model/cloud.pth')
torch.save(edge_B,'./p_model/edge_B.pth')