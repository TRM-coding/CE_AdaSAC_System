model_path='./model/mlp.pth'
device='cuda:4'
from detection.DataGenerator import train_based_self_detection
import gc
import torch

maker=train_based_self_detection(
    model_path=model_path,
    device=device,
)
input_data,output_label,label= maker.make_data_pid(
    batch_size=100,
    learning_rate=1,
    channel=1,
    dim1=28,
    dim2=28,
    output_size=10,
    randn_magnification=10,
    confidence=1000
)
import detection.Model_transfer
import detection.Spliter
import importlib
importlib.reload(detection.Spliter)
importlib.reload(detection.Model_transfer)

searcher=detection.Spliter.Recrusively_reduce_search(
    model_path=model_path,
    input_data=input_data,
    output_label=output_label,
    label=label,
    device=device
)

acc_data=[]
loss_data=[]
best_partition=[]

with torch.no_grad():

    for i in range(1,3):
        searcher.search(
            number_of_layer_to_reduce=i,
            alpha=0.5,
            acc_data=acc_data,
            loss_data=loss_data,
            step=0.05,
        )
        
        gc.collect()
# for i in best_partition:
#     print(i)
#     print()
