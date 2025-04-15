import torch
from detection.evaler import eval
from detection.config import CONFIG

from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
from detection.Loader.VGG16Loader import VGG16Loader
import detection.Spliter
import torch.multiprocessing as mp
from torch import nn
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
import random
import numpy as np
import matplotlib.pyplot as plt
from thop import profile

class quantiseze_model(nn.Module):
    def __init__(self,model_list):
        super(quantiseze_model,self).__init__()
        self.model_list=nn.ModuleList(model_list)
        self.observer = MovingAveragePerChannelMinMaxObserver(ch_axis=0).to(next(self.parameters()).device)
    def forward(self,x):
        x=self.model_list[0](x)
        self.observer(x)
        scale, zero_point = self.observer.calculate_qparams()
        zero_point=torch.zeros_like(zero_point).to(device=x.device)
        x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
        x=self.model_list[1](x_quantized.dequantize())
        self.observer(x)
        scale, zero_point = self.observer.calculate_qparams()
        zero_point=torch.zeros_like(zero_point).to(device=x.device)
        x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
        x=self.model_list[2](x_quantized.dequantize())
        return x

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("CODE:loading_resnet50")
    res_50=Resnet50Loader().load()
    vgg_model=VGG16Loader().load()
    print("CODE:loading_finished")

    device=CONFIG.DEFAULT_DEVICE
    back_device=CONFIG.DEFAULT_DEVICE
    cut_step=CONFIG.CUT_STEP

    inputs_maked=torch.randn(2,3,224,224).to(CONFIG.DEFAULT_DEVICE)
    

    with torch.no_grad():
        searcher_res50=detection.Spliter.Recrusively_reduce_search(
                model=res_50,
                no_weight=True,
                input_data=inputs_maked,
                output_label=inputs_maked,
                label=inputs_maked,
                device=device,
                back_device=back_device,
                highest_loss=0,
                lowest_loss=0,
                local_speed=CONFIG.LOCAL_SPEED,   #Flops/s
                # local_speed=9.6e9,   #Flops/s
                cloud_speed=CONFIG.CLOUD_SPEED,    #Flops/s
                network_speed=CONFIG.NETWORK_SPEED,     #B/s
                acc_cut_point=CONFIG.ACC_CUT_POINT,
        )
        searcher_res50.init(cut_step)
        torch.cuda.empty_cache()
        searcher_res50.model.to(device)
        model_len=CONFIG.MODEL_LAYER_NUMBER
        #评估自生成数据
        maked_acc=[]
        maked_loss=[]
        print("start eval maked_data")
        model,edge_layer_map=searcher_res50.model_reduce([4,7])
        eA,c,eB=searcher_res50.split(model,len(edge_layer_map))
        res_50_list=[]
        flopsx, paramsx = profile(eA, inputs=(inputs_maked,))
        op=c(eA(inputs_maked))
        flopsy, paramsy = profile(eB, inputs=(op,))
        res50_flops=flopsx+flopsy
        res50_params=paramsx+paramsy

        print("res50_flops:",res50_flops)
        print("res50_params:",res50_params)
        
        qm=quantiseze_model([eA,c,eB])
        number_of_layer_reduce=CONFIG.EVAL_REDUCE_NUMBER
        
                    
        searcher_vgg=detection.Spliter.Recrusively_reduce_search(
                model=vgg_model,
                no_weight=True,
                input_data=inputs_maked,
                output_label=inputs_maked,
                label=inputs_maked,
                device=device,
                back_device=back_device,
                highest_loss=0,
                lowest_loss=0,
                local_speed=CONFIG.LOCAL_SPEED,   #Flops/s
                # local_speed=9.6e9,   #Flops/s
                cloud_speed=CONFIG.CLOUD_SPEED,    #Flops/s
                network_speed=CONFIG.NETWORK_SPEED,     #B/s
                acc_cut_point=CONFIG.ACC_CUT_POINT,
        )
        searcher_vgg.init(cut_step)
        torch.cuda.empty_cache()
        searcher_vgg.model.to(device)
        model_len=CONFIG.MODEL_LAYER_NUMBER
        #评估自生成数据
        maked_acc=[]
        maked_loss=[]
        print("start eval maked_data")
        model,edge_layer_map=searcher_vgg.model_reduce([4,7])
        eA,c,eB=searcher_vgg.split(model,len(edge_layer_map))
        res_50_list=[]
        flopsx, paramsx = profile(eA, inputs=(inputs_maked,))
        op=c(eA(inputs_maked))
        flopsy, paramsy = profile(eB, inputs=(op,))
        vgg_flops=flopsx+flopsy
        vgg_params=paramsx+paramsy
        
        qm=quantiseze_model([eA,c,eB])
        number_of_layer_reduce=CONFIG.EVAL_REDUCE_NUMBER
                    
        
        np.savez('./data_flops_parames_res50&vgg.npz',
            res50_flops=res50_flops,
            res50_params=res50_params,
            vgg_flops=vgg_flops,
            vgg_params=vgg_params)


        print("CODE:finish")
