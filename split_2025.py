import torch
from detection.evaler import eval
from detection.config import CONFIG

from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
from detection.Loader.VGG16Loader import VGG16Loader
from detection.Loader.AlexnetLoader import AlexnetLoader
import detection.Spliter
import torch.multiprocessing as mp
from torch import nn
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
import random
import numpy as np
import matplotlib.pyplot as plt

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
    model=Resnet50Loader().load()
    # model=AlexnetLoader().load()
    # model=VGG16Loader().load()
    print("CODE:loading_finished")

    device=CONFIG.DEFAULT_DEVICE
    back_device=CONFIG.DEFAULT_DEVICE
    cut_step=CONFIG.CUT_STEP
    datamaker=train_based_self_detection(
        model=model,
        device=device,
        no_weight=True
    )


    inputs_img=datamaker.make_data_img()
    inputs_maked,output_label,label,highest_loss,lowest_loss= datamaker.make_data_pid(
            total_number=CONFIG.TEST_DATA_TOTAL_NUMBER,
            batch_size=CONFIG.TEST_DATA_BATCH_SIZE,
            learning_rate=CONFIG.TEST_DATA_LEARNING_RATE,
            warm_lr=CONFIG.TEST_DATA_WARM_LR,
            channel=CONFIG.TEST_DATA_CHANNEL,
            dim1=CONFIG.TEST_DATA_DIM1,
            dim2=CONFIG.TEST_DATA_DIM2,
            output_size=CONFIG.TEST_DATA_OUTPUT_SIZE,
            randn_magnification=CONFIG.TEST_DATA_RANDN_MAGNIFICATION,
            confidence=CONFIG.TEST_DATA_CONFIDENCE,
            target_acc=CONFIG.TEST_DATA_TARGET_ACC

    )
    # inputs_maked=(inputs_maked.to(device),label.to(device))
    # inputs_maked=[(inputs_maked[i].to(device),label[i].to(device)) for i in range(len(inputs_maked))]

    little_batch=inputs_img[0][0]
    torch.cuda.empty_cache()

    with torch.no_grad():
        searcher=detection.Spliter.Recrusively_reduce_search(
                model=model,
                no_weight=True,
                input_data=inputs_maked,
                output_label=output_label,
                label=label,
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
        searcher.init(cut_step)
        torch.cuda.empty_cache()
        searcher.model.to(device)
        #model,edge_layer_map=searcher.model_reduce([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 4, 2, 0, 0, 1, 2, 0, 0, 0, 0, 2])
        # model,edge_layer_map=searcher.model_reduce([0,0,0])
        
        # model,edge_layer_map=searcher.model_reduce([2, 2, 3, 3, 2, 4, 2, 2, 3, 0, 3, 4, 1, 2, 5, 8, 6, 5, 0, 2, 1, 3, 0, 3, 2, 4])
        # model,edge_layer_map=searcher.model_reduce([4, 1, 6, 5, 4, 6, 2, 5, 6, 1, 5, 5, 3, 5, 6, 8, 6, 4, 5, 4, 5, 5, 2, 4, 4, 5])
        # Res50:
        model,edge_layer_map=searcher.model_reduce([4,7,5,6,4,7,7,4,6,4,4,2])
        # model,edge_layer_map=searcher.model_reduce([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # VGG16:
        # model,edge_layer_map=searcher.model_reduce([1,3,2,2,3,5,4,4])
        # model,edge_layer_map=searcher.model_reduce([5,6,7,7,8,6,7,4])
        #alex:
        # model,edge_layer_map=searcher.model_reduce([0,0])
        # eA,c,eB=searcher.split(model,len(edge_layer_map))
        # qm=quantiseze_model([eA,c,eB])
        eA,c,eB=searcher.split(model,len(edge_layer_map))

        qm=quantiseze_model([eA,c,eB])
        print("start eval")
        elaver=eval(inputs_img,qm)# remenber to change it
        loss,acc=elaver.eval()
        print("loss:",loss," acc:",acc)
        torch.save(eA,"./clientA_res50.pth")
        torch.save(c,"./clientB_res50.pth")
        torch.save(eB,"./clientC_res50.pth")
        

        print("CODE:finish")
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# from thop import profile, clever_format
if __name__=="__main__":
    eA=torch.load("./clientA_res50.pth",weights_only=False)
    c=torch.load("./clientB_res50.pth",weights_only=False)
    input=torch.rand(2,3,224,224).to('cuda:3')
    inter=eA(input)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 如果只测 GPU FLOPs，也可以去掉 CPU
        record_shapes=True,      # 记录各 op 的输入输出形状，才能估 FLOPs
        with_flops=True,         # 打开 FLOPs 统计
        profile_memory=False     # 不统计显存分配，保持输出简洁
    ) as prof:
    # 用 record_function 标记你关心的计算区间
        out=c(inter)
        
    # flops, params = profile(eA, inputs=(input,))
    print("shape_of_inter:",inter.shape)
    print("edge_flos:---------------------------")
    # print(flops)
    print(prof.key_averages().table(sort_by="flops", row_limit=20))

    # with profile(
    #     activities=[ProfilerActivity.CUDA],  # 如果只测 GPU FLOPs，也可以去掉 CPU
    #     record_shapes=True,      # 记录各 op 的输入输出形状，才能估 FLOPs
    #     with_flops=True,         # 打开 FLOPs 统计
    #     profile_memory=False     # 不统计显存分配，保持输出简洁
    # ) as proff:
        
    print("cloud_flops:-------------------------")
    # print(proff.key_averages().table(sort_by="flops", row_limit=20))
    
