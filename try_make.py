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
import time
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("CODE:loading_resnet50")
    # model=Resnet50Loader().load()
    # model=AlexnetLoader().load()
    model=VGG16Loader().load()
    print("CODE:loading_finished")

    device=CONFIG.DEFAULT_DEVICE
    back_device=CONFIG.DEFAULT_DEVICE
    cut_step=CONFIG.CUT_STEP
    datamaker=train_based_self_detection(
        model=model,
        device=device,
        no_weight=True
    )

    time1=time.time()
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

    time2=time.time()
    print(time2-time1)
 