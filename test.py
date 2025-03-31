from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
from detection.Loader.VGG16Loader import VGG16Loader
import detection.Spliter
import torch.multiprocessing as mp
import torch
from torch import nn
import numpy as np
from detection.config import CONFIG

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)
    print("CODE:loading_resnet50")
    model=VGG16Loader().load()
    print("CODE:loading_finished")
    input=torch.randn(5,3,224,224).to('cuda:3')
    model.to('cuda:3')
    output=model(input)
    print(output.shape)