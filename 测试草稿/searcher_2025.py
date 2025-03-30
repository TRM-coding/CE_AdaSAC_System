from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
import detection.Spliter
import torch
from torch import nn
print("CODE:loading_resnet50")
model=Resnet50Loader().load()
print("CODE:loading_finished")
device='cuda:1'
back_divice='cuda:1'
datamaker=train_based_self_detection(
    model=model,
    device=device,
    no_weight=True
)

input_data,output_label,label,highest_loss,lowest_loss= datamaker.make_data_pid(
        total_number=100,
        batch_size=100,
        learning_rate=1,
        warm_lr=1e-3,
        channel=3,
        dim1=224,
        dim2=224,
        output_size=1000,
        randn_magnification=100,
        confidence=1000000,
        target_acc=0.8

)
print (input_data.dtype)

torch.cuda.empty_cache()

searcher=detection.Spliter.Recrusively_reduce_search(
        model=model,
        no_weight=True,
        input_data=input_data,
        output_label=output_label,
        label=label,
        device=device,
        back_device=back_divice,
        highest_loss=highest_loss,
        lowest_loss=lowest_loss,
        # local_speed=2.72e10,   #Flops/s
        local_speed=9.6e9,   #Flops/s
        cloud_speed=1.7e13,    #Flops/s
        network_speed=1e7,     #B/s
        acc_cut_point=0.7,
        # q=q,
)
searcher.init(0.2)

from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
observer = MovingAveragePerChannelMinMaxObserver(ch_axis=0).to(device)

class quantiseze_model(nn.Module):
    def __init__(self,model_list):
        super(quantiseze_model,self).__init__()
        self.model_list=model_list
    def forward(self,x):
        x=self.model_list[0](x)
        observer(x)
        scale, zero_point = observer.calculate_qparams()
        # scale=scale.to(device)
        # zero_point=zero_point.to(device)
        zero_point=torch.zeros_like(zero_point).to(device=x.device)
        x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
        x=self.model_list[1](x_quantized.dequantize())
        observer(x)
        scale, zero_point = observer.calculate_qparams()
        zero_point=torch.zeros_like(zero_point).to(device=x.device)
        # scale=scale.to(device)
        # zero_point=zero_point.to(device)
        x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
        x=self.model_list[2](x_quantized.dequantize())
        return x
searcher.model.to(device)
model,edge_layer_map=searcher.model_reduce([0,0,0,0,0,0,0,0,0,0,0])
eA,c,eB=searcher.split(model,len(edge_layer_map))

qm=quantiseze_model([eA,c,eB])
qm.to(device)
print(searcher.acc_loss_evaluate(qm))

