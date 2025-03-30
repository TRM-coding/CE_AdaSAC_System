import torch
from detection.evaler import eval
from detection.config import CONFIG

from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
import detection.Spliter
import torch.multiprocessing as mp
from torch import nn
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver


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
    print("CODE:loading_finished")

    device=CONFIG.DEFAULT_DEVICE
    back_device=CONFIG.DEFAULT_DEVICE
    cut_step=CONFIG.CUT_STEP
    datamaker=train_based_self_detection(
        model=model,
        device=device,
        no_weight=True
    )

   

    

    # input_data,output_label,label,highest_loss,lowest_loss= datamaker.make_data_pid(
    #         total_number=CONFIG.TEST_DATA_TOTAL_NUMBER,
    #         batch_size=CONFIG.TEST_DATA_BATCH_SIZE,
    #         learning_rate=CONFIG.TEST_DATA_LEARNING_RATE,
    #         warm_lr=CONFIG.TEST_DATA_WARM_LR,
    #         channel=CONFIG.TEST_DATA_CHANNEL,
    #         dim1=CONFIG.TEST_DATA_DIM1,
    #         dim2=CONFIG.TEST_DATA_DIM2,
    #         output_size=CONFIG.TEST_DATA_OUTPUT_SIZE,
    #         randn_magnification=CONFIG.TEST_DATA_RANDN_MAGNIFICATION,
    #         confidence=CONFIG.TEST_DATA_CONFIDENCE,
    #         target_acc=CONFIG.TEST_DATA_TARGET_ACC

    # )
    inputs=datamaker.make_data_img()
    print(type(inputs))
    print(inputs[0][0].shape)
    little_batch=inputs[0][0]
    torch.cuda.empty_cache()

    with torch.no_grad():
        searcher=detection.Spliter.Recrusively_reduce_search(
                model=model,
                no_weight=True,
                input_data=little_batch,
                output_label=little_batch,
                label=little_batch,
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
        #model,edge_layer_map=searcher.model_reduce([4,4,2,5,3,5,5,1,5,1,4,2,0,5,6,7,5,6,4,2,5,4,0,5,3,4])
        # model,edge_layer_map=searcher.model_reduce([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        model,edge_layer_map=searcher.model_reduce([4])
        # model,edge_layer_map=searcher.model_reduce([0,0])
        print("layer_map_len:",len(edge_layer_map))
        eA,c,eB=searcher.split(model,len(edge_layer_map))

        qm=quantiseze_model([eA,c,eB])

        print("start eval")

    
        elaver=eval(inputs,qm)# remenber to change it
        loss,acc=elaver.eval()
        print("loss:",loss," acc:",acc)
        torch.save(eA,"./clientA.pth")
        torch.save(c,"./clientB.pth")
        torch.save(eB,"./clientC.pth")
        print("CODE:finish")
