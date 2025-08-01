from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
from detection.Loader.VGG16Loader import VGG16Loader
from detection.Loader.AlexnetLoader import AlexnetLoader
import detection.Spliter
import torch.multiprocessing as mp
import torch
from torch import nn
import numpy as np
from detection.config import CONFIG
from thop import profile

if __name__ == '__main__':

    np_task_number_change=[]
    np_f_change=[]

    mp.set_start_method('spawn', force=True)
    print("CODE:loading_alex")
    model=VGG16Loader().load()
    # model=AlexnetLoader().load()
    # model=Resnet50Loader().load()
    print("CODE:loading_finished")
    device=CONFIG.DEFAULT_DEVICE
    back_device=CONFIG.DEFAULT_DEVICE
    quantisized_type=CONFIG.QUANTISIZED_TYPE
    cut_step=CONFIG.CUT_STEP

    from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
    observer = MovingAveragePerChannelMinMaxObserver(ch_axis=0,dtype=quantisized_type).to(device)
    observer.eval()
    class quantiseze_model(nn.Module):
        def __init__(self,model_list):
            super(quantiseze_model,self).__init__()
            self.model_list=model_list
            self.eval()
        def forward(self,x):
            x=self.model_list[0](x)
            observer(x)
            scale, zero_point = observer.calculate_qparams()
            zero_point=torch.zeros_like(zero_point).to(device=x.device)
            x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=quantisized_type)
            x=self.model_list[1](x_quantized.dequantize())
            observer(x)
            scale, zero_point = observer.calculate_qparams()
            zero_point=torch.zeros_like(zero_point).to(device=x.device)
            x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=quantisized_type)
            x=self.model_list[2](x_quantized.dequantize())
            return x
        
    datamaker=train_based_self_detection(
        model=model,
        device=device,
        no_weight=True
    )

    input_data,output_label,label,highest_loss,lowest_loss= datamaker.make_data_pid(
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
    print(input_data.dtype)
    torch.cuda.empty_cache()

    with torch.no_grad():
        searcher=detection.Spliter.Recrusively_reduce_search(
                model=model,
                no_weight=True,
                input_data=input_data,
                output_label=output_label,
                label=label,
                device=device,
                back_device=back_device,
                highest_loss=highest_loss,
                lowest_loss=lowest_loss,
                local_speed=CONFIG.LOCAL_SPEED,   #Flops/s
                # local_speed=9.6e9,   #Flops/s
                cloud_speed=CONFIG.CLOUD_SPEED,    #Flops/s
                network_speed=CONFIG.NETWORK_SPEED,     #B/s
                acc_cut_point=CONFIG.ACC_CUT_POINT,
        )
        searcher.init(cut_step)

        searcher.input_data.shape
        upper_bound=searcher.GA_init(CONFIG .MODEL_LAYER_NUMBER,step=cut_step)

        print()
        upper_num=min(upper_bound)
        quantized_network_list=[]
        quantized_compute_list=[]
        quantized_time_list=[]
        quantized_acc_list=[]
        Qe_list=[]
        Qc_list=[]
        Qt_list=[]
        for i in range(1,CONFIG.MODEL_LAYER_NUMBER):
            torch.cuda.empty_cache()
            cut_num=int((max(upper_bound[:i+1])+min(upper_bound[:i+1]))/2)
            model_r,edge_layer_map_r=searcher.model_reduce([cut_num]*i)
            model_nr,edge_layer_map_nr=searcher.model_reduce([0]*i)
            eA_r,c_r,eB_r=searcher.split(model_r,len(edge_layer_map_r))
            # print(model_r)
            # input()
            eA_nr,c_nr,eB_nr=searcher.split(model_nr,len(edge_layer_map_nr))
            op1=eA_nr(torch.randn(2,3,224,224).to(device))
            op2=c_nr(op1)
            Qe_list.append(profile(eA_nr,inputs=(torch.randn(2,3,224,224).to(device),))[0]+profile(eB_nr,inputs=(op2.to(device),))[0])
            Qc_list.append(profile(c_nr,inputs=(op1.to(device),))[0])
            
            Qt_list.append(op1.numel()+op2.numel())
            qm_r=quantiseze_model([eA_r,c_r,eB_r])
            qm_r.eval()
            qm_nr=quantiseze_model([eA_nr,c_nr,eB_nr])
            qm_nr.eval()
            quantized_network_list.append(searcher.network_evaluate_quantisized(qm_r,quantisized_type))
            acc,loss=searcher.acc_loss_evaluate(qm_nr)
            compute_time=searcher.latency_evaluate(eA_r,c_r,eB_r)
            quantized_compute_list.append(compute_time)
            quantized_acc_list.append(acc)
            # quantized_time_list.append(compute_time+searcher.network_evaluate_quantisized(qm_r))
            quantized_time_list.append(quantized_network_list[-1]+quantized_compute_list[-1])
            # input()
            torch.cuda.empty_cache()

        

        print()
        print("quantized_acc_list:",quantized_acc_list)
        print("quantized_network_list:",quantized_network_list)
        print("max_compute_time:",max(quantized_compute_list))
        print("min_compute_time:",min(quantized_compute_list))
        print("quantized_time_list:",quantized_time_list)

        print()
        print("min_quantized_time_list:",min(quantized_time_list))
        print("max_quantized_time_list:",max(quantized_time_list))
        normaled_time=[(x-min(quantized_time_list))/(max(quantized_time_list)-min(quantized_time_list)) for x in quantized_time_list]
        # normaled_acc=[(x-min(quantized_acc_list))/(max(quantized_acc_list)-min(quantized_acc_list)) for x in quantized_acc_list]
        normaled_acc=[(1/(1+np.exp(np.mean(quantized_acc_list)-x))) for x in quantized_acc_list]
        print("normaled_time:",normaled_time)
        print("normaled_acc:",normaled_acc)

        import numpy as np
        alpha=CONFIG.ALPHA
        print()
        def F_score(alpha,index,pressure=False):
            if(pressure):
                return float((Qe_list[index]/CONFIG.LOCAL_SPEED)/((Qe_list[index]+Qc_list[index])/(CONFIG.CLOUD_SPEED)-Qc_list[index]/CONFIG.CLOUD_SPEED-Qt_list[index]/CONFIG.NETWORK_SPEED))
            return float(alpha*np.exp(1-normaled_time[index])-(1-alpha)*(np.exp((normaled_acc[index]-0.5)**3))-1)
        
        F_per_point=[F_score(alpha,i,CONFIG.PRESSURE_REDUCE) for i in range(0,CONFIG.MODEL_LAYER_NUMBER-1)]


        for i in range(0,CONFIG.MODEL_LAYER_NUMBER-1):
            print("idx:",i,"net_time:",quantized_network_list[i],"compute_time:",quantized_compute_list[i]," total_time:",quantized_compute_list[i]+quantized_network_list[i]," acc:",quantized_acc_list[i],"F_per_point:",F_per_point[i])
        F_per_point=[0]+F_per_point

        best_index=F_per_point.index(max(F_per_point))
        print("best_index:",best_index,"max_F:",min(F_per_point))

        

        
        # searcher.search_GA(
        #     number_of_layer_to_reduce=best_index,
        #     alpha=CONFIG.ALPHA,
        #     step=cut_step
        # )
        mapp=searcher.search_GA_warm(
            number_of_layer_to_reduce=best_index,
            step=cut_step
        )
        # print(mapp)
        
        # input("请确认开始进一步搜索：")
        _,np_task_number_change,np_f_change=searcher.searcer_GA_V2(
            init_specise=mapp,
            alpha_step=CONFIG.ALPHASTEP,
        )
        np.savez("np_task_number_change_vgg.npy",
                np_task_number_change=np_task_number_change,
                np_f_change=np_f_change
        )