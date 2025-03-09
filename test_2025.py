from detection.DataGenerator import train_based_self_detection
from detection.Loader.ResNet50Loader import Resnet50Loader
import detection.Spliter
import torch.multiprocessing as mp
import torch
from torch import nn

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)
    print("CODE:loading_resnet50")
    model=Resnet50Loader().load()
    print("CODE:loading_finished")
    device='cuda:6'
    back_device='cuda:6'
    quantisized_type=torch.qint8
    cut_step=0.2

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
            # scale=scale.to(device)
            # zero_point=zero_point.to(device)
            x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=quantisized_type)
            x=self.model_list[1](x_quantized.dequantize())
            observer(x)
            scale, zero_point = observer.calculate_qparams()
            # scale=scale.to(device)
            # zero_point=zero_point.to(device)
            x_quantized = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=0, dtype=quantisized_type)
            x=self.model_list[2](x_quantized.dequantize())
            return x
        
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
                local_speed=2.72e10,   #Flops/s
                # local_speed=9.6e9,   #Flops/s
                cloud_speed=1.7e13,    #Flops/s
                network_speed=1e7,     #B/s
                acc_cut_point=0.7,
                # q=q,
        )
        searcher.init(cut_step)

        searcher.input_data.shape
        upper_bound=searcher.GA_init(50,step=cut_step)

        print()
        upper_num=min(upper_bound)
        quantized_network_list=[]
        quantized_compute_list=[]
        quantized_time_list=[]
        quantized_acc_list=[]
        for i in range(1,50):
            torch.cuda.empty_cache()
            cut_num=int((max(upper_bound[:i+1])+min(upper_bound[:i+1]))/2)
            model_r,edge_layer_map=searcher.model_reduce([cut_num]*i)
            model_nr,edge_layer_map=searcher.model_reduce([0]*i)
            eA_r,c_r,eB_r=searcher.split(model_r,len(edge_layer_map))
            # print(model_r)
            # input()
            eA_nr,c_nr,eB_nr=searcher.split(model_nr,len(edge_layer_map))
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

        # normaled_acc_list=[(x-min(quantized_acc_list))/max(quantized_acc_list)-min(quantized_acc_list) for x in quantized_acc_list]
        # normaled_time_list=[(x-min(quantized_acc_list))/max(quantized_acc_list)]

        print()
        print("quantized_acc_list:",quantized_acc_list)
        print("quantized_network_list:",quantized_network_list)
        print("quantized_compute_list:",quantized_compute_list)
        print("quantized_time_list:",quantized_time_list)

        print()
        print("min_quantized_time_list:",min(quantized_time_list))
        print("max_quantized_time_list:",max(quantized_time_list))
        normaled_time=[(x-min(quantized_time_list))/(max(quantized_time_list)-min(quantized_time_list)) for x in quantized_time_list]
        normaled_acc=[(x-min(quantized_acc_list))/(max(quantized_acc_list)-min(quantized_acc_list)) for x in quantized_acc_list]
        print("normaled_time:",normaled_time)
        print("normaled_acc:",normaled_acc)

        import numpy as np
        alpha=0.5
        print()
        def F_score(alpha,index):
            return float(alpha*np.exp(1-normaled_time[index])+(1-alpha)*np.exp(normaled_acc[index]))
        F_per_point=[F_score(alpha,i) for i in range(0,48)]


        for i in range(0,48):
            print("idx:",i,"net_time:",quantized_network_list[i],"compute_time:",quantized_compute_list[i],"acc:",quantized_acc_list[i],"F_per_point:",F_per_point[i])
        F_per_point=[0,0]+F_per_point

        best_index=F_per_point.index(max(F_per_point))
        best_index

        

        
        searcher.search_GA(
            number_of_layer_to_reduce=best_index,
            alpha=0.7,
            step=cut_step
        )