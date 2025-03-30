

# from detection.DataGenerator import train_based_self_detection
# from alex import AlexNet
# from detection.Loader.mymodel_file.VGG16Net import VGG16
# from detection.Loader.ResNet50Loader import Resnet50Loader
# from detection.Loader.AlexnetLoader import AlexnetLoader
# import torch
# from torchvision import datasets,transforms
# from torch.utils.data import DataLoader
# from argparse import Namespace
# # from model import MLP

# from torch import nn
# from dataloader import load_data
# from tqdm import tqdm
# import torch.multiprocessing as mp
# import os
# import numpy as np


# if __name__=='__main__':

#     mp.set_start_method('spawn', force=True)

#     device='cuda:2'

#     model_ld=Resnet50Loader()
#     model=model_ld.load()


#     print(model)

#     maker=train_based_self_detection(
#         device=device,
#         model=model,
#         no_weight=True
#     )
#     input_data,output_label,label,highest_loss,lowest_loss= maker.make_data_pid(
#         total_number=100,
#         batch_size=100,
#         learning_rate=1,
#         warm_lr=1e-3,
#         channel=3,
#         dim1=224,
#         dim2=224,
#         output_size=1000,
#         randn_magnification=100,
#         confidence=1000000,
#         target_acc=0.8

#     )
#     input_data=input_data.detach()
#     output_label=output_label.detach()
#     torch.cuda.empty_cache()

#     from matplotlib import pyplot as plt
#     plt.plot(maker.loss_list)

#     plt.savefig('./tables/pic2.png')

#     import detection.Model_transfer
#     import detection.Spliter
#     import importlib
#     import torch

#     #4090-高通骁龙870八核
#     searcher=detection.Spliter.Recrusively_reduce_search(
#         model=model,
#         no_weight=True,
#         input_data=input_data,
#         output_label=output_label,
#         label=label,
#         device=device,
#         highest_loss=highest_loss,
#         lowest_loss=lowest_loss,
#         local_speed=2.72e10,   #Flops/s
#         # local_speed=9.6e9,   #Flops/s
#         cloud_speed=8.258688e13,    #Flops/s
#         network_speed=2e7,     #B/s
#         acc_cut_point=0.5,
#         # q=q,
#     )

#     reduce_step=0.1

#     torch.cuda.empty_cache()

#     import time
#     start_time=time.time()

#     print("开始搜索")
#     # input()
#     alpha_list=[0.2,0.4,0.5]
#     with torch.no_grad():
#         searcher.init(reduce_step)
#         for alpha in alpha_list:
#             torch.cuda.empty_cache()
#             print("alpha:",alpha)
#             for i in range(25,35):
#                 torch.cuda.empty_cache()
#                 print(f'------------------------------------------------------------- 裁剪层数:{i}')

#                 searcher.search_GA(
#                     number_of_layer_to_reduce=i,
#                     alpha=alpha,
#                     step=reduce_step
#                 )
#                 os.makedirs(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}', exist_ok=True)
#                 F_loss=np.array(searcher.F_loss)
#                 F_latency=np.array(searcher.F_latency)
#                 F_acc=np.array(searcher.F_acc)
#                 F_=np.array(searcher.F_list)
#                 best_scheme=np.array(searcher.best_scheme)

#                 np.savetxt(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}/F_loss.txt', F_loss)
#                 np.savetxt(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}/F_latency.txt',F_latency)
#                 np.savetxt(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}/F_acc.txt',F_acc)
#                 np.savetxt(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}/F_.txt',F_)
#                 np.savetxt(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}/best_scheme.txt',best_scheme)

#                 max_latency=searcher.max_latency
#                 min_latency=searcher.min_latency
#                 max_loss=searcher.highest_loss
#                 min_loss=searcher.lowest_loss
#                 network_latency=searcher.network_latency

#                 with open(f'./search_data_n20/layer_{i}_datas/alpha_{alpha}/vars.txt', 'w') as f:
#                     f.write(f'max_latency:{max_latency}\nmin_latency:{min_latency}\nmax_loss:{max_loss}\nmin_loss:{min_loss}\nnetwork_latency:{network_latency}')

            




from detection.DataGenerator import train_based_self_detection
# from alex import AlexNet
# from detection.Loader.mymodel_file.VGG16Net import VGG16
from detection.Loader.VGG16Loader import VGG16Loader
from detection.Loader.ResNet50Loader import Resnet50Loader
from detection.Loader.AlexnetLoader import AlexnetLoader
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from argparse import Namespace
# from model import MLP

from torch import nn
from detection.dataloader import load_data
from tqdm import tqdm
import torch.multiprocessing as mp
import os
import numpy as np


if __name__=='__main__':

    mp.set_start_method('spawn', force=True)

    device='cuda:6'
    back_device='cuda:7'

    model_ld=VGG16Loader()
    model=model_ld.load()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    # model = torch.nn.DataParallel(model).cuda()

    print(model)

    maker=train_based_self_detection(
        device=device,
        model=model,
        no_weight=True
    )
    input_data,output_label,label,highest_loss,lowest_loss= maker.make_data_pid(
        total_number=10,
        batch_size=10,
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
    input_data=input_data.detach()
    output_label=output_label.detach()
    torch.cuda.empty_cache()

    from matplotlib import pyplot as plt
    plt.plot(maker.loss_list)

    plt.savefig('./tables/pic2.png')

    import detection.Model_transfer
    import detection.Spliter
    import importlib
    import torch

    #4090-高通骁龙870八核
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
        cloud_speed=8.258688e13,    #Flops/s
        network_speed=2e7,     #B/s
        acc_cut_point=0.5,
        # q=q,
    )

    reduce_step=0.1

    torch.cuda.empty_cache()

    import time
    start_time=time.time()

    print("开始搜索")
    # input()
    alpha_list=[0.1,0.6,0.7,0.8,0.9]
    with torch.no_grad():
        searcher.init(reduce_step)
        searcher.model.to(searcher.device)
        for alpha in alpha_list:
            torch.cuda.empty_cache()
            for i in range(10,15):
                torch.cuda.empty_cache()
                print(f'------------------------------------------------------------- 裁剪层数:{i}')

                searcher.search_GA(
                    number_of_layer_to_reduce=i,
                    alpha=alpha,
                    step=reduce_step
                )
                os.makedirs(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}', exist_ok=True)
                F_loss=np.array(searcher.F_loss)
                F_latency=np.array(searcher.F_latency)
                F_acc=np.array(searcher.F_acc)
                F_=np.array(searcher.F_list)
                best_scheme=np.array(searcher.best_scheme)

                np.savetxt(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}/F_loss.txt', F_loss)
                np.savetxt(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}/F_latency.txt',F_latency)
                np.savetxt(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}/F_acc.txt',F_acc)
                np.savetxt(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}/F_.txt',F_)
                np.savetxt(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}/best_scheme.txt',best_scheme)

                max_latency=searcher.max_latency
                min_latency=searcher.min_latency
                max_loss=searcher.highest_loss
                min_loss=searcher.lowest_loss
                network_latency=searcher.network_latency

                with open(f'./search_data_n20_vgg/layer_{i}_datas/alpha_{alpha}/vars.txt', 'w') as f:
                    f.write(f'max_latency:{max_latency}\nmin_latency:{min_latency}\nmax_loss:{max_loss}\nmin_loss:{min_loss}\nnetwork_latency:{network_latency}')

        



