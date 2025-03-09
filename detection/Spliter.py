import torch
import numpy as np
# import SVD
import gc
# from detection.SVD import SVD
from .Model_transfer import Model_transfer
from .SVD_model import SVDED_Conv,SVDED_Linear
import copy
from torch import nn

from thop import profile
from detection.splited_model import Splited_Model
from tqdm import tqdm
import sys
import time

from .Loader.mymodel_file.MobileNetv3 import *
from .Loader.mymodel_file.Alexnet import *
from .Loader.mymodel_file.Resnet50 import *
from .Loader.mymodel_file.VGG16Net import *
# import multiprocessing as mp
import torch.multiprocessing as mp
import multiprocessing

import concurrent.futures
import functools

import random
import threading
import queue

from .splited_model import Splited_Model

from multiprocessing import Process, Queue,Manager

from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver




class Recrusively_reduce_search:
    def __init__(self,model,input_data,output_label,label,
                 highest_loss,lowest_loss,network_speed,device,
                 local_speed,cloud_speed,acc_cut_point,back_device,no_weight=True):
        self.model=model.to(device)
        self.model.eval()
        self.used_for_search=copy.deepcopy(self.model)
        self.device=device
        self.back_device=back_device
        self.svder=Model_transfer(model,device)
        self.input_data=input_data.to(device)
        self.input_data=self.input_data.detach()

        self.output_label=output_label.to(device)
        self.label=label.to(device)
        

        self.solution_cnt=0 #有多少个解
        self.F_loss=[] #原始loss
        self.F_latency=[] #原始时间
        self.F_list=[] #评估分值
        self.F_acc=[] #原始acc
        self.best_scheme=[] #最优划分策略
        self.best_F=0 
        self.best_latency=0
        self.best_loss=0
        self.best_acc=0
        self.best_partition_model=[] #最优划分模型
        self.highest_loss=highest_loss
        self.lowest_loss=lowest_loss
        self.network_speed=network_speed
        self.local_speed=local_speed
        self.cloud_speed=cloud_speed
        self.max_latency=0
        self.min_latency=0x3f3f3f3f
        self.svded_layers=[]
        self.is_child={}

        self.acc_cut_point=acc_cut_point
        self.init_size=80
        
        self.generate_epoch=30
        self.network_latency=0
        return

    def move_to_device(self,device):
        self.model.to(device)
        self.input_data.to(device)
        self.output_label.to(device)
        self.label.to(device)
        self.device=self.back_device=device
        return
    
    # def acc_loss_evaluate(self,optimized_model):
    #     # start_time=time.time()
    #     output=optimized_model(self.input_data)
    #     max_index=output.argmax(dim=1)
    #     acc=(max_index==self.label).sum().item()/self.label.shape[0]
        
    #     loss_function=torch.nn.CrossEntropyLoss()
    #     loss=loss_function(output,self.output_label.softmax(dim=1))
    #     # end_time=time.time()
    #     # self.inference_time+=end_time-start_time
    #     return acc,loss.item()
    
    def init(self,reduce_step):
        x=self.input_data
        self.model.eval()
        flops_cloud,_=profile(self.model,inputs=(x,))
        torch.cuda.empty_cache()
        flops_edge=0
        for _,layer in self.model.named_children():
            layer.eval()
            layer_flops,_=profile(layer,inputs=(x,))
            torch.cuda.empty_cache()
            flops_edge+=layer_flops
            flops_cloud-=layer_flops
            latency_compute_i=flops_edge/self.local_speed+flops_cloud/self.cloud_speed
            x=layer(x)
            latency_network_i=x.element_size()*x.numel()/self.network_speed
            self.max_latency=max(self.max_latency,latency_compute_i+latency_network_i)
            self.min_latency=min(self.min_latency,latency_compute_i+latency_network_i)


        with open('Profile_INFO.txt','w')as f:
            print("max_latency",self.max_latency)
            print("min_latency",self.min_latency)
            x=self.input_data.to(self.back_device)

            sys.stdout=f
            for name,layer in tqdm(self.model.named_children()):
                layer.eval()
                if(not (isinstance(layer,nn.Linear) or 
                        isinstance(layer,nn.Conv2d) or 
                        isinstance(layer,Conv2dNormActivation)or
                        isinstance(layer,SqueezeExcitation)or
                        isinstance(layer,MyBot)
                    )
                ):
                    lateri=layer.to(self.back_device)
                    flops,_=profile(lateri,inputs=(x,))
                    
                    torch.cuda.empty_cache()
                    setattr(layer,'flops',flops)
                    x=layer(x)
                    net_Bytes=x.numel()*x.element_size()
                    setattr(layer,'netBytes',net_Bytes)
                    layer.to(self.device)
                    continue
                if(
                    isinstance(layer,Conv2dNormActivation) or 
                    isinstance(layer,SqueezeExcitation) or
                    isinstance(layer,MyBot)
                ):
                    ip=x
                    output=layer(x)
                    net_Bytes=output.numel()*output.element_size()
                    setattr(layer,'netBytes',net_Bytes)
                    for name_c,layer_c in tqdm(layer.named_children()):
                        if(not(isinstance(layer_c,nn.Conv2d) or isinstance(layer_c,nn.Linear))):
                            continue
                        self.svded_layers.append([])
                        self.is_child[len(self.svded_layers)-1]=name
                        for reduce_rate in np.arange(0,1,reduce_step):
                            c_svd=self.svder.layer_svd(layer_c,reduce_rate)
                            self.svded_layers[-1].append((name_c,c_svd))
                            c_svdi=c_svd.to(self.back_device)
                            flops,_=profile(c_svdi,inputs=(x if (isinstance(c_svd,SVDED_Conv) and c_svd.conv_layer.in_channels==x.shape[1]) else ip,))
                            c_svdi.to(self.device)
                            setattr(c_svd,'flops',flops)
                            torch.cuda.empty_cache()
                            # setattr(c_svd,'netBytes',net_Bytes)
                        layer_ci=layer_c.to(self.back_device)
                        x=layer_ci(x if (isinstance(c_svd,SVDED_Conv) and c_svd.conv_layer.in_channels==x.shape[1]) else ip)
                        layer_ci.to(self.device)
                    x=output
                        
                else:
                    self.svded_layers.append([])
                    layeri=layer.to(self.back_device)
                    output=layeri(x)
                    layeri.to(self.device)
                    net_Bytes=output.numel()*output.element_size()
                    setattr(layer,'netBytes',net_Bytes)
                    for reduce_rate in np.arange(0,1,reduce_step):
                        c_svd=self.svder.layer_svd(layer,reduce_rate)
                        self.svded_layers[-1].append((name,c_svd)) 
                        c_svdi=c_svd.to(self.back_device)
                        flops,_=profile(c_svdi,inputs=(x,))
                        c_svd.to(self.device)
                        torch.cuda.empty_cache()
                        setattr(c_svd,'flops',flops)
                        setattr(c_svd,'netBytes',net_Bytes)
                    x=output
                    
            x=self.input_data.to(self.back_device)
            for name,layer in tqdm(self.model.named_children()):
                layer.eval()
                layeri=layer.to(self.back_device)
                flops,_=profile(layeri,inputs=(x,))
                
                torch.cuda.empty_cache()
                setattr(layer,'flops',flops)
                x=layeri(x)
                net_Bytes=x.numel()*x.element_size()
                setattr(layer,'netBytes',net_Bytes)
                layeri.to(self.device)
            
        sys.stdout=sys.__stdout__        
        print("SVD_finished")
        return


    def acc_loss_evaluate(self,optimized_model:nn.modules):
        with torch.no_grad():
        # start_time=time.time()
            ip=self.input_data.to(self.device)
            opl=self.output_label.to(self.device)
            lb=self.label.to(self.device)
            optimized_model.to(self.device)
            output=optimized_model(ip)
            max_index=output.argmax(dim=1)
            acc=(max_index==lb).sum().item()/lb.shape[0]
            
            loss_function=torch.nn.CrossEntropyLoss()
            loss=loss_function(output,opl.softmax(dim=1))
            # end_time=time.time()
            # self.inference_time+=end_time-start_time
        return acc,loss.item()

    def dfs_get_flops(self,layer):
        flops=0
        if(isinstance(layer,SVDED_Conv) or isinstance(layer,SVDED_Linear)):
            return layer.flops
        for _,layeri in layer.named_children():
            flops+=self.dfs_get_flops(layeri)
        return flops

    def flops_evaluate(self,model_edge_A,model_cloud,model_edge_B):
        
        total_flops_edge=0
        total_flops_cloud=0

        for layer in model_edge_A.model_list:
            total_flops_edge+=self.dfs_get_flops(layer)
   
        for layer in model_cloud.model_list:
            total_flops_cloud+=layer.flops
        

        for layer in model_edge_B.model_list:
            total_flops_edge+=self.dfs_get_flops(layer)

        return total_flops_edge,total_flops_cloud

    def latency_evaluate(self,model_edge_A,model_cloud,model_edge_B):
        with open('Profile_INFO.txt','w')as f:
            sys.stdout=f
            flops_local,flops_cloud=self.flops_evaluate(model_edge_A,model_cloud,model_edge_B)
            sys.stdout=sys.__stdout__
        return flops_local/self.local_speed+flops_cloud/self.cloud_speed
    
    def network_evaluate(self,model_edge_A):
        return model_edge_A.model_list[-1].netBytes/self.network_speed
    
    def network_evaluate_quantisized(self,quantisized_model,quantisized_type):
        ip=self.input_data
        observer=MovingAveragePerChannelMinMaxObserver(ch_axis=1,dtype=quantisized_type).to(self.device)
        observer.eval()
        op=quantisized_model.model_list[0](ip)
        observer(op)
        scale,zero_point=observer.calculate_qparams()
        # print(scale,zero_point)
        op_quantized=torch.quantize_per_channel(op,scales=scale,zero_points=zero_point,axis=1,dtype=quantisized_type)
        memory_bits=op_quantized.nelement()*op_quantized.element_size()
        return memory_bits/self.network_speed
        
    
    def Total_F(self,model,alpha,model_edge_A,model_cloud,model_edge_B):
        model.eval()
        model.to(self.back_device)
        acc,loss=self.acc_loss_evaluate(model)
        model.to(self.device)
        normaled_loss=(loss-self.lowest_loss)/(self.highest_loss-self.lowest_loss)
        compute_latency=self.latency_evaluate(
            model_edge_A=model_edge_A,
            model_cloud=model_cloud,
            model_edge_B=model_edge_B)
        # if(self.network_latency==0):
        #     input()
        # network=0
        network=self.network_latency

        normaled_time=((compute_latency+network)-self.min_latency)/(self.max_latency-self.min_latency)

        F=alpha*(np.exp(1-normaled_loss))+(1-alpha)*np.exp(1-normaled_time)
        return F,compute_latency+network,loss,acc,self.network_latency
    
    # def Total_F(self,model,alpha,quantisized_model):
    #     model.eval()
    #     model.to(self.back_device)
    #     acc,loss=self.acc_loss_evaluate(model)
    #     model.to(self.device)
    #     normaled_loss=(loss-self.lowest_loss)/(self.highest_loss-self.lowest_loss)

    #     compute_latency=self.latency_evaluate(
    #         model_edge_A=model_edge_A,
    #         model_cloud=model_cloud,
    #         model_edge_B=model_edge_B)
    #     self.network_latency=self.network_evaluate(model_edge_A=model_edge_A)
    #     if(self.network_latency==0):
    #         input()
    #     network=0
        

    #     normaled_time=((compute_latency+network)-self.min_latency)/(self.max_latency-self.min_latency)

    #     F=alpha*(np.exp(1-normaled_loss))+(1-alpha)*np.exp(1-normaled_time)
    #     return F,compute_latency+network,loss,acc,self.network_latency

    def taski(self,tasks,q,gpu_usage:list):
        task_gpu=gpu_usage.index(min(gpu_usage))
        gpu_usage[task_gpu]+=1
        self.move_to_device(task_gpu)
        torch.cuda.empty_cache()
        print("子进程任务量:",len(tasks))
        F_score_list=[0 for _ in tasks]
        schemes=[]
        for idx,ti in enumerate(tasks):
            speciesi,alpha=ti
            schemes.append(speciesi)
            model,layer_map=self.model_reduce(speciesi)
            eA,c,eB=self.split(model,len(layer_map))
            F_i,latency,loss,acc,net_latency=self.Total_F(model,alpha,eA,c,eB)
            F_score_list[idx]=F_i
            # torch.cuda.empty_cache()
        
        q.put(
            (
                copy.deepcopy(schemes),
                copy.deepcopy(F_score_list),
                (
                    latency,
                    loss,
                    acc,
                    net_latency
                )
            )
        )
        torch.cuda.empty_cache()
        gpu_usage[task_gpu]-=1
        return
        
    def sigmoid(self,x):
        return (1-np.exp(-x))/(1+np.exp(-x))
    def ifexpand(self,max_cut,cut_i,alpha)->bool:
        if(abs(cut_i-alpha*max_cut)==0):
            return True
        score=1/abs((cut_i-alpha*max_cut))
        p=self.sigmoid(score)
        r=random.random()
        if(r<p):
            return True
        else:
            return False
    
    def search_GA(self,number_of_layer_to_reduce,alpha,step=0.1):
        upper_bound=self.GA_init(number_of_layer_to_reduce,step)
        print("每层裁减上限:",upper_bound)
        init_species=[]
        for i in range(self.init_size):
            listi=[random.randint(0,upper_bound[j]) for j,_ in enumerate(range(number_of_layer_to_reduce))]
            init_species.append(listi)
        lass_F=0
        generate_epoch=self.generate_epoch
        # F_score_list=[0 for _ in range(len(init_species))]
        scnt=0
        numworker=6
        pool = multiprocessing.Pool(processes=numworker)
        manager=Manager()
        q=manager.Queue()
        gpu_usage=manager.list()
        # gpu_usage={}
        # 除了0号GPU外，其他GPU都是空闲的
        for i in range(0,7):
            gpu_usage.append(0)
        gpu_usage[0]=1000000
        while(generate_epoch):
            F_score_list=[]
            st=time.time()
            results = []
            threads=[]
            task_list=[]
            torch.cuda.empty_cache()
            # partial_task = functools.partial(self.taski)
            for idx,speciesi in enumerate(init_species):
                task_list.append((speciesi,alpha))
            
            i=0
            torch.cuda.empty_cache()

            
        # 将任务分配给进程池中的进程
            print("开始分配进程,总任务量:",len(task_list))
            pool.starmap(self.taski, [ 
                (task_list[i:min(len(task_list), i + len(task_list) // numworker)],q,gpu_usage) 
                                  for i in range(0, len(init_species), len(init_species) // numworker)])
            
            
            print("All tasks are completed.")
            ed=time.time()
            print("Generate_time:",ed-st)
            
            F_score_list.clear()
            init_species=[]
            while not q.empty():
                results.append(q.get())
            for ki in results:
                schemes = ki[0]
                for i in range(len(schemes)):
                    init_species.append(tuple(schemes[i]))
                f_list=ki[1]
                F_score_list+=f_list
                f_i=max(f_list)
                max_idx=f_list.index(f_i)
                latency,loss,acc,net_latency=ki[2]
                self.network_latency=net_latency
                # TODO:splited_after_finished

                if(f_i>lass_F):
                    self.best_scheme=schemes[max_idx]
                    self.best_acc=acc
                    self.best_loss=loss
                    self.best_latency=latency+self.network_latency
                    lass_F=f_i    
                    print("a new one")
                torch.cuda.empty_cache()
            print("latency:",latency)
            print("loss:",loss)
            
            unique_dict=dict(zip(init_species,F_score_list))
            unique_species=list(unique_dict.keys())
            unique_F=list(unique_dict.values())

            sum_cut_list=[sum(sublist) for sublist in unique_species]
            max_cut=max(sum_cut_list)

            len_u=len(unique_species)
            for i in range(len_u):
                cut_i=sum(unique_species[i])
                if(self.ifexpand(max_cut=max_cut,cut_i=cut_i,alpha=(1-alpha))):
                    unique_species.append(unique_species[i])
                    unique_F.append(unique_F[i])
                    

            cb=list(zip(unique_species,unique_F))

            cbsorted=sorted(cb,key=lambda x:x[1],reverse=True)
            cbsorted=cbsorted[:self.init_size]
            init_species=[x[0] for x in cbsorted]
            F_score_list=[x[1] for x in cbsorted]


            sums=sum(F_score_list)
            for i,_ in enumerate(F_score_list):
                F_score_list[i]/=sums
            F_Integral=[]
            for i in range(len(F_score_list)):
                F_Integral.append(0)
                for j in range(i+1):
                    F_Integral[i]+=F_score_list[j]

            maxx=len(init_species)
            
            for _ in range(maxx):
                r=random.random()
                father=[]
                for i,p in enumerate(F_Integral):
                    if len(father)>1:
                        break
                    if r<p:
                        father.append(init_species[i])
                
                if(len(father)<=1):
                    continue
                
                r=random.random()
                son1=father[0][:int(len(father[0])*r)]+father[1][int(len(father[0])*r):]
                son2=father[1][:int(len(father[0])*r)]+father[0][int(len(father[0])*r):]
                init_species.append(son1)
                init_species.append(son2)
        
            #Save DATA
            self.F_list.append(lass_F)
            self.F_loss.append(self.best_loss)
            self.F_latency.append(self.best_latency)
            self.F_acc.append(self.best_acc)

            print("lass_f:",lass_F)
            print()
            scnt+=1
            generate_epoch-=1
            print("solutions:",scnt,end='\r')
        
        pool.close()
        pool.join()
        manager.shutdown()


    def GA_init(self,number_of_layer_to_reduce,step):
        self.F_loss=[]
        self.F_latency=[]
        self.F_list=[]
        self.F_acc=[]
        self.best_scheme=[]
        self.best_partition_model=[]

        upper_bound=[0 for _ in range(number_of_layer_to_reduce)]
        upper_bound_0=[0 for _ in range(number_of_layer_to_reduce)]
        for i in range(0,number_of_layer_to_reduce):
            reduce_rate=[0 for _ in range(i+1)]
            for reduce_rate_j in np.arange(0,1/step):
                reduce_rate[-1]=int(reduce_rate_j)
                model,_=self.model_reduce(reduce_rate)
                model.eval()
                acc,loss=self.acc_loss_evaluate(model)
                if(acc<self.acc_cut_point):
                    break
                else:
                    upper_bound[i]=reduce_rate_j
            reduce_rate[i]=0
            print(f"第{i}层上限检测:",end='\r')
            torch.cuda.empty_cache()
        upper_bound=[int(i) for i in upper_bound]
        model_max_reduce,layer_map=self.model_reduce(upper_bound)
        model_min_reduce,_=self.model_reduce(upper_bound_0)
        eA_worst,c_worst,eB_worst=self.split(model_max_reduce,len(layer_map))
        eA_best,c_best,eB_best=self.split(model_min_reduce,len(layer_map))
        acc_worst,loss_worst=self.acc_loss_evaluate(model_max_reduce)
        time_best=self.latency_evaluate(eA_worst,c_worst,eB_worst)
        time_worst=self.latency_evaluate(eA_best,c_best,eB_best)

        self.highest_loss=loss_worst
        self.max_latency=time_worst
        self.min_latency=time_best
        print("Loss上限:",loss_worst)
        print("acc上限:",acc_worst)
        print("最长耗时:",self.max_latency)
        print("最短耗时:",self.min_latency)
        torch.cuda.empty_cache()
        return upper_bound
        
    def model_reduce(self,reduce_rate:list):
        model=copy.deepcopy(self.model)
        # model=self.model
        edge_layer_map={}
        for i,reduce_index in enumerate(reduce_rate):
            name=self.svded_layers[i][reduce_index][0]
            svded_layer=self.svded_layers[i][reduce_index][1]
            if(i in self.is_child):
                father_name=self.is_child[i]
                father=getattr(model,father_name)
                svded_layer=self.svded_layers[i][reduce_index][1]
                setattr(father,name,svded_layer)
                edge_layer_map[father_name]=1
            else:
                setattr(model,name,svded_layer)
                edge_layer_map[name]=1
            torch.cuda.empty_cache()
        model.eval()
        return model,edge_layer_map


    def split(self,model,split_number):
        cloud_model=Splited_Model()
        cloud_model.eval()
        edge_model_A=Splited_Model()
        edge_model_A.eval()
        edge_model_B=Splited_Model()
        edge_model_B.eval()
        model_list=[]
        for _,layer in model.named_children():
            model_list.append(layer)
        
        Tot=split_number
        i=0    
        while(i<Tot):
            if(not (isinstance(model_list[i],SVDED_Conv) or 
                    isinstance(model_list[i],SVDED_Linear) or 
                    isinstance(model_list[i],SqueezeExcitation) or
                    isinstance(model_list[i],Conv2dNormActivation)or
                    isinstance(model_list[i],MyBot))):
                Tot=Tot+1
            edge_model_A.model_list.append(model_list[i])
            i=i+1
            
        for i in range(Tot,len(model_list)-1):
            cloud_model.model_list.append(model_list[i])
        if(isinstance(model_list[-1],SVDED_Linear) or isinstance(model_list[-1],SVDED_Conv)):
            cloud_model.model_list.append(model_list[-1].U)
            edge_model_B.model_list.append(model_list[-1].V)
            edge_model_B.model_list.append(model_list[-1].bias)
        else:
            edge_model_B.model_list.append(model_list[-1])
        return edge_model_A,cloud_model,edge_model_B
    