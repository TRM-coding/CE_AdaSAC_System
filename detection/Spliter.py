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

from splited_model import Splited_Model

from multiprocessing import Process, Queue,Manager




class Recrusively_reduce_search:
    def __init__(self,model,input_data,output_label,label,
                 highest_loss,lowest_loss,network_speed,device,
                 local_speed,cloud_speed,acc_cut_point,no_weight=True,model_path=None):
        self.model=model.to(device)
        if(not no_weight):
            self.model.load_state_dict(torch.load(model_path))
        self.used_for_search=copy.deepcopy(self.model)
        self.device=device
        self.svder=Model_transfer(model,device)
        self.input_data=input_data.to(device)
        self.output_label=output_label.to(device)
        self.label=label.to(device)
        self.loss_data=[] # 模型的原始loss数组
        self.acc_data=[]  #模型的原始acc
        self.solution_cnt=0 #有多少个解
        self.F_loss=[] #归一化过后的loss
        self.F_latency=[] #归一化过后的时间
        self.F_list=[] #评估分值
        self.best_scheme=[] #最优划分策略
        self.best_F=0 
        self.best_latency=0
        self.best_loss=0
        self.best_acc=0
        self.best_partition=[] #最优划分模型
        self.highest_loss=highest_loss
        self.lowest_loss=lowest_loss
        self.network_speed=network_speed
        self.local_speed=local_speed
        self.cloud_speed=cloud_speed
        self.max_latency=0
        self.min_latency=0x3f3f3f3f
        self.svded_layers=[]
        self.is_child={}
        self.inference_time=0
        self.flops_time=0
        self.acc_cut_point=acc_cut_point
        self.init_size=100
        self.GA_show=[]
        self.generate_epoch=20
        # self.flops_map={}
        self.GA_worker_results=[]
        # self.q=q
        return
    
    def acc_loss_evaluate(self,optimized_model):
        # start_time=time.time()
        output=optimized_model(self.input_data)
        max_index=output.argmax(dim=1)
        acc=(max_index==self.label).sum().item()/self.label.shape[0]
        
        loss_function=torch.nn.CrossEntropyLoss()
        loss=loss_function(output,self.output_label.softmax(dim=1))
        # end_time=time.time()
        # self.inference_time+=end_time-start_time
        return acc,loss.item()
    
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
            x=self.input_data
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
                    flops,_=profile(layer,inputs=(x,))
                    torch.cuda.empty_cache()
                    setattr(layer,'flops',flops)
                    x=layer(x)
                    net_Bytes=x.numel()*x.element_size()
                    setattr(layer,'netBytes',net_Bytes)
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
                            flops,_=profile(c_svd,inputs=(x if (isinstance(c_svd,SVDED_Conv) and c_svd.conv_layer.in_channels==x.shape[1]) else ip,))
                            setattr(c_svd,'flops',flops)
                            torch.cuda.empty_cache()
                            # setattr(c_svd,'netBytes',net_Bytes)
                        x=layer_c(x if (isinstance(c_svd,SVDED_Conv) and c_svd.conv_layer.in_channels==x.shape[1]) else ip)
                    
                    x=output
                        
                else:
                    self.svded_layers.append([])
                    output=layer(x)
                    net_Bytes=output.numel()*output.element_size()
                    setattr(layer,'netBytes',net_Bytes)
                    for reduce_rate in np.arange(0,1,reduce_step):
                        c_svd=self.svder.layer_svd(layer,reduce_rate)
                        self.svded_layers[-1].append((name,c_svd))       
                        flops,_=profile(c_svd,inputs=(x,))
                        torch.cuda.empty_cache()
                        setattr(c_svd,'flops',flops)
                        setattr(c_svd,'netBytes',net_Bytes)
                    x=output
                    
            x=self.input_data
            for name,layer in tqdm(self.model.named_children()):
                layer.eval()
                flops,_=profile(layer,inputs=(x,))
                torch.cuda.empty_cache()
                setattr(layer,'flops',flops)
                x=layer(x)
                net_Bytes=x.numel()*x.element_size()
                setattr(layer,'netBytes',net_Bytes)
            sys.stdout=sys.__stdout__
                
            print("SVD_finished")
            return


    def acc_loss_evaluate(self,optimized_model):
        with torch.no_grad():
        # start_time=time.time()
            output=optimized_model(self.input_data)
            max_index=output.argmax(dim=1)
            acc=(max_index==self.label).sum().item()/self.label.shape[0]
            
            loss_function=torch.nn.CrossEntropyLoss()
            loss=loss_function(output,self.output_label.softmax(dim=1))
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
        
    
    def Total_F(self,model,alpha,model_edge_A,model_cloud,model_edge_B):
        model.eval()
        acc,loss=self.acc_loss_evaluate(model)
        normaled_loss=(loss-self.lowest_loss)/(self.highest_loss-self.lowest_loss)

        compute_latency=self.latency_evaluate(
            model_edge_A=model_edge_A,
            model_cloud=model_cloud,
            model_edge_B=model_edge_B)
        self.network_latency=self.network_evaluate(model_edge_A=model_edge_A)
        network=0
        

        normaled_time=((compute_latency+network)-self.min_latency)/(self.max_latency-self.min_latency)

        F=alpha*(np.exp(1-normaled_loss))+(1-alpha)*np.exp(1-normaled_time)
        self.F_loss.append(np.exp(1-normaled_loss))
        self.F_latency.append(np.exp(1-normaled_time))
        self.F_list.append(F)
        return F,compute_latency+network,loss,acc
    

    def taski(self,tasks,q):
        torch.cuda.empty_cache()
        print("子进程任务量:",len(tasks))
        F_score_list=[0 for _ in tasks]
        schemes=[]
        for idx,ti in enumerate(tasks):
            speciesi,alpha=ti
            schemes.append(speciesi)
            model,layer_map=self.model_reduce(speciesi)
            eA,c,eB=self.split(model,len(layer_map))
            F_i,latency,loss,acc=self.Total_F(model,alpha,eA,c,eB)
            F_score_list[idx]=F_i
            torch.cuda.empty_cache()
        
        q.put(
            (
                copy.deepcopy(schemes),
                copy.deepcopy(F_score_list),
                (
                    latency,
                    loss,
                    acc,
                )
            )
        )
        torch.cuda.empty_cache()
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
        numworker=4
        pool = multiprocessing.Pool(processes=numworker)
        manager=Manager()
        q=manager.Queue()
        while(generate_epoch):
            F_score_list=[]
            st=time.time()
            results = []
            threads=[]
            task_list=[]
            # partial_task = functools.partial(self.taski)
            for idx,speciesi in enumerate(init_species):
                task_list.append((speciesi,alpha))
            
            i=0
            torch.cuda.empty_cache()
            
        # 将任务分配给进程池中的进程
            print("开始分配进程,总任务量:",len(task_list))
            pool.starmap(self.taski, [ 
                (task_list[i:min(len(task_list), i + len(task_list) // numworker)],q) 
                                  for i in range(0, len(init_species), len(init_species) // numworker)])
            
            
            print("All tasks are completed.")
            ed=time.time()
            print("time:",ed-st)
            # with mp.Pool(processes=3) as pool:
            #     results = list(pool.starmap(self.taski, task_list))
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
                latency,loss,acc=ki[2]
                # TODO:splited_after_finished

                if(f_i>lass_F):
                    self.best_scheme=schemes[max_idx]
                    # self.best_partition.clear()
                    # self.best_partition.append(eA)
                    # self.best_partition.append(c)
                    # self.best_partition.append(eB)
                    self.best_acc=acc
                    self.best_loss=loss
                    self.best_latency=latency
                    lass_F=f_i    
                    print("a new one")
            print("latency:",latency)
            print("loss:",loss)
            # if(generate_epoch==self)
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

            # print("种群:\n",unique_species)
                
                

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
                # print("vvnt:",vvnt,end='\r')
        

            self.GA_show.append(lass_F)
            self.F_loss.append(self.best_loss)
            self.F_latency.append(self.best_latency)
            print("lass_f:",lass_F)
            print()
            scnt+=1
            generate_epoch-=1
            print("solutions:",scnt,end='\r')
        
        # eA,c,eB=self.split(self.model,self.best_scheme)
        # self.best_partition.append(eA)
        # self.best_partition.append(c)
        # self.best_partition.append(eB)
        pool.close()
        pool.join()
        manager.shutdown()


    def GA_init(self,number_of_layer_to_reduce,step):
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
            print("第{i}层上限检测:",i,end='\r')
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
        return model,edge_layer_map

    
    def search_r(self,number_of_layer_to_reduce,alpha,acc_data,loss_data,step=0.1,reduce_rate=[]):
        flag=True
        if(len(reduce_rate)==number_of_layer_to_reduce):
            print("serched_solution:",self.solution_cnt,end='\r')
            self.solution_cnt+=1
            
            model,edge_layer_map=self.model_reduce(reduce_rate)
            
            edge_num=len(edge_layer_map)
            edge_A,cloud,edge_B=self.split(model,edge_num)
            F_score,this_latency,this_loss,this_acc=self.Total_F(
                model=model,
                alpha=alpha,
                model_edge_A=edge_A,
                model_cloud=cloud,
                model_edge_B=edge_B)

           
            self.acc_data.append(this_acc)
            self.loss_data.append(this_loss)
            

            if(this_acc<self.acc_cut_point):
                # print()
                # print("CUT")
                return False
            
            if(F_score>self.best_F):
                self.best_partition.clear()
                self.best_partition.append(edge_A)
                self.best_partition.append(cloud)
                self.best_partition.append(edge_B)
                self.best_F=F_score
                self.best_scheme=copy.deepcopy(reduce_rate)
                self.best_latency=this_latency
                self.best_loss=this_loss
                self.best_acc=this_acc

            return True
        
        for i in range(int(1/step)):
            reduce_rate.append(i)
            flag=self.search_r(number_of_layer_to_reduce,alpha,acc_data,loss_data,step,reduce_rate,)
            reduce_rate.pop()
            if(not flag):
                break
        return True
    
    
    def search(self,number_of_layer_to_reduce,alpha,acc_data,loss_data,step=0.1,reduce_rate=[]):
        self.search_r(number_of_layer_to_reduce,alpha,acc_data,loss_data,step=step,reduce_rate=reduce_rate)

    def split(self,model,split_scheme_list):
        cloud_model=Splited_Model()
        edge_model_A=Splited_Model()
        edge_model_B=Splited_Model()
        model_list=[]
        for _,layer in model.named_children():
            model_list.append(layer)
        
        Tot=split_scheme_list
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
    
    # def recrusively_split(
    #         self,
    #         split_scheme_list,
    #         svded_model,
    #         proceed_layer=[],
    #         flag=0,
    #         edge_model_A=Splited_Model(),
    #         cloud_model=Splited_Model(),
    #         data_channel_A_edge=None,
    #         data_channel_A_cloud=None,
    #         ):
        
    #     if(len(list(svded_model.children()))==0):
    #         if(svded_model in proceed_layer):
    #             return 
    #         if(len(proceed_layer)==len(split_scheme_list)):
    #             flag=1
    #             data_channel_A_edge=edge_model_A.model_list[-1]
    #             edge_model_A.model_list.pop()
    #             data_channel_A_cloud=svded_model
    #         else:

    #             if(flag):
    #                 cloud_model.model_list.append(svded_model)
    #             else:
    #                 edge_model_A.model_list.append(svded_model)
    #         proceed_layer.append(svded_model)


    #     for name,child in svded_model.named_children():
    #         self.recrusively_split(child)
    #     return edge_model_A,cloud_model,data_channel_A_edge,data_channel_A_cloud
    
    # def split_model(self,split_scheme_list,svded_model):
    #     model=copy.deepcopy(svded_model)
    #     model=self.svder.recrusively_update_based_on_list(model,split_scheme_list)
    #     edge_model_A,cloud_model,data_channel_A_edge,data_channel_A_cloud=self.recrusively_split(
    #         split_scheme_list=split_scheme_list,
    #         svded_model=model
    #     )
    #     data_channel_B_edge=cloud_model.model_list[-1]
    #     cloud_model.model_list.pop()
    #     data_channel_B_cloud=cloud_model.model_list[-1]
    #     cloud_model.model_list.pop()
    #     return edge_model_A,cloud_model,data_channel_A_edge,data_channel_A_cloud,data_channel_B_edge,data_channel_B_cloud