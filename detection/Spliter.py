import torch
import numpy as np
# import SVD
import gc
# from detection.SVD import SVD
from detection.Model_transfer import Model_transfer
from detection.SVD_model import SVDED_Conv,SVDED_Linear
import copy
from torch import nn

from thop import profile
from detection.splited_model import Splited_Model
from tqdm import tqdm
import sys
import time




class Recrusively_reduce_search:
    def __init__(self,model,model_path,input_data,output_label,label,
                 highest_loss,lowest_loss,network_speed,device,
                 local_speed,cloud_speed,acc_cut_point):
        self.model=model.to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.used_for_search=copy.deepcopy(self.model)
        self.device=device
        self.svder=Model_transfer(model,model_path,device)
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
        # self.flops_map={}
        return
    
    def init(self,reduce_step):
        x=self.input_data
        flops_cloud,_=profile(self.model,inputs=(x,))
        flops_edge=0
        for _,layer in self.model.named_children():
            layer_flops,_=profile(layer,inputs=(x,))
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
                if(not (isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv2d)) or isinstance(layer,nn.Sequential)):
                    flops,_=profile(layer,inputs=(x,))
                    setattr(layer,'flops',flops)
                    x=layer(x)
                    net_Bytes=x.numel()*x.element_size()
                    setattr(layer,'netBytes',net_Bytes)
                    continue
                if(isinstance(layer,nn.Sequential)):
                    for name_c,layer_c in tqdm(layer.named_children()):
                        self.svded_layers.append([])
                        self.is_child[len(self.svded_layers)-1]=name
                        output=layer_c(x)
                        net_Bytes=output.numel()*output.element_size()
                        setattr(layer_c,'netBytes',net_Bytes)
                        for reduce_rate in np.arange(0,1,reduce_step):
                            c_svd=self.svder.layer_svd(layer_c,reduce_rate)
                            self.svded_layers[-1].append((name_c,c_svd))
                            flops,_=profile(c_svd,inputs=(x,))
                            setattr(c_svd,'flops',flops)
                            setattr(c_svd,'netBytes',net_Bytes)
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
                        setattr(c_svd,'flops',flops)
                        setattr(c_svd,'netBytes',net_Bytes)
                    x=output
                    
            x=self.input_data
            for name,layer in tqdm(self.model.named_children()):
                flops,_=profile(layer,inputs=(x,))
                setattr(layer,'flops',flops)
                x=layer(x)
                net_Bytes=x.numel()*x.element_size()
                setattr(layer,'netBytes',net_Bytes)
            sys.stdout=sys.__stdout__
                
            print("SVD_finished")
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


    # def acc_evaluate(self,optimized_model):
    #     # test_input=self.input_data.to(self.device)
    #     # test_label=self.label.to(self.device)
    #     output=optimized_model(self.input_data)
    #     max_index=output.argmax(dim=1)
    #     acc=(max_index==self.label).sum().item()/self.label.shape[0]
    #     return acc
    
    # def loss_evaluate(self,optimized_model):
        
    #     output=optimized_model(self.input_data)
    #     loss_function=torch.nn.CrossEntropyLoss()
    #     loss=loss_function(output,self.output_label.softmax(dim=1))
    #     return loss.item()
    
    def flops_evaluate(self,model_edge_A,model_cloud,model_edge_B):
        
        total_flops_edge=0
        total_flops_cloud=0
        
        for layer in model_edge_A.model_list:
            total_flops_edge+=layer.flops
   
        for layer in model_cloud.model_list:
            total_flops_cloud+=layer.flops
        

        for layer in model_edge_B.model_list:
            total_flops_edge+=layer.flops

        return total_flops_edge,total_flops_cloud

    def latency_evaluate(self,model_edge_A,model_cloud,model_edge_B):
        with open('Profile_INFO.txt','w')as f:
            sys.stdout=f
            flops_local,flops_cloud=self.flops_evaluate(model_edge_A,model_cloud,model_edge_B)
            sys.stdout=sys.__stdout__
        return flops_local/self.local_speed+flops_cloud/self.cloud_speed
    
    def network_evaluate(self,model_edge_A):
        # data_number=0
        # test_input=self.input_data

        # edge_A_output=model_edge_A(test_input)
        # data_number=edge_A_output.numel()*edge_A_output.element_size()
        # return data_number/self.network_speed
        return model_edge_A.model_list[-1].netBytes/self.network_speed
        
    
    def Total_F(self,model,alpha,model_edge_A,model_cloud,model_edge_B):
        
        acc,loss=self.acc_loss_evaluate(model)
        normaled_loss=(loss-self.lowest_loss)/(self.highest_loss-self.lowest_loss)

        compute_latency=self.latency_evaluate(
            model_edge_A=model_edge_A,
            model_cloud=model_cloud,
            model_edge_B=model_edge_B)
        network=self.network_evaluate(model_edge_A=model_edge_A)
        

        normaled_time=((compute_latency+network)-self.min_latency)/(self.max_latency-self.min_latency)

        F=alpha*(np.exp(1-normaled_loss))+(1-alpha)*np.exp(1-normaled_time)
        self.F_loss.append(np.exp(1-normaled_loss))
        self.F_latency.append(np.exp(1-normaled_time))
        self.F_list.append(F)
        return F,compute_latency+network,loss,acc
        

    
    def search_r(self,number_of_layer_to_reduce,alpha,acc_data,loss_data,step=0.1,reduce_rate=[]):
        flag=True
        if(len(reduce_rate)==number_of_layer_to_reduce):
            print("serched_solution:",self.solution_cnt,end='\r')
            self.solution_cnt+=1
            
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
            if(not (isinstance(model_list[i],SVDED_Conv) or isinstance(model_list[i],SVDED_Linear) or isinstance(model_list[i],nn.Sequential))):
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