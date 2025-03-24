import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from detection.config import CONFIG
from multiprocessing import Process,Array,Lock,Manager
import multiprocessing
import tqdm
import copy


class eval:
    def __init__(self,inputs_and_targets,model):
        self.inputs_and_targets=inputs_and_targets
        self.total_number=CONFIG.TEST_DATA_TOTAL_NUMBER
        self.model=model
        return
    

    def taski(self,q,model,inputs_and_targets,gpu_usage:list,lock):
        
        torch.cuda.empty_cache()
        while(lock.lock):
            continue
        lock.lock=True

        if min(gpu_usage) in gpu_usage:
            task_gpu=gpu_usage.index(min(gpu_usage))
        else:
            task_gpu=gpu_usage.randint(0,len(gpu_usage)-1)
        lock.lock=False
        gpu_usage[task_gpu]+=1
        device=task_gpu
        print("子进程成功申请GPU:",task_gpu)
        
        model=self.model.to(device)
        model.eval()
        
        loss=0
        acc=0
        with torch.no_grad():
            #eval the model with imagenet
            for data in inputs_and_targets:
                data=list(data)
                data[0]=data[0].to(device)
                data[1]=data[1].to(device)
                output = model(data[0])
                criterion = torch.nn.CrossEntropyLoss()
                loss += criterion(output, data[1]).item()
                acc += (output.argmax(1) == data[1]).sum().item()
        q.put(copy.deepcopy((loss,acc)))
        gpu_usage[task_gpu]-=1
        torch.cuda.empty_cache()
        return
    
    def eval(self):
        self.model.eval()
        loss=0
        acc=0
        datas=self.inputs_and_targets
        pool=multiprocessing.Pool(processes=CONFIG.WORKERNUMBER)
        manager=Manager()
        q=manager.Queue()
        lock=manager.Namespace()
        lock.lock=False
        gpu_usage=manager.list()
        with torch.no_grad():
            #eval the model with imagenet

            for i in range(CONFIG.GPU_AVAILABLE[0],CONFIG.GPU_AVAILABLE[1]+1):
                gpu_usage.append(0)
            for i in CONFIG.UNAVAILABLE:
                gpu_usage[i]=1000000

            
            pool.starmap(self.taski,[(q,self.model,datas[i:min(len(datas),i+len(datas)//CONFIG.WORKERNUMBER)],gpu_usage,lock) for i in range(0,self.total_number,len(datas)//CONFIG.WORKERNUMBER)])
            pool.close()
            pool.join()
            while not q.empty():
                l,a=q.get()
                loss+=l
                acc+=a
            loss=loss/self.total_number
            acc=acc/self.total_number
            return loss,acc
                
                