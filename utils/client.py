import socket
import pickle
import torch
import requests
import time

from splited_model import Splited_Model

device='cuda:5'

edge_A=torch.load('./p_model/edge_A.pth')
edge_A=edge_A.to(device)
edge_B=torch.load('./p_model/edge_B.pth')
edge_B=edge_B.to(device)



start_time=time.time()

tensor_input=torch.randn(1,3,224,224).to(device)
A_out=edge_A(tensor_input)
print("边侧中间数据已上传")

payload = {
    "data": A_out.tolist()
}

response = requests.post("http://127.0.0.1:5000/endpoint", json=payload)
cloud_data = response.json()['result']
print("边侧二次推理")
output=edge_B(torch.tensor(cloud_data).to(device))
print(output.shape)

end_time=time.time()

print("总耗时:",end_time-start_time)

eA=torch.load('./p_model/edge_A.pth').to(device)
cloud=torch.load('./p_model/cloud.pth').to(device)
eB=torch.load('./p_model/edge_B.pth').to(device)

start_time=time.time()


o1=eA(tensor_input)
print('fi-o1')
o2=cloud(o1)
print('fi-o2')
o3=eB(o2)

end_time=time.time()

print("边侧独立推理总耗时:",end_time-start_time)

