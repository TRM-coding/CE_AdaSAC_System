import torch
from detection.config import CONFIG

from detection.DataGenerator import train_based_self_detection

from detection.splited_model import Splited_Model
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
import time
import httpx
import struct
import numpy as np
import base64
import torch
from thop import profile
import pickle

import socket

def quantisez(tensor:torch.Tensor,observer:MovingAveragePerChannelMinMaxObserver):
    observer(tensor)
    scale, zero_point = observer.calculate_qparams()
    zero_point=torch.zeros_like(zero_point)
    tensor_quantized = torch.quantize_per_channel(tensor, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
    return tensor_quantized

def dequantise(numpy_data:torch.tensor):
    detensor=numpy_data.dequantize()
    return detensor

def get_data(conn, chunk_size=4096):
    header_data = conn.recv(1024)
    if not header_data:

        raise Exception("No header received")
    try:
        data_length = pickle.loads(header_data)
    except Exception as e:
        raise Exception("Failed to unpickle header") from e
    
    # 发送 header 确认
    conn.sendall("HEADER RECEIVED".encode("utf-8"))
    
    # 分批接收实际数据
    received_bytes = 0
    chunks = []
    while received_bytes < data_length:
        chunk = conn.recv(min(chunk_size, data_length - received_bytes))
        if not chunk:
            break
        chunks.append(chunk)
        received_bytes += len(chunk)
    
    full_data = b"".join(chunks)
    
    # 发送数据接收完成的确认
    conn.sendall("DATA RECEIVED".encode("utf-8"))
    
    try:
        data = pickle.loads(full_data)
    except Exception as e:
        raise Exception("Failed to unpickle data") from e
    
    return data


def send_data(conn, data, chunk_size=4096):

    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    print("数据长度:", data_length)
    
    # 发送 header（数据长度）
    header = pickle.dumps(data_length)
    conn.sendall(header)
    
    # 等待 header 确认
    ack = conn.recv(1024)
    if ack.decode('utf-8') != "HEADER RECEIVED":
        raise Exception("Header acknowledgment not received")
    
    # 分批发送实际数据
    conn.sendall(serialized_data)
    
    # 等待数据确认
    ack2 = conn.recv(1024)
    if ack2.decode('utf-8') != "DATA RECEIVED":
        raise Exception("Data acknowledgment not received")


if __name__ == "__main__":
    print("CODE:云侧握手")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((CONFIG.CLOUDIP, CONFIG.CLOUD_PORT))
    print("CODE:握手成功")

    with torch.no_grad():     
        print("CODE:loading_model")
        model_A=Splited_Model()
        model_C=Splited_Model()
        observer = MovingAveragePerChannelMinMaxObserver(ch_axis=0)


        model_A_=torch.load("./clientA.pth",map_location=torch.device('cpu'))
        model_A_.to('cpu')
        model_A_.eval()
        model_C=torch.load("./clientC.pth",map_location=torch.device('cpu'))
        model_C.to('cpu')

       
        model_C.eval()
        print("CODE:loading_finished")

        edge_compute_time=0
        net_time=0
        cloud_time=0
        datamaker=train_based_self_detection(
            model=model_A,
            device='cpu',
            no_weight=True
        )

        #inputs_and_labels=datamaker.make_data_img()

        #input_batch=inputs_and_labels[0][0]
        
        input_batch=torch.rand(2,3,224,224)
        print("code:设备预热")
        flops,params=profile(model_A_,inputs=(input_batch,))
        model_A=torch.compile(model_A_)
        # model_A=torch.jit.script(model_A_)
        print("flops:",flops)
        print("start_warmup")
        print("边侧开始推理")
        for i in range(20):
            model_A(input_batch)
        total_time=0
        for i in range(200):
            start_time=time.perf_counter()
            op_a=model_A(input_batch)
            time1=time.perf_counter()
            total_time+=time1-start_time
        print("纯推理速度：",total_time/200)
        print(op_a.shape)
        start_time=time.perf_counter()
        output=quantisez(op_a,observer)
        end_time=time.perf_counter()
        edge_compute_time=total_time/200+end_time-start_time

        print("边侧推理1结束，开始上传数据")
        # 发送数据
        print("发送预热数据")
        warm_batch=torch.rand(1,3,224,224)
        send_data(client_socket, warm_batch)
        print("CODE:预热数据发送完成")
    
        start_time=time.perf_counter()
        send_data(client_socket, output)
        end_time=time.perf_counter()
        output=get_data(client_socket)
        
        print("传输完成，获取到云侧推理结果")
        net_time+=end_time-start_time

        start_time=time.perf_counter()

        input_data=output.dequantize()


        end_result=model_C(input_data)
        end_time=time.perf_counter()
        edge_compute_time+=end_time-start_time
        print("边侧推理结束")

        print("边侧推理时间:",edge_compute_time)
        print("网络传输时间:",net_time)
        print("总时间:",edge_compute_time+net_time+cloud_time)

        

