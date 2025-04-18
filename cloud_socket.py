import torch
from detection.config import CONFIG

from detection.DataGenerator import train_based_self_detection

from detection.splited_model import Splited_Model
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
import time
import requests
import flask
from flask import app, request, jsonify
import struct
import numpy as np
import base64
import socket

import pickle

app=flask.Flask(__name__)
device='cuda:3'

# model_B=model_B_.to(device)


def quantisez(tensor:torch.Tensor,observer:MovingAveragePerChannelMinMaxObserver):
    observer(tensor)
    scale, zero_point = observer.calculate_qparams()
    zero_point=torch.zeros_like(zero_point).to(device=tensor.device)
    tensor_quantized = torch.quantize_per_channel(tensor, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
    return tensor_quantized

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
    
    data=data.dequantize()

    return data


def send_data(conn, data, chunk_size=4096):
    data=data.to('cpu')
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    print("数据长度：",data_length)
    
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
    # data=request.get_json()
    model_B_=Splited_Model()
    model_B_=torch.load("./clientB_alex.pth")
    model_B_.to(device)
    model_B_.eval()
    model_B=model_B_
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (CONFIG.CLOUDIP, CONFIG.CLOUD_PORT)  # 例如使用 12345 端口
    server_socket.bind(server_address)
    server_socket.listen(5)
    print("Server is listening on port 5000...")

    print("等待网络预热数据")
    conn, addr = server_socket.accept()
    print(f"Connected to {addr}")
    # 接收数据
    data = get_data(conn)
    print("CODE:云侧握手成功")

    print("等待边侧推理数据")
    # 接收数据
    tensor_de = get_data(conn)
    print("CODE:接收到边侧推理结果")
    print("CODE:云侧开始预热")
    input_data=tensor_de.to(device)
    # model_B.to('cpu')
    op_b=None
    for i in range(10):
        op_b=model_B(input_data)
    print("CODE:云侧预热完成")
    # 处理数据
    
    input_data=tensor_de.to(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    cloud_time=0
    with torch.no_grad():
        for i in range(500):
            starter.record()
            op_b=model_B(input_data)
            ender.record()
            torch.cuda.synchronize()
            cloud_time += starter.elapsed_time(ender)
    cloud_time=cloud_time/500
    print("CODE:云侧推理结束")
    print("云侧推理时间：",cloud_time,"ms")
    print("返回中间数据")
    # 量化数据
    observer = MovingAveragePerChannelMinMaxObserver(ch_axis=0).to(device)
    output=quantisez(op_b,observer)
    start_time=time.perf_counter()
    send_data(conn, output)
    end_time=time.perf_counter()
    print("结果数据时间:",end_time-start_time)

    print("CODE:云侧返回数据完成")
    conn.close()
    
    

        

