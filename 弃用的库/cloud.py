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

import pickle

app=flask.Flask(__name__)
device='cuda:3'
model_B=Splited_Model()
model_B=torch.load("./clientB.pth")
model_B.to(device)
model_B.eval()


def quantisez(tensor:torch.Tensor,observer:MovingAveragePerChannelMinMaxObserver):
    observer(tensor)
    scale, zero_point = observer.calculate_qparams()
    tensor_quantized = torch.quantize_per_channel(tensor, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
    return tensor_quantized


def dequantise(numpy_data,scale,zero_point):
    tensor=torch.from_numpy(numpy_data)
    zero=torch.tensor(zero_point)
    scale=torch.tensor(scale)
    zero=zero.reshape(-1,1,1,1)
    scale=scale.reshape(-1,1,1,1)
    detensor=(tensor.to(torch.float32)-zero)*scale
    return detensor


@app.route('/cloud', methods=['POST', 'GET'])
def shaking():
    # data=request.get_json()
    print("CODE:云侧握手")
    print("CODE:loading_model")
    return jsonify({"status":"ok"})

@app.route('/inff_warm', methods=['POST'])
def inff():
    print("云侧收到预热数据")
    return jsonify({"status":"ok"})

@app.route('/inff', methods=['POST'])
def iff():
    data=request.get_json()
    print("CODE:接收到边侧推理结果")
    print("CODE:云侧开始推理")

    

    start_time=time.time()
    tensor=data['data']
    tensor_de=pickle.loads(tensor)
    input_data=tensor_de.to(device)
    op_b=model_B(input_data)
    end_time=time.time()
    cloud_time=end_time-start_time
    print("CODE:云侧推理结束")
    
    output=quantisez(op_b)
    data = {
        "data": pickle.dumps(output),
        "cloud_time": cloud_time
    }

    return jsonify(data)
if __name__ == "__main__":

    app.run(port=CONFIG.CLOUD_PORT,host='0.0.0.0')
    

        

