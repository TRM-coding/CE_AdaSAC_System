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

app=flask.Flask(__name__)
device='cuda:3'
model_B=Splited_Model()
model_B=torch.load("./clientB.pth")
model_B.to(device)
model_B.eval()


def quantisez(tensor:torch.Tensor,observer:MovingAveragePerChannelMinMaxObserver):
    observer=observer.to(device)
    observer(tensor)
    tensor.to(device)
    scale, zero_point = observer.calculate_qparams()
    tensor_quantized = torch.quantize_per_channel(tensor, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
    numpy_data=tensor_quantized.int_repr().cpu().numpy()
    shape=list(tensor.shape)
    return numpy_data,scale.cpu(),zero_point.cpu(),shape

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

@app.route('/inff', methods=['POST', 'GET'])
def iff():
    data=request.get_json()
    print("CODE:接收到边侧推理结果")
    print("CODE:云侧开始推理")

    tensor=data['data']
    decoded_tensor = base64.b64decode(tensor)
    scale=data['scale']
    decoded_scale = base64.b64decode(scale)
    zero_point=data['zero_point']
    decoded_zero = base64.b64decode(zero_point)
    shape=data['shape']
    decoded_shape = base64.b64decode(shape)
    decoded_shape=np.frombuffer(decoded_shape,dtype=np.int64)
    decoded_tensor=np.frombuffer(decoded_tensor,dtype=np.int8).reshape(decoded_shape)
    decoded_scale=np.frombuffer(decoded_scale,dtype=np.float32)
    decoded_zero=np.frombuffer(decoded_zero,dtype=np.int32)

    start_time=time.time()
    tensor=dequantise(decoded_tensor,decoded_scale,decoded_zero)
    input_data=tensor.to(device)
    op_b=model_B(input_data)
    end_time=time.time()
    cloud_time=end_time-start_time
    print("CODE:云侧推理结束")
    
    output=quantisez(op_b,MovingAveragePerChannelMinMaxObserver(ch_axis=0))
    data={
        "data": base64.b64encode(output[0].tobytes()).decode("utf-8"),
        "scale":base64.b64encode(np.array(output[1]).tobytes()).decode("utf-8"),
        "zero_point":base64.b64encode(np.array(output[2]).tobytes()).decode("utf-8"),
        "shape":base64.b64encode(np.array(output[3]).tobytes()).decode("utf-8"),
        "cloud_time":cloud_time
    }

    

    return jsonify(data)
if __name__ == "__main__":

    app.run(port=CONFIG.CLOUD_PORT,host='0.0.0.0')
    

        

