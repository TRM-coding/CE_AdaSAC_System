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



def quantisez(tensor:torch.Tensor,observer:MovingAveragePerChannelMinMaxObserver):
    observer(tensor)
    scale, zero_point = observer.calculate_qparams()
    tensor_quantized = torch.quantize_per_channel(tensor, scales=scale, zero_points=zero_point, axis=0, dtype=torch.qint8)
    numpy_data=tensor_quantized.int_repr().numpy()
    scale=scale.numpy()
    zero_point=zero_point.numpy()
    shape=list(tensor.shape)
    return numpy_data,scale,zero_point,shape

def dequantise(numpy_data,scale,zero_point):
    tensor=torch.from_numpy(numpy_data)
    zero=torch.tensor(zero_point)
    scale=torch.tensor(scale)
    zero=zero.reshape(-1,1)
    scale=scale.reshape(-1,1)
    detensor=(tensor.to(torch.float32)-zero)*scale
    return detensor

headers = {
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"  # 示例认证头
}


if __name__ == "__main__":
    print("CODE:云侧握手")
    client = httpx.Client(http2=True)
    client.get(f'{CONFIG.CLOUDIP}/cloud',headers=headers)    # 预热
    resp = client.get(f'{CONFIG.CLOUDIP}/cloud', params={"q": "peak"}, headers=headers)
    print("CODE:云侧握手完成")
    print("CODE:loading_model")
    model_A=Splited_Model()
    model_C=Splited_Model()
    observer = MovingAveragePerChannelMinMaxObserver(ch_axis=0)


    model_A=torch.load("./clientA.pth")
    model_A.to('cpu')
    model_A.eval()
    model_C=torch.load("./clientC.pth")
    model_C.to('cpu')
    model_A.eval()
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

    inputs_and_labels=datamaker.make_data_img()

    input_batch=inputs_and_labels[0][0]
    
    print("边侧开始推理")
    start_time=time.time()
    op_a=model_A(input_batch)
    #time1=time.time()
    #print("纯推理速度：",time1-start_time)
    #print(op_a.shape)
    output=quantisez(op_a,observer)
    data={
        "data": base64.b64encode(output[0].tobytes()).decode("utf-8"),
        "scale":base64.b64encode(output[1].tobytes()).decode("utf-8"),
        "zero_point":base64.b64encode(output[2].tobytes()).decode("utf-8"),
        "shape":base64.b64encode(np.array(output[3]).tobytes()).decode("utf-8")
    }
    
    end_time=time.time()
    edge_compute_time=end_time-start_time
    print("边侧推理1结束，开始上传数据")
    start_time=time.time()

    response=client.post(f'{CONFIG.CLOUDIP}/inff',json=data,headers=headers,timeout=CONFIG.TIMEOUT)
    end_time=time.time()
    print("传输完成，获取到云侧推理结果")
    cloud_time=response.json()["cloud_time"]
    net_time=end_time-start_time-cloud_time
    
    data=response.json()


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

    input_data=dequantise(decoded_tensor,decoded_scale,decoded_zero)


    end_result=model_C(input_data)
    end_time=time.time()
    edge_compute_time+=end_time-start_time
    print("边侧推理结束")

    print("边侧推理时间:",edge_compute_time)
    print("网络传输时间:",net_time)
    print("云侧推理时间:",cloud_time)
    print("总时间:",edge_compute_time+net_time+cloud_time)

        

