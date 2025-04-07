import torch
from detection.splited_model import Splited_Model
import time
from thop import profile
if __name__ == "__main__":
    # 测试数据
    data = torch.randn(1, 3, 224, 224)  # 示例数据
    print("CODE:测试数据生成")
    with torch.no_grad():     
        print("CODE:loading_model")
        model_A=Splited_Model()
        model_A_=torch.load("./clientA_v.pth",map_location=torch.device('cpu'))
        model_A_.to('cpu')
        model_A_.eval()
        input_batch=torch.rand(10,3,224,224)
        # model_A=torch.compile(model_A_)
        model_A=model_A_
        print(model_A)
        flops, param = profile(model_A, inputs=(input_batch,))
        print("CODE:模型计算量:",flops)
        print("CODE:模型参数量:",param)
        print("CODE:模型加载完成")
        print("CODE:开始测试")
        time_=0
        for i in range(5):
            sp=time.perf_counter()
            op_a=model_A(input_batch)
            end=time.perf_counter()
            time_+=end-sp
        print("CODE:模型测试完成")
        print("CODE:模型测试时间:",time_/50)