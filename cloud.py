
import torch

from splited_model import Splited_Model

device='cuda:5'

cloud=torch.load('./p_model/cloud.pth')
cloud=cloud.to(device)

from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/endpoint', methods=['POST'])
def handle_tensor():
    # 接收并解析 JSON 数据
    data = request.get_json()
    print('接受边侧设备中间数据')
    tensor_data = torch.tensor(data["data"]).to(device)  # 转换为 Tensor

    # 对 Tensor 进行操作（例如计算总和）
    # result_tensor = tensor_data.sum()
    output=cloud(tensor_data)

    # 返回结果（转为 JSON 格式）
    print("返回云侧数据")
    return jsonify({
        "result": output.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
