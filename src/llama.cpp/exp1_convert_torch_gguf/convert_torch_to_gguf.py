import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf

class SVD_MUL_MAT(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, C):
        # C 由外部调用者传进来，可以是标量，也可以是 batch-wise / token-wise 的张量
        return F.linear(x, self.weight, self.bias) * C
    

import torch
from gguf import GGUFWriter, GGMLQuantizationType

def main():
    # 1. 构建 PyTorch 模块，并加载你实际训练好的权重
    in_features = 16
    out_features = 32
    model = SVD_MUL_MAT(in_features, out_features)


    model.eval()

    # 2. 创建 GGUFWriter
    #
    #   - out_path: 输出文件路径
    #   - arch_name: 任意字符串，llama.cpp 内部用来区分模型架构
    #                你可以先随便起一个，比如 "my_custom_op"
    out_path = "my_custom_op.gguf"
    writer = GGUFWriter(out_path, "my_custom_op")

    # 3. 一些可选的元数据（你可以先不写，后面再精细化）
    writer.add_name("my_custom_op_only")
    writer.add_description("GGUF file that contains only the weights of MyCustomOp")
    writer.add_author("your_name")  # 随便填

    # 4. 把 state_dict 里的张量逐个写入 GGUF
    #    命名规则很关键：之后你在 llama.cpp 里要用同样的名字去找这些 tensor。
    state_dict = model.state_dict()

    for name, tensor in state_dict.items():
        # 确保是 CPU + float32（gguf 也支持量化，这里先用最简单情况）
        t = tensor.detach().cpu().to(torch.float32).numpy()

        # 设定一个 GGUF 里的名字，比如 "myop.weight"、"myop.bias"
        # 你也可以直接用 state_dict 的名字：
        # gguf_name = f"myop.{name}"
        gguf_name = f"myop.{name}"

        # 写入张量；先不量化，保持 F32
        writer.add_tensor(
            name=gguf_name,
            tensor=t
        )

    # 5. 写出 GGUF 文件
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    print(f"saved GGUF to: {out_path}")

if __name__ == "__main__":
    main()