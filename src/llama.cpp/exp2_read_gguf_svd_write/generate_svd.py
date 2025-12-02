from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf import GGUFType
from gguf import GGMLQuantizationType   # 如果你要显式指定 F32 等
import torch
from gguf import GGUFWriter, GGUFValueType, ReaderField, OrderedDict
import numpy as np

def copy_metadata(reader, writer):
    for keyi, field in reader.fields.items():
        # 这几个是由 writer 自己负责写的，通常跳过
        if keyi in ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]:
            continue

        key=keyi
        if 'rope' not in keyi:
            key=keyi.replace("qwen2", "qwen2_svd")

        # field.types 是一个列表，取 [0] 即字段类型
        ftype = field.types[0]

        if ftype == GGUFValueType.STRING:
            # STRING 的值在 parts[field.data[0]] 里，是一串 int，需要转回字符
            s = ''.join(chr(i) for i in field.parts[field.data[0]])
            writer.add_string(key=key, val=s)

        elif ftype == GGUFValueType.ARRAY:
            # ARRAY 直接用 contents()
            writer.add_array(key=key, val=field.contents())

        else:
            # 其他标量类型：INT / FLOAT / BOOL ...
            v = field.parts[field.data[0]][0]
            writer.add_key_value(key=key, val=v, vtype=ftype)

gguf_path = "/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf"

reader = GGUFReader(gguf_path)
writer = GGUFWriter(gguf_path + ".svd"+".gguf","qwen2_svd")
copy_metadata(reader, writer)

need_svd = ['ffn_up', 'ffn_down']


def convert_to_torch(gguf_tensor):
    # 读出 numpy 一维数组（不转置）
    W_np = np.array(gguf_tensor.data, copy=False).astype("float32", copy=False)

    # 按 GGUF 的维度 reshape （C-order）
    W_np = W_np.reshape(gguf_tensor.shape)

    return torch.from_numpy(W_np)

def svd_factorize_torch(torch_tensor: torch.Tensor, device='cuda:5'):
    # device = "cuda" 或 "cpu"
    if device is None:
        device = torch_tensor.device  # 默认跟随输入

    # 确保在 2D
    if torch_tensor.ndim != 2:
        torch_tensor = torch_tensor.view(torch_tensor.shape[0], -1)

    # 移到 GPU
    torch_tensor = torch_tensor.to(device)

    # SVD on GPU
    U, S, Vh = torch.linalg.svd(torch_tensor, full_matrices=False)

    s_sqrt = torch.sqrt(S)

    U_factor = U * s_sqrt.unsqueeze(0)
    V_factor = s_sqrt.unsqueeze(1) * Vh

    return U_factor.cpu(), V_factor.cpu()


# 先复制 KV（下一节详细），再写 tensor
# 这里先只演示 tensor 部分
import re
for t in reader.tensors:
    name = t.name
    do_svd = any(sub in name for sub in need_svd)

    if do_svd:
        torch_tensor = convert_to_torch(t)
        U, V = svd_factorize_torch(torch_tensor)
        out1=None
        out2=None
        if "ffn_down" in name:
            out1 = re.sub(r'ffn_down', r'ffn_down_svd_u', name)
            out2 = re.sub(r'ffn_down', r'ffn_down_svd_v', name)
        if "ffn_up" in name:
            out1 = re.sub(r'ffn_up', r'ffn_up_svd_u', name)
            out2 = re.sub(r'ffn_up', r'ffn_up_svd_v', name)
        writer.add_tensor(
            name=out1,
            tensor=U.detach().cpu().numpy().astype("float32"),
            raw_shape=tuple(U.shape)[::-1],
            raw_dtype=GGMLQuantizationType.F32,  # 新的 U/V 用 F32
        )
        writer.add_tensor(
            name=out2,
            tensor=V.detach().cpu().numpy().astype("float32"),
            raw_shape=tuple(V.shape)[::-1],
            raw_dtype=GGMLQuantizationType.F32,
        )
    
    # 未改动张量：按原始信息写回原始张量
    writer.add_tensor(
        name=name,
        tensor=t.data,              # 原样写回
        raw_shape=tuple(t.shape)[::-1], # 原始形状
        raw_dtype=t.tensor_type,  # 原始量化类型/精度
    )

# 写文件
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
