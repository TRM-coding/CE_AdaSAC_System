import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "./models/qwen2_5_1_5b"

tok = AutoTokenizer.from_pretrained(name, use_fast=False)
m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, device_map=None).to("cpu").eval()

enc = tok("Hello world", return_tensors="pt")
input_ids = enc["input_ids"]          # [B, T]
attention_mask = enc.get("attention_mask")

def fwd():
    return m(input_ids=input_ids, attention_mask=attention_mask)

# 用 input_data 显式传入字典
summary(m, input_data=dict(input_ids=input_ids, attention_mask=attention_mask),
        depth=6,  # 调整打印深度
        col_names=("input_size","output_size","num_params","kernel_size","mult_adds"))
