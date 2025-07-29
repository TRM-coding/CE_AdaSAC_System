# 首先确保已经安装了 thop
# pip install thop

import torch
from torchvision import models
from thop import profile

# 加载预训练模型
model = models.alexnet(weights=True)
model.eval()

# 构造一个大小为 (1,3,224,224) 的随机输入
input_tensor = torch.randn(1, 3, 224, 224)

# 计算 MACs 和 参数量
macs, params = profile(model, inputs=(input_tensor,), verbose=False)

# 将 MACs 转为 FLOPs
flops = 2 * macs

# 输出结果
print(f"ResNet‑50 MACs: {macs:,}")
print(f"ResNet‑50 Params: {params:,}")
print(f"Estimated FLOPs: {flops:,}")

# 如果想看 GFLOPs，可除以 1e9
print(f"Estimated GFLOPs: {flops/1e9} GFLOPs")
