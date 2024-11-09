import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义转换，使用 ToTensor 将 PIL 图像转换为 Tensor
transform = transforms.ToTensor()

# 加载 MNIST 数据集，使用 transform 来转换每个图像
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 使用 DataLoader 批量加载数据
train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False)

# 获取所有图像和标签
for images, labels in train_loader:
    # 将图像从 PyTorch Tensor 转换为 NumPy 数组
    train_imgs = images.numpy()  # 形状是 (60000, 1, 28, 28)

# 查看转换后的 NumPy 数组形状
print(f"Train images shape: {train_imgs.shape}")