import torch

ts_tensor=torch.randn(2048,4096,1,1)

if ts_tensor.ndim != 2:
        ts_tensor = ts_tensor.view(ts_tensor.shape[0], -1)

U,S,V=torch.linalg.svd(ts_tensor,full_matrices=False)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("V shape:", V.shape)