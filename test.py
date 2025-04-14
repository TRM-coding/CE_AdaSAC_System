# import torch
# import torch.nn.functional as F
# import time
# # 参数设置（以 GPT-3 小模型为例）
# batch_size = 2
# seq_length = 1024
# hidden_dim = 12288
# num_heads = 12
# head_dim = hidden_dim // num_heads  # 64

# # 随机生成输入张量: shape (batch_size, seq_length, hidden_dim)
# x = torch.randn(batch_size, seq_length, hidden_dim)
# # 随机生成投影权重
# W_q = torch.randn(hidden_dim, hidden_dim)
# W_k = torch.randn(hidden_dim, hidden_dim)
# W_v=torch.randn(hidden_dim, hidden_dim)

# st_time=time.perf_counter()
# # 计算 Q, K, V，形状均为 (batch_size, seq_length, hidden_dim)
# x=x.to('cuda:3')
# W_q=W_q.to('cuda:3')
# W_k=W_k.to('cuda:3')
# W_v=W_v.to('cuda:3')
# for i in range(20):                             
#     Q = x @ W_q
#     K = x @ W_k
#     V=x@W_v
#     attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (head_dim ** 0.5)
# V = V.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
# V = V.reshape(batch_size * num_heads, seq_length, head_dim)

# # 将 Q, K, V reshape 成 (batch_size, seq_length, num_heads, head_dim)
# # 然后转置成 (batch_size, num_heads, seq_length, head_dim)
# Q = Q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
# K = K.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

# # 将 batch 和 head 维度合并，变成 (batch_size * num_heads, seq_length, head_dim)，便于使用 torch.bmm
# Q = Q.reshape(batch_size * num_heads, seq_length, head_dim)
# K = K.reshape(batch_size * num_heads, seq_length, head_dim)


# # 利用 bmm 计算注意力分数: (batch_size * num_heads, seq_length, seq_length)
# attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (head_dim ** 0.5)

# # 对注意力分数做 softmax，得到注意力权重
# attn_probs = F.softmax(attn_scores, dim=-1)
# ed_time=time.perf_counter()
# output=torch.bmm(attn_probs, V)
# print("Time taken:", ed_time-st_time)

# print("Attention output shape:", attn_probs.shape)


# # import torch
# # import torch.nn.functional as F
# # from torch import nn
# # import time

# # # 参数设置（以 GPT-2 小模型为例）
# # batch_size = 1
# # seq_length = 1024
# # hidden_dim = 12288
# # num_heads = 12
# # head_dim = hidden_dim // num_heads  # 64

# # # 随机生成输入张量: shape (batch_size, seq_length, hidden_dim)
# # x = torch.randn(batch_size, seq_length, hidden_dim)
# # attn_probs = torch.randn(batch_size*num_heads, seq_length, seq_length)
# # W_v = torch.randn(hidden_dim, hidden_dim)

# # st_time=time.perf_counter()

# # V = x @ W_v
# # V = V.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
# # V = V.reshape(batch_size * num_heads, seq_length, head_dim)

# # # attn_output = torch.bmm(attn_probs, V)
# # # attn_output = attn_output.view(batch_size, num_heads, seq_length, head_dim)
# # # attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, hidden_dim)
# # ed_time=time.perf_counter()
# # print("计算时间:",ed_time-st_time)
# # print(V.shape)



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2",cache_dir='./gpt2')
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2",cache_dir='./gpt2')

print(list(model.named_children()))



