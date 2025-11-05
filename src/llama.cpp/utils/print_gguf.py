from gguf.gguf_reader import GGUFReader

r = GGUFReader("./gguf_models/qwen.gguf")

# 元数据（键值表）
kv = r.fields   # 部分版本叫 r.kv_data 或 r.fields ，按实际对象属性为准
print({k: kv[k] for k in list(kv)[:30]})  # 先看前若干项

# 张量清单
print(len(r.tensors))
for t in r.tensors[:20]:
    print(t.name, t.shape, t.dtype)
