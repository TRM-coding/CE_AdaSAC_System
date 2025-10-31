import sys
sys.path.insert(0, './src/llama.cpp/build/ops')
import opslib as ops

print("=== 所有可用函数 ===")
for attr in sorted(dir(ops)):
    if not attr.startswith('__'):
        print(f"  {attr}")

print("\n=== 可调用函数 ===")
for attr in sorted(dir(ops)):
    obj = getattr(ops, attr)
    if callable(obj) and not attr.startswith('__'):
        print(f"  {attr}")
