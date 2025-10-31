import sys
sys.path.insert(0, './src/llama.cpp/build/ops')
import opslib as ops

print("=== 所有函数名（精确匹配）===")
for attr in dir(ops):
    print(f"  {attr}")

print("\n=== 搜索包含 'view' 的函数 ===")
for attr in dir(ops):
    if 'view' in attr.lower():
        print(f"  ✅ {attr}")

print("\n=== 搜索包含 'vtem' 的函数 ===")
for attr in dir(ops):
    if 'vtem' in attr.lower():
        print(f"  ✅ {attr}")

print("\n=== 所有以 'run_' 开头的函数 ===")
for attr in dir(ops):
    if attr.startswith('run_'):
        print(f"  ✅ {attr}")
