import sys
sys.path.insert(0, './src/llama.cpp/build/ops')
import opslib as ops

print("🔍 查找所有包含 'view' 的函数:")
for attr in dir(ops):
    if 'view' in attr.lower():
        print(f"  - {attr}")

print("\n🔍 所有可用的函数:")
view_funcs = [f for f in dir(ops) if f.startswith('run_')]
for func in view_funcs:
    print(f"  - {func}")
