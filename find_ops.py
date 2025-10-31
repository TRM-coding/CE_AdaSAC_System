import sys
import os

# 尝试不同的路径
paths_to_try = [
    '../build/ops',
    './build/ops', 
    '../3dparty/build/ops',
    './3dparty/build/ops',
    '../exp1/build/ops',
    './exp1/build/ops',
    '../../build/ops'
]

print("🔍 搜索 opslib 模块...")
for path in paths_to_try:
    full_path = os.path.abspath(path)
    if os.path.exists(path):
        print(f"✅ 路径存在: {path}")
        sys.path.insert(0, path)
        try:
            import opslib as ops
            print(f"🎉 成功导入 opslib from: {path}")
            break
        except ImportError as e:
            print(f"❌ 导入失败 from {path}: {e}")
    else:
        print(f"❌ 路径不存在: {path}")

# 列出当前 Python 路径
print("\n📁 当前 Python 路径:")
for p in sys.path:
    print(f"  {p}")
