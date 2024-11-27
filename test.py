import copy
import time
import sys

# 构造一个约 200 MB 的数据结构
large_data = [{"list": list(range(1000000)), "dict": {i: i**2 for i in range(10000)}} for _ in range(10)]
print(f"数据大小：{sys.getsizeof(large_data) / (1024 * 1024):.2f} MB")

# 测试深拷贝时间
start = time.time()
copied_data = copy.deepcopy(large_data)
end = time.time()

print(f"深拷贝耗时：{end - start:.2f} 秒")
