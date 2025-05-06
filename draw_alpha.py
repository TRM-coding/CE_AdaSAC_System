import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns
import matplotlib.pyplot as plt

# 加载 .npz 文件
data = np.load('./alpha_test.npz')

# 通过键名访问数组
flops = data['flops']
acc = data['acc']
flops = flops[1:]
acc = acc[1:]

# 排序
sorted_pairs = sorted(zip(acc, flops), key=lambda x: x[0])

# 分别拆分成两个列表
acc, flops = zip(*sorted_pairs)

# 转换为列表
acc = list(acc)
flops = list(flops)

# 使用加载的数据

# x 表示裁剪的百分比
x = list(i for i in range(len(flops)))

# 设置插值，生成 30 个新的点
spl = make_interp_spline(acc, flops, k=1)  # k=3 表示三次样条插值
new_acc_ = np.linspace(min(acc), max(acc), len(acc) + 4)  # 生成更多的 x 值
new_flops_ = spl(new_acc_)  # 计算插值后的 y 值

spl = make_interp_spline(new_acc_, new_flops_, k=5)  # k=3 表示三次样条插值
new_acc = np.linspace(min(new_acc_), max(new_acc_), len(new_acc_) + 30)  # 生成更多的 x 值
new_flops = spl(new_acc)  # 计算插值后的 y 值

# 创建图形和坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置 Seaborn 风格，调整字体大小适合论文格式
plt.rcParams.update({
    'font.family': 'serif',        # 使用衬线字体
    'font.size': 11,               # 设置字体大小
    'axes.labelsize': 11,          # 设置坐标轴标签的字体大小
    'axes.titlesize': 12,          # 图标题的字体大小
    'legend.fontsize': 8,         # 图例字体大小
    'xtick.labelsize': 8,         # x 坐标刻度字体大小
    'ytick.labelsize': 8,         # y 坐标刻度字体大小
    'lines.linewidth': 2.0,        # 设置线条宽度
    'figure.figsize': (6, 4),      # 图形的大小
})

# 绘制每个方案的曲线和散点

# 绘制虚线
ax1.plot(new_acc, new_flops, label='alpha trade off', linestyle=':', color='blue')

# 绘制插值后的曲线
# ax1.plot(new_acc, new_flops, label='Interpolated curve', linestyle='-', color='green')

# 绘制散点
ax1.scatter(new_acc, new_flops, marker='x', s=40, color='blue', edgecolor='black', zorder=5)

# 设置横纵坐标标签
ax1.set_xlabel('Loss', fontsize=14)
ax1.set_ylabel('Flops(Normaled)', fontsize=14)

# 设置标题
ax1.set_title('Performance Comparison', fontsize=16)

# 添加网格，便于阅读数据
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置图例
ax1.legend(fontsize=8, loc='upper right')

# 调整布局，使标签和图例不重叠
plt.tight_layout()

# 保存为图像文件，适合论文使用
plt.savefig('search_res50_alpha_interpolated.png', dpi=300)  # 设置 dpi 为 300 确保高质量输出
plt.close()
