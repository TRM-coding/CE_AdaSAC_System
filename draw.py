import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns
import matplotlib.pyplot as plt
# 加载 .npz 文件
data = np.load('./data_arrays_vgg.npz')

# 通过键名访问数组
maked_acc = data['maked_acc']
maked_loss = data['maked_loss']
img_acc    = data['img_acc']
img_loss   = data['img_loss']

# 使用加载的数据

# x 表示裁剪的百分比
x = list(i for i in range(len(maked_acc[0])))

# 设置颜色列表
colar_ = ['blue', 'green', 'orange', 'purple']

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
for i, y_values in enumerate(maked_acc[:]):
    # 绘制虚线
    ax1.plot(x, y_values, label=f"Scheme {i+1}_forecast", linestyle=':', color=colar_[i])

    # 绘制实线
    ax1.plot(x, img_acc[i] / 100 / (max(img_acc[i]) / 100), label=f"Scheme {i+1}_IMG", linestyle='-', color=colar_[i])

    # 绘制散点
    ax1.scatter(x, y_values, marker='o', s=30, color=colar_[i], edgecolor='black', zorder=5)
    ax1.scatter(x, img_acc[i] / 100 / (max(img_acc[i]) / 100), marker='o', s=30, color=colar_[i], edgecolor='black', zorder=5)

# 设置横纵坐标标签
ax1.set_xlabel('Clipping Percentage (%)', fontsize=14)
ax1.set_ylabel('Accuracy (%)', fontsize=14)

# 设置标题
ax1.set_title('Performance Comparison', fontsize=16)

# 添加网格，便于阅读数据
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置图例
ax1.legend(fontsize=8,loc='upper left', bbox_to_anchor=(1.05, 1))

# 调整布局，使标签和图例不重叠
plt.tight_layout()

# 保存为图像文件，适合论文使用
plt.savefig('scatter_plot.png', dpi=300)  # 设置 dpi 为 300 确保高质量输出
plt.close()