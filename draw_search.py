import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns
import matplotlib.pyplot as plt
# 加载 .npz 文件
data = np.load('./np_task_number_change.npy.npz')

# 通过键名访问数组
np_task_number_change = data['np_task_number_change']
np_f_change = data['np_f_change'][:90]


# 使用加载的数据

# x 表示裁剪的百分比
x = list(i for i in range(len(np_f_change)))

# 设置颜色列表
colar_ = ['blue', 'green', 'orange', 'purple']

# 创建图形和坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2= ax1.twinx()  # 创建共享 x 轴的第二个 y 轴

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
ax1.plot(x, np_task_number_change, label=f"task_number", linestyle=':', color='blue')

# 绘制实线
ax2.plot(x, np_f_change, label=f"F_score", linestyle='-', color='green')

# 绘制散点
ax1.scatter(x, np_task_number_change, marker='o', s=30, color='blue', edgecolor='black', zorder=5)
ax2.scatter(x, np_f_change, marker='x', s=30, color='green', edgecolor='black', zorder=5)

# 设置横纵坐标标签
ax1.set_xlabel('epoch', fontsize=14)
ax1.set_ylabel('task_number', fontsize=14)
ax2.set_ylabel('f_score', fontsize=14)
# 设置标题
ax1.set_title('Performance Comparison', fontsize=16)



# 添加网格，便于阅读数据
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
# 设置图例
ax1.legend(fontsize=8,loc='upper left', bbox_to_anchor=(1.05, 1))

ax2.legend(fontsize=8,loc='upper left', bbox_to_anchor=(1.05, 0.8))

# 调整布局，使标签和图例不重叠
plt.tight_layout()

# 保存为图像文件，适合论文使用
plt.savefig('search_res50.png', dpi=300)  # 设置 dpi 为 300 确保高质量输出
plt.close()