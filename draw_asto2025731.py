"""
F-Score Alpha值比较图绘制脚本

使用说明：
1. 修改MAX_ALPHA_DISPLAY来控制显示的alpha数量
   - 设为None：显示所有alpha值
   - 设为数字（如5）：只显示前5个alpha值

2. 修改SPECIFIC_ALPHAS来指定特定的alpha值
   - 设为None：使用MAX_ALPHA_DISPLAY规则
   - 设为列表（如['0.1', '0.3', '0.5']）：只显示指定的alpha值

3. 修改FIGURE_WIDTH和FIGURE_HEIGHT来调整图形尺寸
"""

import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# 设置全局字体和风格 - 符合IEEE期刊标准
plt.style.use('default')  # 使用默认样式作为基础
mpl.rcParams.update({
    'font.family': 'Times New Roman',     # IEEE标准字体
    'font.size': 12,                      # 主要字体大小
    'axes.labelsize': 14,                 # 轴标签字体大小
    'axes.titlesize': 16,                 # 标题字体大小
    'xtick.labelsize': 12,                # x轴刻度标签大小
    'ytick.labelsize': 12,                # y轴刻度标签大小
    'legend.fontsize': 11,                # 图例字体大小
    'figure.titlesize': 16,               # 图标题大小
    'lines.linewidth': 2.5,               # 线条宽度
    'lines.markersize': 8,                # 标记大小
    'axes.linewidth': 1.2,                # 坐标轴线宽
    'grid.linewidth': 0.8,                # 网格线宽
    'xtick.major.width': 1.2,             # x轴主刻度线宽
    'ytick.major.width': 1.2,             # y轴主刻度线宽
    'xtick.minor.width': 0.8,             # x轴次刻度线宽
    'ytick.minor.width': 0.8,             # y轴次刻度线宽
    'axes.edgecolor': 'black',            # 坐标轴边框颜色
    'axes.axisbelow': True,               # 网格在数据下方
    'figure.dpi': 100,                    # 显示分辨率
    'savefig.dpi': 300,                   # 保存分辨率
    'savefig.bbox': 'tight',              # 紧凑布局
    'savefig.pad_inches': 0.1,            # 边距
})

# ===================== 配置参数 =====================
# 设置要显示的alpha数量 (设为None表示显示所有alpha值)
MAX_ALPHA_DISPLAY = 5  # 可以设置为具体数字，如5、8等

# 可选：指定特定的alpha值列表 (如果设置了此项，将忽略MAX_ALPHA_DISPLAY)
SPECIFIC_ALPHAS =[0.0,0.2,0.4,0.6,0.8]  # 例如：['0.1', '0.3', '0.5', '0.7', '0.9']

# 图形尺寸设置
FIGURE_WIDTH = 10   # 图形宽度
FIGURE_HEIGHT = 6   # 图形高度
# ====================================================

# 加载数据
data = np.load('./np_task_number_change.npy.npz',allow_pickle=True)

np_task_number_change = data['np_task_number_change']
np_f_change = data['np_f_change']

# 如果np_f_change是一个包含字典的numpy数组，需要先转换
if isinstance(np_f_change, np.ndarray) and np_f_change.ndim == 0:
    # 如果是0维数组，使用item()获取内容
    np_f_change = np_f_change.item()
elif isinstance(np_f_change, np.ndarray) and np_f_change.ndim > 0:
    # 如果是多维数组，取第一个元素
    np_f_change = np_f_change[0]

# 获取所有alpha值
all_alpha_keys = list(np_f_change.keys())
print(f"All available alpha values: {all_alpha_keys}")

# 根据配置选择要显示的alpha值
if SPECIFIC_ALPHAS is not None:
    # 如果指定了特定的alpha值列表，需要转换为字符串进行匹配
    alpha_keys = [key for key in SPECIFIC_ALPHAS if key in all_alpha_keys]
    if len(alpha_keys) != len(SPECIFIC_ALPHAS):
        missing = set([str(x) for x in SPECIFIC_ALPHAS]) - set(alpha_keys)
        print(f"Warning: The following specified alphas are not found: {missing}")
elif MAX_ALPHA_DISPLAY is None:
    # 显示所有alpha值
    alpha_keys = all_alpha_keys
else:
    # 显示前N个alpha值
    alpha_keys = all_alpha_keys[:MAX_ALPHA_DISPLAY]

# 检查是否有有效的alpha值
if len(alpha_keys) == 0:
    raise ValueError("没有找到有效的alpha值！请检查SPECIFIC_ALPHAS配置或数据文件内容。")
    
print(f"Displaying alpha values: {alpha_keys} (Total: {len(alpha_keys)})")

# 对每个alpha的F-score进行0-1归一化
normalized_f_change = {}
for alpha_key in alpha_keys:
    original_data = np.array(np_f_change[alpha_key])
    
    # 计算最大值和最小值
    min_val = np.min(original_data)
    max_val = np.max(original_data)
    
    # 进行min-max归一化
    if max_val - min_val != 0:
        normalized_data = (original_data - min_val) / (max_val - min_val)
    else:
        # 如果所有值相同，归一化为0
        normalized_data = np.zeros_like(original_data)
    
    normalized_f_change[alpha_key] = normalized_data
    print(f"Alpha {alpha_key}: 原始范围 [{min_val:.4f}, {max_val:.4f}] -> 归一化范围 [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")

# 准备数据 - 使用第一个alpha的数据长度作为x轴
first_alpha_data = normalized_f_change[alpha_keys[0]]
x = np.arange(len(first_alpha_data))

# IEEE标准配色方案 - 为多条曲线准备更多颜色
colors_list = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色  
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 橄榄色
    '#17becf',  # 青色
]

# 网格和边框颜色
grid_color = '#cccccc'
edge_color = '#333333'

# 创建图形 - 使用配置的尺寸
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

# 为每个alpha值绘制曲线
legend_handles = []  # 存储图例句柄
legend_labels = []   # 存储图例标签

for i, alpha_key in enumerate(alpha_keys):
    
    alpha_data = normalized_f_change[alpha_key]  # 使用归一化后的数据
    current_x = np.arange(len(alpha_data))
    
    # 选择颜色
    color = colors_list[i % len(colors_list)]
    
    # 数据平滑处理 - 提高视觉效果
    if len(current_x) > 3:  # 确保有足够的点进行插值
        x_smooth = np.linspace(current_x.min(), current_x.max(), 300)
        
        # 使用样条插值平滑曲线
        spl_f = make_interp_spline(current_x, alpha_data, k=3)
        f_smooth = spl_f(x_smooth)
        
        # 绘制平滑曲线
        line, = ax.plot(x_smooth, f_smooth, 
                        color=color, 
                        linewidth=2.5, 
                        alpha=0.8,
                        zorder=2)
    else:
        # 如果点太少，绘制原始数据
        line, = ax.plot(current_x, alpha_data, 
                        color=color, 
                        linewidth=2.5, 
                        alpha=0.8,
                        zorder=2)
    
    # 绘制数据点 - 使用不同形状增强区分度
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker = markers[i % len(markers)]
    
    scatter = ax.scatter(current_x, alpha_data, 
                         color=color,
                         marker=marker,
                         s=60, 
                         edgecolors='white',
                         linewidth=1.5,
                         alpha=0.9,
                         zorder=3)
    
    # 创建组合图例句柄（包含线段和散点）
    from matplotlib.lines import Line2D
    combined_handle = Line2D([0], [0], 
                            color=color, 
                            linewidth=2.5,
                            marker=marker,
                            markersize=8,
                            markerfacecolor=color,
                            markeredgecolor='white',
                            markeredgewidth=1.5,
                            alpha=0.8)
    
    legend_handles.append(combined_handle)
    legend_labels.append(f'α = {alpha_key}')

# 设置坐标轴标签 - 使用专业术语，调整间距
ax.set_xlabel('Iteration Epoch', fontweight='bold', labelpad=10)
ax.set_ylabel('Normalized F-Score (0-1)', fontweight='bold', labelpad=10)

# 设置坐标轴范围和刻度
max_x_len = max([len(normalized_f_change[key]) for key in alpha_keys])
ax.set_xlim(-1, max_x_len)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# 设置y轴范围为0-1（归一化后的范围）
ax.set_ylim(-0.05, 1.05)

# 添加精细网格
ax.grid(True, linestyle='--', alpha=0.3, color=grid_color, linewidth=0.8)
ax.set_axisbelow(True)

# 创建图例 - 使用组合的图例句柄
legend = ax.legend(legend_handles, legend_labels,
                   loc='center right',
                   frameon=True,
                   fancybox=False,
                   shadow=True,
                   framealpha=0.95,
                   edgecolor='black',
                   facecolor='white',
                   borderpad=1.0,
                   columnspacing=1.0,
                   handlelength=2.0,
                   handletextpad=0.8,
                   ncol=2 if len(alpha_keys) > 5 else 1)  # 如果alpha值太多，使用两列
legend.get_frame().set_linewidth(1.2)

# 设置标题
# ax.set_title('Normalized F-Score Comparison Across Different Alpha Values', 
#               fontweight='bold', fontsize=14, pad=20)

# 优化数值标注 - 可选：标注每条曲线的最高点
# for i, alpha_key in enumerate(alpha_keys):
#     alpha_data = normalized_f_change[alpha_key]
#     max_f_idx = np.argmax(alpha_data)
#     max_f_value = alpha_data[max_f_idx]
#     ax.annotate(f'{max_f_value:.3f}',
#                 xy=(max_f_idx, max_f_value),
#                 xytext=(5, 5), textcoords='offset points',
#                 fontsize=9, 
#                 color=colors_list[i % len(colors_list)],
#                 fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存高质量图像 - 多种格式
plt.savefig('alpha_f_score_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('alpha_f_score_comparison.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('alpha_f_score_comparison.eps', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# 显示图形
plt.show()

# 清理资源
plt.close('all')