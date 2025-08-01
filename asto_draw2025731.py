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

# 加载数据
data = np.load('./np_task_number_change.npy.npz')
np_task_number_change = data['np_task_number_change']
np_f_change = data['np_f_change'][:90]

# 准备数据
x = np.arange(len(np_f_change))

# IEEE标准配色方案 - 高对比度且色盲友好
colors = {
    'task_number': '#1f77b4',    # 深蓝色
    'f_score': '#ff7f0e',        # 橙色
    'grid': '#cccccc',           # 浅灰色网格
    'edge': '#333333'            # 深灰色边框
}

# 创建图形 - 调整尺寸以容纳图例和标注
fig, ax1 = plt.subplots(figsize=(10, 5))

# 创建第二个y轴
ax2 = ax1.twinx()

# 数据平滑处理 - 提高视觉效果
if len(x) > 3:  # 确保有足够的点进行插值
    x_smooth = np.linspace(x.min(), x.max(), 300)
    
    # 使用样条插值平滑曲线
    spl_task = make_interp_spline(x, np_task_number_change, k=3)
    task_smooth = spl_task(x_smooth)
    
    spl_f = make_interp_spline(x, np_f_change, k=3)
    f_smooth = spl_f(x_smooth)
    
    # 只绘制F-Score平滑曲线
    line2 = ax2.plot(x_smooth, f_smooth, 
                     color=colors['f_score'], 
                     linewidth=2.5, 
                     alpha=0.8,
                     label='F-Score',
                     zorder=2)
else:
    # 如果点太少，只绘制F-Score原始数据
    line2 = ax2.plot(x, np_f_change, 
                     color=colors['f_score'], 
                     linewidth=2.5, 
                     alpha=0.8,
                     label='F-Score',
                     zorder=2)

# 绘制数据点 - 使用不同形状增强区分度
scatter2 = ax2.scatter(x, np_f_change, 
                      color=colors['f_score'],
                      marker='s',  # 方形标记
                      s=60, 
                      edgecolors='white',
                      linewidth=1.5,
                      alpha=0.9,
                      zorder=3)

# 设置坐标轴标签 - 使用专业术语，调整间距
ax1.set_xlabel('Training Epoch', fontweight='bold', labelpad=10)
ax2.set_ylabel('F-Score', fontweight='bold', color=colors['f_score'], labelpad=10)

# 设置坐标轴颜色
ax2.tick_params(axis='y', labelcolor=colors['f_score'])

# 设置坐标轴范围和刻度
ax1.set_xlim(-1, len(x)+1)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

# 动态调整y轴范围，避免与图例重叠
f_range = np.max(np_f_change) - np.min(np_f_change)
ax2.set_ylim(np.min(np_f_change) - f_range*0.05, 
             np.max(np_f_change) + f_range*0.1)

# 添加精细网格
ax1.grid(True, linestyle='--', alpha=0.3, color=colors['grid'], linewidth=0.8)
ax1.set_axisbelow(True)

# 创建组合图例 - 优化位置避免遮挡
lines2, labels2 = ax2.get_legend_handles_labels()
# 只显示F-Score图例
legend = ax2.legend(lines2, labels2, 
                   loc='upper right',
                   frameon=True,
                   fancybox=False,
                   shadow=True,
                   framealpha=0.95,
                   edgecolor='black',
                   facecolor='white',
                   borderpad=1.0,
                   columnspacing=1.0,
                   handlelength=2.0,
                   handletextpad=0.8)
legend.get_frame().set_linewidth(1.2)

# 设置标题 - 更专业的表述，减小字体避免重叠
ax1.set_title('Training Performance Analysis: Task Number vs F-Score Evolution', 
              fontweight='bold', fontsize=14, pad=25)

# 优化数值标注位置 - 避免与图例重叠
if len(x) > 0:
    # 标注最高F-Score点，调整位置避免与图例冲突
    max_f_idx = np.argmax(np_f_change)
    
    # 根据最高点位置智能选择标注方向
    if x[max_f_idx] > len(x) * 0.7:  # 如果在右侧，向左标注
        xytext = (-60, -30)
        ha = 'right'
    else:  # 如果在左侧，向右标注
        xytext = (30, 20)
        ha = 'left'
    
    ax2.annotate(f'Peak F-Score: {np_f_change[max_f_idx]:.3f}',
                xy=(x[max_f_idx], np_f_change[max_f_idx]),
                xytext=xytext, textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                         alpha=0.8, edgecolor='navy'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                               color='navy', lw=1.5),
                fontsize=11, fontweight='bold', ha=ha)

# 调整布局
plt.tight_layout()

# 保存高质量图像 - 多种格式
plt.savefig('search_res50_infocom.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('search_res50_infocom.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('search_res50_infocom.eps', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# 显示图形
plt.show()

# 清理资源
plt.close('all')