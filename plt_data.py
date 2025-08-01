import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# 设置学术风格

# 真实数据
data = {
    'Method': ['auto-split', 'NEURO.CLOUD', 'QDMP.CLOUD', 'CE_AdaSAC'],
    'alexnet_Gflops_edge': [0, 0.0465, 0.0713, 0.19],
    'alexnet_cloud_throughtput': [0, 12.3993329158891,  16.5980531976491,32.29118902616232],
    'vgg16_Gflops_edge': [0, 1.62, 1.953,  2.400185],
    'vgg16_cloud_throughtput': [9.77010932752338, 15.8378207158695,  16.3737484252231,51.3347],
    'resnet50_G_flops_edge': [0, 0.372, 0.42129,  1.323],
    'resnet50_cloud_throughtput': [8.33, 16.3150767624362,   16.5451112755471,45.81271761040865]
}

df = pd.DataFrame(data)

# 模型列表和方法
models = ['AlexNet', 'VGG-16', 'ResNet-50']  # 更规范的命名
methods = ['Auto-Split', 'NEURO.CLOUD', 'QDMP', 'CE_AdaSAC']  # 匹配数据中的实际方法数量

# 设置matplotlib参数，符合学术论文要求
plt.rcParams.update({
    'font.size': 20,  # 增大基础字体大小
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 22,  # 增大轴标签字体
    'axes.titlesize': 24,  # 增大标题字体
    'xtick.labelsize': 18,  # 增大x轴刻度标签
    'ytick.labelsize': 18,  # 增大y轴刻度标签
    'legend.fontsize': 18,  # 增大图例字体
    'figure.figsize': (16, 8),  # 稍微增大图表尺寸
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# 数据处理
edge_data = np.array([
    df['alexnet_Gflops_edge'].values,
    df['vgg16_Gflops_edge'].values, 
    df['resnet50_G_flops_edge'].values
])

throughput_data = np.array([
    df['alexnet_cloud_throughtput'].values,
    df['vgg16_cloud_throughtput'].values,
    df['resnet50_cloud_throughtput'].values
])

# 处理特殊值（inf和None）
for i in range(throughput_data.shape[0]):
    for j in range(throughput_data.shape[1]):
        if throughput_data[i, j] is None:
            throughput_data[i, j] = 0
        elif np.isinf(throughput_data[i, j]):
            throughput_data[i, j] = 100  # 用一个很大的值代替inf，便于绘图

# 创建图形
fig, ax1 = plt.subplots(figsize=(16, 8))

# 设置颜色方案 - 使用更专业的配色
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592720', '#1B5E20']
patterns = ['', '///', '...', '+++', 'xxx', '|||']

# 柱状图参数
n_models = len(models)
n_methods = len(methods)
width = 0.12  # 每个柱子的宽度
spacing = 0.02  # 柱子间距

# 计算每组柱子的位置
x_positions = np.arange(n_models)
bar_positions = []

for i in range(n_methods):
    offset = (i - (n_methods-1)/2) * (width + spacing)
    bar_positions.append(x_positions + offset)

# 绘制吞吐量柱状图 (左Y轴)
bars_throughput = []
for i, method in enumerate(methods):
    bars = ax1.bar(bar_positions[i], throughput_data[:, i], width, 
                   label=method, color=colors[i], alpha=0.8, 
                   edgecolor='black', linewidth=0.8, hatch=patterns[i])
    bars_throughput.append(bars)

# 设置左Y轴（吞吐量）
ax1.set_xlabel('Neural Network Models', fontsize=24, fontweight='bold')
ax1.set_ylabel('Cloud Throughput (fps)', fontsize=24, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(models, fontsize=20, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 创建右Y轴（边侧计算压力）
ax2 = ax1.twinx()

# 绘制边侧计算压力的散点图（只有点，没有连线）
for i, method in enumerate(methods):
    if i < len(colors):  # 确保不超出颜色数组范围
        ax2.scatter(bar_positions[i], edge_data[:, i], 
                   s=120, color=colors[i], marker='o',
                   facecolors='white', edgecolors=colors[i], 
                   linewidths=3, alpha=0.9, zorder=5)

# 设置右Y轴
ax2.set_ylabel('Edge Computing Load (GFLOPs)', fontsize=24, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=18)

# 设置Y轴范围，确保数据清晰可见
ax1.set_ylim(0, max(throughput_data.flatten()) * 1.1)
ax2.set_ylim(0, max(edge_data.flatten()) * 1.2)

# 添加图例 - 合并两种样式到一个图例中
legend_elements = []
for i, method in enumerate(methods):
    # 创建复合图例元素：包含柱状图和散点图样式
    bar_patch = mpatches.Patch(color=colors[i], alpha=0.8, 
                              edgecolor='black', linewidth=0.8,
                              hatch=patterns[i], label=method)
    legend_elements.append(bar_patch)

# 创建主图例
main_legend = ax1.legend(handles=legend_elements, 
                        loc='upper left', bbox_to_anchor=(0.02, 0.99),
                        title='Methods', title_fontsize=16, fontsize=16,
                        frameon=True, fancybox=True, shadow=True,
                        ncol=2)
main_legend.get_frame().set_facecolor('white')
main_legend.get_frame().set_alpha(0.9)

# 添加说明文字
ax1.text(0.08, 0.75, 'Bars: Cloud Throughput\nDots: Edge Computing Load', 
         transform=ax1.transAxes, fontsize=16, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
         verticalalignment='top')

# 添加数值标签
def add_value_labels(ax, bars_list, data, format_str='{:.1f}'):
    """在柱状图上添加数值标签"""
    for i, bars in enumerate(bars_list):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # 只为有值的柱子添加标签
                value = data[j, i]
                if np.isinf(value):
                    label = '∞'
                elif value == 0:
                    label = '0'
                else:
                    label = format_str.format(value)
                
                # 所有标签都显示在柱子上方3个像素的位置
                # 特殊处理：VGG-16的第三个柱子（QDMP方法）显示在下方
                if j == 1 and i == 2:  # j=1是VGG-16，i=2是QDMP方法
                    y_offset = -5  # 负值表示向下偏移
                    va_pos = 'top'  # 标签顶部对齐到柱子底部
                else:
                    # 其他标签都显示在柱子上方
                    y_offset = 5
                    va_pos = 'bottom'
                
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, y_offset),  # 固定在柱子上方3个像素
                           textcoords="offset points",
                           ha='center', va=va_pos,
                           fontsize=14, fontweight='bold',  # 增大数值标签字体
                           rotation=90 if len(label) > 4 else 0,
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.8, edgecolor='none'))

# 添加吞吐量数值标签
add_value_labels(ax1, bars_throughput, throughput_data, '{:.1f}')

# 添加边侧计算压力数值标签
for i, method in enumerate(methods):
    for j, pos in enumerate(bar_positions[i]):
        value = edge_data[j, i]
        if value > 0:
            # 根据数值大小和位置调整标签位置
            if j == 1 and i == 1:  # VGG-16的NEURO.CLOUD，避免与柱子标签重叠
                x_offset = 5
                y_offset = -25
                ha_pos = 'right'
            elif j == 1 and i == 2:  # VGG-16的QDMP.CLOUD
                x_offset = 5
                y_offset = 10
                ha_pos = 'left'
            elif j == 1 and i == 3:  # VGG-16的CE_AdaSAC，数值较大
                x_offset = 5
                y_offset = 15
                ha_pos = 'left'
            elif value > 2:  # 高数值点
                x_offset = 8
                y_offset = 8
                ha_pos = 'left'
            else:  # 其他情况
                x_offset = 5
                y_offset = 5
                ha_pos = 'left'
            
            ax2.annotate(f'{value:.3f}',
                        xy=(pos, value),
                        xytext=(x_offset, y_offset),
                        textcoords="offset points",
                        ha=ha_pos, va='bottom',
                        fontsize=12, fontweight='bold',  # 增大边侧计算压力标签字体
                        bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='lightblue', alpha=0.8, 
                                edgecolor='darkblue', linewidth=0.5))

# 设置标题
plt.title('Performance Comparison: Cloud Throughput vs Edge Computing Load\nAcross Different Neural Network Architectures', 
          fontsize=26, fontweight='bold', pad=25)  # 增大标题字体

# 调整布局
plt.tight_layout()

# 保存高质量图片
plt.savefig('overhead_througput.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('overhead_througput.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# 显示图片
plt.show()

# 打印数据摘要
print("=== 数据摘要 ===")
print(f"模型数量: {n_models}")
print(f"方法数量: {n_methods}")
print("\n各方法在不同模型上的表现:")
for i, model in enumerate(models):
    print(f"\n{model}:")
    for j, method in enumerate(methods):
        throughput = throughput_data[i, j]
        edge_load = edge_data[i, j]
        throughput_str = "∞" if np.isinf(throughput) else f"{throughput:.2f}"
        print(f"  {method}: 吞吐量={throughput_str} fps, 边侧负载={edge_load:.4f} GFLOPs")

