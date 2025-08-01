import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# 真实数据
# data = {
#     'Method': ['Auto-Split', 'NEURO', 'QDMP', 'CE-AdaSAC'],
#     'alexnet_Gflops_edge': [None, 0.0465, 0.0713, 0.09],
#     'alexnet_cloud_throughtput': [None, 12.3993329158891,  16.5980531976491,32.29118902616232],
#     'vgg16_Gflops_edge': [0, 1.62, 1.953,  1.696952499],
#     'vgg16_cloud_throughtput': [9.77010932752338, 15.8378207158695,  16.3737484252231,51.3347],
#     'resnet50_G_flops_edge': [0, 0.372, 0.42129,  1.1965],
#     'resnet50_cloud_throughtput': [8.33, 16.3150767624362,   16.5451112755471,52.24489795918367]
# }


data = {
    'Method': [ 'Neurosurgeon', 'QDMP', 'CE-AdaSAC'],
    'alexnet_Gflops_edge': [ 0.0465, 0.0713, 0.09],
    'alexnet_cloud_throughtput': [12.3993329158891,  16.5980531976491,32.29118902616232],
    'vgg16_Gflops_edge': [ 1.62, 1.953,  1.696952499],
    'vgg16_cloud_throughtput': [15.8378207158695,  16.3737484252231,51.3347],
    'resnet50_G_flops_edge': [0.372, 0.42129,  1.1965],
    'resnet50_cloud_throughtput': [ 16.3150767624362,   16.5451112755471,52.24489795918367]
}

df = pd.DataFrame(data)

# 模型列表和方法
models = ['AlexNet', 'VGG-16', 'ResNet-50']
methods = [ 'Neurosurgeon', 'QDMP', 'CE-AdaSAC']

# 设置matplotlib参数
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (20, 10),
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
# 首先标记哪些是真正的缺失值
missing_throughput = np.isnan(throughput_data)
missing_edge = np.isnan(edge_data)

for i in range(throughput_data.shape[0]):
    for j in range(throughput_data.shape[1]):
        if np.isnan(throughput_data[i, j]):  # 修复：使用np.isnan检查NaN
            throughput_data[i, j] = 0
        elif np.isinf(throughput_data[i, j]):
            throughput_data[i, j] = 100

# 同样处理edge_data
for i in range(edge_data.shape[0]):
    for j in range(edge_data.shape[1]):
        if np.isnan(edge_data[i, j]):
            edge_data[i, j] = 0

# 设置颜色方案
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
patterns = ['', '///', '...', '+++']

# 添加数值标签函数
def add_value_labels_simple(ax, bars_list, data, format_str='{:.1f}', y_offset=3):
    """简化的数值标签添加函数"""
    for i, bars in enumerate(bars_list):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):  # 只要不是NaN就显示标签（包括0值）
                value = data[j, i] if len(data.shape) > 1 else data[i]
                if np.isinf(value):
                    label = '∞'
                else:
                    label = format_str.format(value)
                
                # 对于0值，标签显示在柱子上方一点点
                if height == 0:
                    y_pos = y_offset
                else:
                    y_pos = y_offset
                
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, y_pos),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.8, edgecolor='none'))

# 创建图例元素
legend_elements = []
for i, method in enumerate(methods):
    bar_patch = mpatches.Patch(color=colors[i], alpha=0.8, 
                              edgecolor='black', linewidth=0.8,
                              hatch=patterns[i], label=method)
    legend_elements.append(bar_patch)

# 创建图1 - 边缘计算负载
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

# 柱状图参数
n_models = len(models)
n_methods = len(methods)
width = 0.15
spacing = 0.02

# 计算每组柱子的位置
n_models = len(models)
n_methods = len(methods)
width = 0.15        # 柱子宽度
spacing = 0.04      # 柱子间距 - 这里调整！

# 计算每组柱子的位置
x_positions = np.arange(n_models)
bar_positions = []

for i in range(n_methods):
    offset = (i - (n_methods-1)/2) * (width + spacing)  # 这里使用spacing
    bar_positions.append(x_positions + offset)

# 图1: 固定云端吞吐量，显示边缘计算负载变化
# 计算归一化的云端吞吐量（使用每个模型的最大值进行归一化，排除缺失值）
throughput_max_per_model = np.zeros(n_models)
for i in range(n_models):
    valid_values = throughput_data[i, ~missing_throughput[i, :]]  # 排除缺失值
    if len(valid_values) > 0:
        throughput_max_per_model[i] = np.max(valid_values)
    else:
        throughput_max_per_model[i] = 1  # 避免除零

throughput_normalized = np.zeros_like(throughput_data)

for i in range(n_models):
    if throughput_max_per_model[i] > 0:
        for j in range(n_methods):
            if not missing_throughput[i, j]:  # 只对非缺失数据进行归一化
                throughput_normalized[i, j] = throughput_data[i, j] / throughput_max_per_model[i]
            else:
                throughput_normalized[i, j] = 0  # 缺失数据保持为0

# 设置固定的归一化吞吐量值（例如都设为1.0表示最优性能）
fixed_throughput = 1.0

# 计算相应的边缘负载调整比例
edge_data_adjusted1 = np.zeros_like(edge_data)
for i in range(n_models):
    for j in range(n_methods):
        if throughput_normalized[i, j] > 0:
            # 边缘负载与吞吐量成反比关系
            adjustment_factor = fixed_throughput / throughput_normalized[i, j]
            edge_data_adjusted1[i, j] = edge_data[i, j] * adjustment_factor
        else:
            edge_data_adjusted1[i, j] = edge_data[i, j]

# 绘制图1 - 显示调整后的边缘计算负载（用柱子）
bars1 = []
for i, method in enumerate(methods):
    # 只为有效数据绘制柱子，显示调整后的边缘负载
    bar_heights = []
    for j in range(n_models):
        if not missing_edge[j, i]:  # 非缺失数据显示调整后的边缘负载（包括0值）
            bar_heights.append(edge_data_adjusted1[j, i])
        else:  # 缺失数据不显示柱子
            bar_heights.append(np.nan)  # 使用NaN而不是0，这样缺失数据不会显示柱子
    
    bars = ax1.bar(bar_positions[i], bar_heights, width, 
                   label=method, color=colors[i], alpha=0.8, 
                   edgecolor='black', linewidth=0.8, hatch=patterns[i])
    bars1.append(bars)

# 设置左Y轴（调整后的边缘计算负载）
ax1.set_xlabel('Neural Network Models', fontsize=16, fontweight='bold')
ax1.set_ylabel('Edge Computing Overhead (GFLOPs)', fontsize=16, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(models, fontsize=18, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 添加图例
ax1.legend(handles=legend_elements, 
          loc='upper left', bbox_to_anchor=(0.01, 0.93),
          title='Methods', title_fontsize=16, fontsize=16,
          frameon=True, fancybox=True, shadow=True,
          ncol=1)

# 添加标签
add_value_labels_simple(ax1, bars1, edge_data_adjusted1, '{:.2f}')

# 设置Y轴范围
valid_edge_adjusted1 = edge_data_adjusted1[~np.isnan(edge_data_adjusted1)]  # 排除NaN值，包含0值
if len(valid_edge_adjusted1) > 0:
    max_val = max(valid_edge_adjusted1)
    if max_val > 0:
        ax1.set_ylim(0, max_val * 1.1)
    else:
        ax1.set_ylim(0, 0.1)  # 如果最大值是0，设置一个小的范围以便看到柱子
else:
    ax1.set_ylim(0, 1)

# 调整布局并保存第一个图
plt.tight_layout()
plt.savefig('edge_computing_load_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('edge_computing_load_comparison.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# 创建图2 - 云端吞吐量
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
# 计算归一化的边缘负载（排除缺失值）
edge_max_per_model = np.zeros(n_models)
for i in range(n_models):
    valid_values = edge_data[i, ~missing_edge[i, :]]  # 排除缺失值
    if len(valid_values) > 0:
        edge_max_per_model[i] = np.max(valid_values)
    else:
        edge_max_per_model[i] = 1  # 避免除零

edge_normalized = np.zeros_like(edge_data)

for i in range(n_models):
    if edge_max_per_model[i] > 0:
        for j in range(n_methods):
            if not missing_edge[i, j]:  # 只对非缺失数据进行归一化
                edge_normalized[i, j] = edge_data[i, j] / edge_max_per_model[i]
            else:
                edge_normalized[i, j] = 0  # 缺失数据保持为0

# 设置固定的归一化边缘负载值
fixed_edge_load = 0.5  # 设为中等水平

# 计算相应的吞吐量调整
throughput_adjusted2 = np.zeros_like(throughput_data)
for i in range(n_models):
    for j in range(n_methods):
        if edge_normalized[i, j] > 0:
            # 吞吐量与边缘负载成反比关系
            adjustment_factor = fixed_edge_load / edge_normalized[i, j] if edge_normalized[i, j] > 0 else 1
            throughput_adjusted2[i, j] = throughput_data[i, j] * adjustment_factor
        else:
            throughput_adjusted2[i, j] = throughput_data[i, j]

# 绘制图2
bars2 = []
for i, method in enumerate(methods):
    # 只为有效数据绘制柱子，显示调整后的云端吞吐量
    bar_heights = []
    for j in range(n_models):
        if not missing_throughput[j, i]:  # 非缺失数据显示调整后的云端吞吐量
            bar_heights.append(throughput_adjusted2[j, i])
        else:  # 缺失数据不显示柱子
            bar_heights.append(np.nan)  # 使用NaN而不是0，这样缺失数据不会显示柱子
    
    bars = ax2.bar(bar_positions[i], bar_heights, width, 
                   label=method, color=colors[i], alpha=0.8, 
                   edgecolor='black', linewidth=0.8, hatch=patterns[i])
    bars2.append(bars)

# 设置左Y轴（调整后的吞吐量）
ax2.set_xlabel('Neural Network Models', fontsize=15, fontweight='bold')
ax2.set_ylabel('Adjusted Cloud Throughput (RPS)', fontsize=15, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=16)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(models, fontsize=18, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 不再需要右Y轴了
# ax2.set_title('Adjusted Cloud Throughput\n(Normalized to Fixed Edge Computing Load)', 
#               fontsize=22, fontweight='bold', pad=20)

# 添加图例（只在第一个子图上）
legend_elements = []
for i, method in enumerate(methods):
    bar_patch = mpatches.Patch(color=colors[i], alpha=0.8, 
                              edgecolor='black', linewidth=0.8,
                              hatch=patterns[i], label=method)
    legend_elements.append(bar_patch)

ax1.legend(handles=legend_elements, 
          loc='upper left', bbox_to_anchor=(0.01, 0.93),
          title='Methods', title_fontsize=14, fontsize=14,
          frameon=True, fancybox=True, shadow=True,
          ncol=1)

ax2.legend(handles=legend_elements, 
          loc='upper left', bbox_to_anchor=(0.01, 0.93),
          title='Methods', title_fontsize=14, fontsize=14,
          frameon=True, fancybox=True, shadow=True,
          ncol=1)

# 添加数值标签函数
def add_value_labels_simple(ax, bars_list, data, format_str='{:.1f}', y_offset=3):
    """简化的数值标签添加函数"""
    for i, bars in enumerate(bars_list):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):  # 只要不是NaN就显示标签（包括0值）
                value = data[j, i] if len(data.shape) > 1 else data[i]
                if np.isinf(value):
                    label = '∞'
                else:
                    label = format_str.format(value)
                
                # 对于0值，标签显示在柱子上方一点点
                if height == 0:
                    y_pos = y_offset
                else:
                    y_pos = y_offset
                
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, y_pos),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.8, edgecolor='none'))

# 添加标签
add_value_labels_simple(ax1, bars1, edge_data_adjusted1, '{:.2f}')
add_value_labels_simple(ax2, bars2, throughput_adjusted2, '{:.1f}')

# 不再需要添加边缘负载标签，因为已经通过柱子显示了

# 设置Y轴范围
valid_edge_adjusted1 = edge_data_adjusted1[~np.isnan(edge_data_adjusted1)]  # 排除NaN值，包含0值
if len(valid_edge_adjusted1) > 0:
    max_val = max(valid_edge_adjusted1)
    if max_val > 0:
        ax1.set_ylim(0, max_val * 1.1)
    else:
        ax1.set_ylim(0, 0.1)  # 如果最大值是0，设置一个小的范围以便看到柱子
else:
    ax1.set_ylim(0, 1)

valid_throughput_adjusted2 = throughput_adjusted2[throughput_adjusted2 > 0]
if len(valid_throughput_adjusted2) > 0:
    ax2.set_ylim(0, max(valid_throughput_adjusted2) * 1.1)
else:
    ax2.set_ylim(0, 100)

# 不再需要设置ax2_twin的范围了

# 添加说明文字
# ax1.text(0.02, 0.85, 'Bars: Adjusted Edge Computing Load', 
#          transform=ax1.transAxes, fontsize=14, 
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
#          verticalalignment='top')

# ax2.text(0.02, 0.85, 'Bars: Adjusted Cloud Throughput', 
#          transform=ax2.transAxes, fontsize=14, 
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
#          verticalalignment='top')

# 调整整体标题
# fig.suptitle('Normalized Performance Analysis: Trade-offs between Cloud Throughput and Edge Computing Load', 
#              fontsize=24, fontweight='bold', y=0.95)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # 为总标题留出空间

# 保存图片
plt.savefig('normalized_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('normalized_comparison.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()

# 打印数据摘要
print("=== 归一化分析摘要 ===")
print("\n图1 - 固定云端吞吐量分析:")
print("云端吞吐量固定为: 1.0 (归一化值)")
print("调整后的边缘计算负载:")
for i, model in enumerate(models):
    print(f"\n{model}:")
    for j, method in enumerate(methods):
        edge_load = edge_data_adjusted1[i, j]
        print(f"  {method}: {edge_load:.4f} GFLOPs")

print("\n图2 - 固定边缘计算负载分析:")
print(f"边缘计算负载固定为: {fixed_edge_load} (归一化比例)")
print("调整后的云端吞吐量:")
for i, model in enumerate(models):
    print(f"\n{model}:")
    for j, method in enumerate(methods):
        throughput = throughput_adjusted2[i, j]
        print(f"  {method}: {throughput:.2f} RPS")
