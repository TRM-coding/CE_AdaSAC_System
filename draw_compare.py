import numpy as np
import matplotlib.pyplot as plt

# 模型名称与方案名称
models = ["ResNet50", "VGG", "AlexNet"]
names = ['auto-split', 'NEURO', 'QDMP', 'U8', 'CLOUD8', 'CE_AdaSAC(Ours)']

# 数据（缺失值用 np.nan 填充）
resnet50_time_unnormal = np.array([120, 80, 118, 1227, 118, 60])
resnet50_time= np.array([(i-min(resnet50_time_unnormal))/(max(resnet50_time_unnormal)-min(resnet50_time_unnormal)) for i in resnet50_time_unnormal])+0.1
resnet50_acc  = np.array([0.71, 0.76, 0.73, 0.69, 0.71, 0.73])

vgg_time_unnormal = np.array([102, 63, 117, 960, 117, 50])
vgg_time= np.array([(i-min(vgg_time_unnormal))/(max(vgg_time_unnormal)-min(vgg_time_unnormal)) for i in vgg_time_unnormal])+0.1
vgg_acc  = np.array([0.68, 0.71, 0.66, 0.65, 0.68, 0.70])

# 对于 AlexNet，第一个和最后一个方案缺失数据，这里用 np.nan 表示缺失值
alexnet_time_un = np.array([0, 61, 115, 2200, 115, 40])
alexnet_time= np.array([((i-min(alexnet_time_un))/(max(alexnet_time_un)-min(alexnet_time_un))) for i in alexnet_time_un])+0.1
alexnet_acc  = np.array([0.58, 0.65, 0.57, 0.55, 0.59, 60])

# 将各模型数据整理成列表，方便后续按模型遍历
time_data = [resnet50_time, vgg_time, alexnet_time]
acc_data  = [resnet50_acc, vgg_acc, alexnet_acc]

# 设置组数与每组中方案数
n_models = len(models)
n_methods = len(names)

# x 轴上，每个模型位置
x = np.arange(n_models)

# 每组内部总宽度设为 0.8，每个柱子的宽度
width = 0.8 / n_methods

# 设置论文风格（可选）
# plt.style.use('seaborn-whitegrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

# 使用一个合适的颜色映射，各方案采用依次颜色
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(n_methods)]
# 将最后一列方案（自己的数据）着重显示为红色
colors[-1] = 'red'
colors[-3] = 'gray'

# 绘制主坐标轴（时间数据）的柱状图：按方案分组绘制，每个模型内部有 n_methods 根柱子
for i in range(n_methods):
    # 对于每个方案，计算各模型组内的柱子中心位置
    pos = x - 0.4 + i * width + width / 2.0
    # 对每个模型获取当前方案的时间数据
    times = np.array([model_time[i] for model_time in time_data])
    
    if i == n_methods - 1:
        # 最后一列数据加粗边框以着重表示
        ax1.bar(pos, times, width=width, color=colors[i],
                edgecolor='black', linewidth=2, label=names[i])
    else:
        ax1.bar(pos, times, width=width, color=colors[i], label=names[i])

# 设置主坐标轴标签与刻度
# ax1.set_xlabel("Models", fontsize=12)
ax1.set_ylabel("Time(normaled)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)

# 创建第二坐标轴（准确率数据），准确率值在 0～1 范围内
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_ylim(0, 1)

# 对每个方案，绘制五角星标记表示准确率数据
for i in range(n_methods):
    pos = x - 0.4 + i * width + width / 2.0
    acc = np.array([model_acc[i] if i < len(model_acc) else np.nan for model_acc in acc_data])
    ax2.scatter(pos, acc, marker='*', s=150, color=colors[i],
                edgecolors='black', zorder=5)

# 设置图表标题与图例
ax1.set_title("Performance Comparison of Models under Different Schemes", fontsize=14)
ax1.legend(fontsize=10, loc='upper left', title='Schemes', bbox_to_anchor=(1.05, 1))


ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()

plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')