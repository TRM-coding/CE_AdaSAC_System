import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set academic style similar to plt_data.py
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.figsize': (18, 8),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Data definition
methods = ["FLOAT-16", "INT-8", "CE_AdaSAC_α=0.9", "CE_AdaSAC_α=0.1", "Split Only"]
models = ['GPT-J-6B', "ResNet-50", "VGG-16", "AlexNet"]

# Loss data
loss_data = {
    'GPT-J-6B': [1.9674, 2.2101, 2.08253515625, 3.4496484375, 1.9674],
    'ResNet-50': [0.9682, 0.9678, 1.0641642015576362, 1.1258292582035065, 0.9681],
    'VGG-16': [1.1538, 1.1538, 1.1534, 4.497893967866897, 1.1716370739638806],
    'AlexNet': [1.9227, 1.9226, 2.2105, 4.880354819774627,1.9227]
}

# Overhead data (as percentage)
overhead_data = {
    'GPT-J-6B': [100, 100, 78.1, 64.94, 83.32],
    'ResNet-50': [100, 100, 32.02, 14.47, 43.87],
    'VGG-16': [100, 100, 10.16, 5.465, 66.02],
    'AlexNet': [100, 100, 6.926, 3.576, 41.199]
}

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(18, 12))
ax2 = ax1.twinx()

# Color scheme for professional publication - highlight CE_AdaSAC methods
colors = ['#7F8C8D', '#95A5A6', '#E74C3C', '#C0392B', '#34495E']  # CE_AdaSAC methods use striking reds
edge_colors = ['#5D6D7E', '#7B8B9C', '#A93226', '#8B2635', '#2C3E50']
patterns = ['', '///', '|||', 'xxx', '+++']  # Different patterns for CE_AdaSAC methods

# Set positions for bars
n_models = len(models)
n_methods = len(methods)
x_pos = np.arange(n_models)
bar_width = 0.16  # Increased bar width to accommodate horizontal labels
spacing = 0.02

# Plot overhead bars (primary y-axis)
overhead_bars = []
for i, method in enumerate(methods):
    x_offset = (i - n_methods/2 + 0.5) * (bar_width + spacing)
    overhead_values = [overhead_data[model][i] for model in models]
    bars = ax1.bar(x_pos + x_offset, overhead_values, 
                   width=bar_width, 
                   label=f'{method}',
                   color=colors[i], 
                   edgecolor='black',
                   linewidth=0.8,
                   alpha=0.8,
                   hatch=patterns[i])
    overhead_bars.append(bars)

# Plot loss scatter points (secondary y-axis) - no lines between points
for i, method in enumerate(methods):
    x_offset = (i - n_methods/2 + 0.5) * (bar_width + spacing)
    loss_values = [loss_data[model][i] for model in models]
    ax2.scatter(x_pos + x_offset, loss_values, 
               s=400,  # Increased size for better visibility
               color=colors[i], 
               marker='*',  # Five-pointed star
               facecolors=colors[i], 
               edgecolors='black',  # Black edge for better contrast
               linewidths=2, 
               alpha=0.9, 
               zorder=5)

# Add connecting lines for each model's loss values across different methods
for j, model in enumerate(models):
    model_loss_values = [loss_data[model][i] for i in range(n_methods)]
    model_x_positions = []
    for i in range(n_methods):
        x_offset = (i - n_methods/2 + 0.5) * (bar_width + spacing)
        model_x_positions.append(x_pos[j] + x_offset)
    
    # Draw connecting line for this model
    ax2.plot(model_x_positions, model_loss_values, 
             color='blue', 
             linewidth=4, 
             alpha=0.6, 
             linestyle='--',
             zorder=3)

# Customize primary y-axis (overhead)
ax1.set_xlabel('Neural Network Models', fontsize=24, fontweight='bold')
ax1.set_ylabel('Computation Overhead (%)', fontsize=24, fontweight='bold', color='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=24, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.set_ylim(0, 140)  # Increased upper limit to make room for legend
ax1.grid(True, alpha=0.3, axis='y')

# Customize secondary y-axis (loss)
ax2.set_ylabel('Model Loss', fontsize=24, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=18)

# Set appropriate y-axis limits for loss - adjust to leave space for legend
all_loss_values = []
for model in models:
    all_loss_values.extend(loss_data[model])
loss_min, loss_max = min(all_loss_values), max(all_loss_values)
loss_range = loss_max - loss_min
ax2.set_ylim(loss_min - 0.1*loss_range, loss_max + 0.4*loss_range)  # Increased upper margin for legend space

# Create comprehensive legend
overhead_patches = [mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i], 
                                      edgecolor='black', alpha=0.8,
                                      hatch=patterns[i]) 
                   for i in range(n_methods)]

# Create main legend positioned inside the plot area
main_legend = ax1.legend(overhead_patches, [f'{method}' for method in methods], 
                        loc='upper center', bbox_to_anchor=(0.5, 0.99),
                        title='Methods', title_fontsize=24, fontsize=20,
                        frameon=True, fancybox=True, shadow=True,
                        ncol=5)  # 3 columns to save vertical space
main_legend.get_title().set_fontweight('bold')
main_legend.get_frame().set_facecolor('white')
main_legend.get_frame().set_alpha(0.95)
main_legend.get_frame().set_edgecolor('gray')
main_legend.get_frame().set_linewidth(1)

# Add explanation text positioned to not conflict with legend
ax1.text(0.28, 0.79, 'Bars: Edge Computation Overhead (%) | Dots: Model Loss', 
         transform=ax1.transAxes, fontsize=20, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9,
                  edgecolor='gray', linewidth=1),
         verticalalignment='top')

# Add title
# plt.title('Performance Comparison: Edge Overhead vs Model Loss\nAcross Different Neural Network Architectures', 
#           fontsize=26, fontweight='bold', pad=25)

# Remove background color for cleaner look

# Find and mark the shortest bar in each model group with horizontal lines
for j, model in enumerate(models):
    model_overhead_values = overhead_data[model]
    # min_overhead_in_model = min(model_overhead_values)
    model_overhead_values = overhead_data[model]
    min_overhead_in_model = sorted(model_overhead_values)[1]
    
    # Calculate x-range for this model's bars
    x_start = x_pos[j] - (n_methods/2) * (bar_width + spacing) - spacing/2
    x_end = x_pos[j] + (n_methods/2) * (bar_width + spacing) + spacing/2
    
    # Add horizontal line for minimum value in this model group
    # ax1.hlines(y=min_overhead_in_model, xmin=x_start, xmax=x_end, 
    #            color='red', linestyle='-', linewidth=5, alpha=0.7, zorder=1)
    
    # Add text annotation for the minimum value
    # ax1.text(x_end , min_overhead_in_model+3, f'{min_overhead_in_model:.1f}%', 
    #          fontsize=12, fontweight='bold',
    #          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9,
    #                   edgecolor='red', linewidth=1),
    #          verticalalignment='center', color='red',zorder=10)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure in high resolution for publication
plt.savefig('acc_overhead.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('acc_overhead.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Display the plot
plt.show()

print("Professional publication-ready plots saved as:")
print("- acc_overhead.png (300 DPI)")
print("- acc_overhead.pdf (Vector format)")

