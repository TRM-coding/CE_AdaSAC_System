import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set IEEE/INFOCOM style parameters
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 16

# Load the data
with open('GPTJ_ISP_VAL_generated.pkl','rb') as f:
    data_dict = pickle.load(f)

with open('GPTJ_ISP_VAL.pkl','rb') as f:
    data_dict2 = pickle.load(f)

# Extract data for plotting
x_axis = []  # Total compression ratio (sum of SVD compression ratios)
y_axis = []  # Loss values
x2_axis = []
y2_axis = []

for key, value in data_dict.items():
    compression_ratio = sum(key)/len(key)  # Sum of all layer compression ratios
    x_axis.append(compression_ratio)
    y_axis.append(value['loss'].cpu())

for key, value in data_dict2.items():
    compression_ratio = sum(key)/len(key)  # Sum of all layer compression ratios
    x2_axis.append(compression_ratio)
    y2_axis.append(value['loss'])

# Sort data by compression ratio for better visualization
sorted_indices = np.argsort(x_axis)
x_axis = np.array(x_axis)[sorted_indices]
y_axis = np.array(y_axis)[sorted_indices]

sorted_indices2 = np.argsort(x2_axis)
x2_axis = np.array(x2_axis)[sorted_indices2]
y2_axis = np.array(y2_axis)[sorted_indices2]

# Create the figure with dual y-axes (INFOCOM standard format)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot first dataset on the left y-axis
color1 = '#2E86AB'  # Deep blue
ax1.set_xlabel('Total Compression Ratio', fontweight='bold', fontsize=14)
ax1.set_ylabel('Loss (Dataset 1)', color=color1, fontweight='bold', fontsize=14)

# Plot first dataset with professional styling
line1 = ax1.plot(x_axis, y_axis, 'o-', color=color1, linewidth=3, 
                markersize=10, markerfacecolor=color1, markeredgewidth=1, 
                markeredgecolor='white', label='Generated Data', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)

# Create second y-axis for the second dataset
ax2 = ax1.twinx()
color2 = '#F18F01'  # Vibrant orange
ax2.set_ylabel('Loss (Dataset 2)', color=color2, fontweight='bold', fontsize=14)

# Plot second dataset with different markers and line style
line2 = ax2.plot(x2_axis, y2_axis, '^--', color=color2, linewidth=3, 
                markersize=10, markerfacecolor=color2, markeredgewidth=1, 
                markeredgecolor='white', label='Original Data', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

# Adjust y-axis ranges for better visualization
loss_range1 = max(y_axis) - min(y_axis)
ax1.set_ylim(min(y_axis) - 0.1 * loss_range1, max(y_axis) + 0.1 * loss_range1)

loss_range2 = max(y2_axis) - min(y2_axis)
ax2.set_ylim(min(y2_axis) - 0.1 * loss_range2, max(y2_axis) + 0.1 * loss_range2)

# Set title with improved formatting
plt.title('GPT-J Model Performance vs Compression Ratio\nISP Algorithm Analysis (Dual Dataset Comparison)', 
          fontweight='bold', pad=25, fontsize=16)

# Create combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
legend = ax1.legend(lines, labels, loc='upper left', frameon=True, 
                   fancybox=True, shadow=True, fontsize=12,
                   bbox_to_anchor=(0.02, 0.98))
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Improve layout and aesthetics
plt.tight_layout()

# Add subtle background with better contrast
ax1.set_facecolor('#f8f9fa')

# Set axis limits with better spacing - use the combined x range
all_x = np.concatenate([x_axis, x2_axis])
x_range = max(all_x) - min(all_x)
ax1.set_xlim(min(all_x) - 0.02 * x_range, max(all_x) + 0.02 * x_range)

# Add minor ticks for better readability
ax1.minorticks_on()
ax2.minorticks_on()

# Improve grid appearance
ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
ax1.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.4)

# Save the figure in multiple formats for publication
plt.savefig('gptj_isp_performance_analysis_isp_generated.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('gptj_isp_performance_analysis_generated.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('gptj_isp_performance_analysis_generated.eps', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()

# Print summary statistics
print("="*60)
print("GPT-J ISP Algorithm Performance Analysis Summary")
print("="*60)
print("Dataset 1 (Generated):")
print(f"  Number of configurations: {len(x_axis)}")
print(f"  Compression ratio range: {min(x_axis):.3f} - {max(x_axis):.3f}")
print(f"  Loss range: {min(y_axis):.4f} - {max(y_axis):.4f}")
print(f"  Lowest loss: {min(y_axis):.4f} at compression ratio {x_axis[np.argmin(y_axis)]:.3f}")

print("\nDataset 2 (Original):")
print(f"  Number of configurations: {len(x2_axis)}")
print(f"  Compression ratio range: {min(x2_axis):.3f} - {max(x2_axis):.3f}")
print(f"  Loss range: {min(y2_axis):.4f} - {max(y2_axis):.4f}")
print(f"  Lowest loss: {min(y2_axis):.4f} at compression ratio {x2_axis[np.argmin(y2_axis)]:.3f}")
print("="*60)