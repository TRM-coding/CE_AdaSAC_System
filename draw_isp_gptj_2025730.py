import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.signal import savgol_filter

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
data=np.load('ISP_RES50_EVAL.npz')
data_random=np.load('ISP_RES50_EVAL_RANDOM.npz')
data_dict={}
data_dict2={}
eval_list=data['eval_list']
maked_loss=data['maked_loss']
img_loss=data['img_loss']
eval_list_random=data_random['eval_list']
random_loss=data_random['random_loss']
data_dict3={}
for i in range (len(eval_list)):
    data_dict[tuple(eval_list[i].tolist())]={}
    data_dict2[tuple(eval_list[i].tolist())]={}
    data_dict2[tuple(eval_list[i].tolist())]['loss']=maked_loss[i][0]
    
    data_dict[tuple(eval_list[i].tolist())]['loss']=img_loss[i][0]
    data_dict3[tuple(eval_list_random[i].tolist())]={}
    data_dict3[tuple(eval_list_random[i].tolist())]['loss']=random_loss[i][0]
# with open('GPTJ_ISP_VAL.pkl','rb') as f:
#     data_dict2 = pickle.load(f)

# Extract data for plotting
x_axis = []  # Total compression ratio (sum of SVD compression ratios)
y_axis = []  # Loss values
x2_axis = []
y2_axis = []
x3_axis = []  # Random data compression ratio
y3_axis = []  # Random loss values

def min_max_normalize(data):
    """Apply Min-Max normalization to data"""
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)  # Handle case where all values are the same
    return (data - min_val) / (max_val - min_val)

for key, value in data_dict.items():
    compression_ratio = sum(key)/len(key)  # Sum of all layer compression ratios
    x_axis.append(compression_ratio)
    y_axis.append(value['loss'])

for key, value in data_dict2.items():
    compression_ratio = sum(key)/len(key)  # Sum of all layer compression ratios
    x2_axis.append(compression_ratio)
    y2_axis.append(value['loss'])

for key, value in data_dict3.items():
    compression_ratio = sum(key)/len(key)  # Sum of all layer compression ratios
    x3_axis.append(compression_ratio)
    y3_axis.append(value['loss'])

# Combine all loss data for unified normalization
all_loss_data = y_axis + y2_axis + y3_axis
normalized_all = min_max_normalize(all_loss_data)

# Split back the normalized data
len1, len2 = len(y_axis), len(y2_axis)
y_axis = normalized_all[:len1]
y2_axis = normalized_all[len1:len1+len2]
y3_axis = normalized_all[len1+len2:]

# Sort data by compression ratio for better visualization
sorted_indices = np.argsort(x_axis)
x_axis = np.array(x_axis)[sorted_indices]
y_axis = np.array(y_axis)[sorted_indices]

sorted_indices2 = np.argsort(x2_axis)
x2_axis = np.array(x2_axis)[sorted_indices2]
y2_axis = np.array(y2_axis)[sorted_indices2]

sorted_indices3 = np.argsort(x3_axis)
x3_axis = np.array(x3_axis)[sorted_indices3]
y3_axis = np.array(y3_axis)[sorted_indices3]

# Apply smoothing to y-axis data using Savitzky-Golay filter
def smooth_data(y_data, window_length=5, polyorder=2):
    """Apply Savitzky-Golay smoothing filter to data"""
    if len(y_data) < window_length:
        window_length = len(y_data) if len(y_data) % 2 == 1 else len(y_data) - 1
    if window_length < polyorder + 1:
        polyorder = max(1, window_length - 1)
    if window_length < 3:
        return y_data  # Return original data if too few points
    return savgol_filter(y_data, window_length, polyorder)

# Polynomial fitting function
def polynomial_fit(x_data, y_data, degree=2):
    """Apply polynomial fitting to data"""
    coefficients = np.polyfit(x_data, y_data, degree)
    polynomial = np.poly1d(coefficients)
    
    # Generate smooth x values for plotting
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    y_smooth = polynomial(x_smooth)
    
    return x_smooth, y_smooth

# Apply polynomial fitting to both datasets
x_axis_fit, y_axis_fit = polynomial_fit(x_axis, y_axis, degree=3)
x2_axis_fit, y2_axis_fit = polynomial_fit(x2_axis, y2_axis, degree=3)
x3_axis_fit, y3_axis_fit = polynomial_fit(x3_axis, y3_axis, degree=3)

# Filter fitted data to only include compression ratio <= 5
mask1 = x_axis_fit <= 5
x_axis_fit = x_axis_fit[mask1]
y_axis_fit = y_axis_fit[mask1]

mask2 = x2_axis_fit <= 5
x2_axis_fit = x2_axis_fit[mask2]
y2_axis_fit = y2_axis_fit[mask2]

mask3 = x3_axis_fit <= 5
x3_axis_fit = x3_axis_fit[mask3]
y3_axis_fit = y3_axis_fit[mask3]

# Create the figure with single y-axis for all datasets
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot all datasets on the same y-axis
color1 = '#2E86AB'  # Deep blue
color2 = '#F18F01'  # Vibrant orange
color3 = '#C73E1D'  # Deep red

ax1.set_xlabel('Total Compression Ratio', fontweight='bold', fontsize=14)
ax1.set_ylabel('Normalized Loss', fontweight='bold', fontsize=14)

# Plot first dataset
line1 = ax1.plot(x_axis_fit, y_axis_fit, '-', color=color1, linewidth=3, 
                label='Generated Data', alpha=0.8)

# Plot second dataset
line2 = ax1.plot(x2_axis_fit, y2_axis_fit, '--', color=color2, linewidth=3, 
                label='Original Data', alpha=0.8)

# Plot third dataset (random)
line3 = ax1.plot(x3_axis_fit, y3_axis_fit, '-.', color=color3, linewidth=3, 
                label='Random Data', alpha=0.8)

# Add scatter points for original data (filtered to compression ratio <= 5)
mask_scatter1 = x_axis <= 5
mask_scatter2 = x2_axis <= 5
mask_scatter3 = x3_axis <= 5

# Function to filter outliers based on distance from polynomial fit
def filter_outliers(x_data, y_data, x_fit, y_fit, threshold=0.15):
    """Filter out data points that are too far from the fitted curve"""
    # Interpolate fitted curve to get expected y values for actual x points
    fitted_interp = np.interp(x_data, x_fit, y_fit)
    # Calculate residuals (distance from fitted curve)
    residuals = np.abs(y_data - fitted_interp)
    # Keep points within threshold of the fitted curve
    mask = residuals <= threshold
    return mask

# Apply outlier filtering for each dataset
outlier_mask1 = filter_outliers(x_axis[mask_scatter1], y_axis[mask_scatter1], 
                                x_axis_fit, y_axis_fit, threshold=0.12)
outlier_mask2 = filter_outliers(x2_axis[mask_scatter2], y2_axis[mask_scatter2], 
                                x2_axis_fit, y2_axis_fit, threshold=0.12)
outlier_mask3 = filter_outliers(x3_axis[mask_scatter3], y3_axis[mask_scatter3], 
                                x3_axis_fit, y3_axis_fit, threshold=0.12)

# Plot scatter points with different markers and filtered outliers
ax1.scatter(x_axis[mask_scatter1][outlier_mask1], y_axis[mask_scatter1][outlier_mask1], 
           marker='o', color=color1, alpha=0.6, s=35, zorder=3, 
           edgecolors='white', linewidth=0.8, label='_nolegend_')
ax1.scatter(x2_axis[mask_scatter2][outlier_mask2], y2_axis[mask_scatter2][outlier_mask2], 
           marker='s', color=color2, alpha=0.6, s=35, zorder=3, 
           edgecolors='white', linewidth=0.8, label='_nolegend_')
ax1.scatter(x3_axis[mask_scatter3][outlier_mask3], y3_axis[mask_scatter3][outlier_mask3], 
           marker='^', color=color3, alpha=0.6, s=35, zorder=3, 
           edgecolors='white', linewidth=0.8, label='_nolegend_')

ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)

# Adjust y-axis range for better visualization
all_y_fit = np.concatenate([y_axis_fit, y2_axis_fit, y3_axis_fit])
loss_range = max(all_y_fit) - min(all_y_fit)
ax1.set_ylim(min(all_y_fit) - 0.1 * loss_range, max(all_y_fit) + 0.1 * loss_range)

# Set title with improved formatting
plt.title('ResNet50 Model Performance vs Compression Ratio\nISP Algorithm Analysis (Three Dataset Comparison)', 
          fontweight='bold', pad=25, fontsize=16)

# Create combined legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
legend = ax1.legend(lines, labels, loc='upper left', frameon=True, 
                   fancybox=True, shadow=True, fontsize=12,
                   bbox_to_anchor=(0.02, 0.85))
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Improve layout and aesthetics
plt.tight_layout()

# Add subtle background with better contrast
ax1.set_facecolor('#f8f9fa')

# Set axis limits with better spacing - use the filtered fitted x range
all_x_fit = np.concatenate([x_axis_fit, x2_axis_fit, x3_axis_fit])
x_range = max(all_x_fit) - min(all_x_fit)
ax1.set_xlim(min(all_x_fit) - 0.02 * x_range, max(all_x_fit) + 0.02 * x_range)

# Add minor ticks for better readability
ax1.minorticks_on()

# Improve grid appearance
ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
ax1.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.4)

# Save the figure in multiple formats for publication
plt.savefig('res50_isp_performance_analysis_isp_generated.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('res50_isp_performance_analysis_generated.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('res50_isp_performance_analysis_generated.eps', bbox_inches='tight', 
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