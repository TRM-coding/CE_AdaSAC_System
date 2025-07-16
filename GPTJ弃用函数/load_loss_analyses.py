import pickle
import matplotlib.pyplot as plt

def load_evaluation_results(path='evaluation_results.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load data
evaluation_results = load_evaluation_results()

# Compute sums and avg_losses
list_sums = []
avg_losses = []
for lst, meta in evaluation_results.values():
    list_sums.append(sum(lst))
    avg_losses.append(meta.get('avg_loss'))

# 对数据按 list_sums 排序
pairs = sorted(zip(list_sums, avg_losses), key=lambda x: x[0])
x_sorted, y_sorted = zip(*pairs)

# Plot line chart
fig, ax = plt.subplots()
ax.plot(x_sorted, y_sorted, marker='o', linestyle='-')
ax.set_xlabel('List Sum')
ax.set_ylabel('Avg Loss')
ax.set_title('List Sum vs Avg Loss (Line Plot)')
plt.tight_layout()

# Save the figure to a file
output_path = 'list_sum_vs_avg_loss_line.png'
fig.savefig(output_path, dpi=300)

# Display the plot
plt.show()

print(f"Line plot saved to: {output_path}")
