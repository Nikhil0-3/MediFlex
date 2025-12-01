import matplotlib.pyplot as plt
import numpy as np

# Data for the comparison of models
models = ['CNN', 'Transformer', 'XLNet', 'BiLSTM']
accuracy = [84.10, 85.00, 88.00, 82.50]
precision = [97.85, 96.00, 97.00, 94.50]
recall = [94.17, 92.50, 94.00, 90.00]
f1_score = [95.97, 94.00, 95.00, 92.50]
auc_roc = [99.25, 98.50, 99.00, 97.50]

# Set up the plot (larger figure for clarity)
fig, ax = plt.subplots(figsize=(14, 8))

# Set the position of the bars on the x-axis
x = np.arange(len(models))

# Set the width of the bars
width = 0.18

# Professional and distinct colors for each metric
color_accuracy = '#1f77b4'  # Blue
color_precision = '#ff6347'  # Tomato red
color_recall = '#32cd32'     # Lime green
color_f1_score = '#ff1493'   # Deep pink
color_auc_roc = '#00bfff'    # Deep sky blue

# Plotting the bars for each metric
bars_accuracy = ax.bar(x - 2*width, accuracy, width, label='Accuracy', color=color_accuracy)
bars_precision = ax.bar(x - width, precision, width, label='Precision', color=color_precision)
bars_recall = ax.bar(x, recall, width, label='Recall', color=color_recall)
bars_f1_score = ax.bar(x + width, f1_score, width, label='F1-Score', color=color_f1_score)
bars_auc_roc = ax.bar(x + 2*width, auc_roc, width, label='AUC-ROC', color=color_auc_roc)

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Models', fontsize=16)
ax.set_ylabel('Percentage (%)', fontsize=16)
ax.set_title('Comparison of Model Performance Metrics', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14)
ax.legend(fontsize=12)

# Customize gridlines for clarity
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the values on top of each bar for better readability
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',  # Display percentage value with two decimal places
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset the text slightly
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

# Add values to each set of bars
add_values(bars_accuracy)
add_values(bars_precision)
add_values(bars_recall)
add_values(bars_f1_score)
add_values(bars_auc_roc)

# Show the plot
plt.tight_layout()
plt.show()
