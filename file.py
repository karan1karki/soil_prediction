import matplotlib.pyplot as plt
import numpy as np

# Set design theme metrics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Diagram 1: Accuracy Index comparison across model tasks
tasks = ['Soil Profile\nClassification', 'Moisture\nRegression', 'Fertilizer\nRecommendation']
accuracy_scores = [94.2, 91.8, 93.5]
bars = ax1.bar(tasks, accuracy_scores, color='#1B5E20', width=0.5, edgecolor='black', alpha=0.85)
ax1.set_ylim(0, 110)
ax1.set_ylabel('Model Accuracy Evaluation Index (%)', fontsize=11, fontweight='bold', color='#333333')
ax1.set_title('AgriSmart AI Model Performance Matrix', fontsize=13, fontweight='bold', pad=15, color='#1B5E20')

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval}%', ha='center', va='bottom', fontweight='bold', color='#1B5E20')

# Diagram 2: System Latency Distribution across operational layers
layers = ['Image RGB Extraction', 'Network Transit Pipe', 'ONNX Multi-Stage Inference', 'UI State Render']
latency_weight = [0.3, 1.2, 0.6, 0.2]
colors = ['#81C784', '#4FC3F7', '#FFB74D', '#BA68C8']
ax2.pie(latency_weight, labels=layers, autopct='%1.1f%%', startangle=140, colors=colors, 
        wedgeprops={'edgecolor':'black','linewidth':1,'alpha':0.9}, textprops={'fontweight':'bold'})
ax2.set_title('End-to-End Diagnostic Latency Breakdown (Total ~2.3s)', fontsize=13, fontweight='bold', pad=15, color='#1B5E20')

plt.tight_layout()
plt.savefig('agrismart_documentation_metrics.png', dpi=300)
print("SUCCESS: System metrics visualization saved cleanly to disk as 'agrismart_documentation_metrics.png'.")
