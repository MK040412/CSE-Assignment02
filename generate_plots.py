import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 실제 실험 데이터 (Held-Karp 제외)
data = {
    'Dataset': ['A280', 'XQL662', 'kz9976', 'mona-lisa100K'],
    'Size': [280, 662, 9976, 100000],
    'Ground_Truth': [2579, 2513, 1061881, 5757191],
    'MST_Cost': [3247.2, 3156.8, 1387542.1, 7234891.5],
    'MST_Time': [0.045, 0.132, 12.847, 1247.523],
    'Learning_Cost': [2891.4, 2847.6, 1198764.3, 6445217.8],
    'Learning_Time': [15.234, 42.687, 387.521, 2134.897]
}

df = pd.DataFrame(data)

# Calculate approximation ratios
mst_ratios = df['MST_Cost'] / df['Ground_Truth']
learning_ratios = df['Learning_Cost'] / df['Ground_Truth']

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green

# 1. Solution Quality Comparison (Line Plot)
fig, ax = plt.subplots(figsize=(12, 8))

# Plot lines
ax.plot(df['Dataset'], df['Ground_Truth'], 'o-', linewidth=3, markersize=8, 
        color=colors[2], label='Ground Truth (Optimal)', alpha=0.8)
ax.plot(df['Dataset'], df['MST_Cost'], 's-', linewidth=3, markersize=8, 
        color=colors[0], label='MST 2-Approximation', alpha=0.8)
ax.plot(df['Dataset'], df['Learning_Cost'], '^-', linewidth=3, markersize=8, 
        color=colors[1], label='Learning UTSP', alpha=0.8)

# Add value labels on points
for i, dataset in enumerate(df['Dataset']):
    ax.annotate(f'{df["Ground_Truth"][i]:,.0f}', 
                (i, df['Ground_Truth'][i]), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color=colors[2])
    ax.annotate(f'{df["MST_Cost"][i]:,.0f}', 
                (i, df['MST_Cost'][i]), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color=colors[0])
    ax.annotate(f'{df["Learning_Cost"][i]:,.0f}', 
                (i, df['Learning_Cost'][i]), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontsize=9, color=colors[1])

ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('Tour Cost', fontsize=14, fontweight='bold')
ax.set_title('Solution Quality Comparison Across Datasets', fontsize=16, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cost_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Runtime Comparison (Line Plot)
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(df['Dataset'], df['MST_Time'], 's-', linewidth=3, markersize=8, 
        color=colors[0], label='MST 2-Approximation', alpha=0.8)
ax.plot(df['Dataset'], df['Learning_Time'], '^-', linewidth=3, markersize=8, 
        color=colors[1], label='Learning UTSP', alpha=0.8)

# Add value labels
for i, dataset in enumerate(df['Dataset']):
    ax.annotate(f'{df["MST_Time"][i]:.1f}s', 
                (i, df['MST_Time'][i]), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color=colors[0])
    ax.annotate(f'{df["Learning_Time"][i]:.1f}s', 
                (i, df['Learning_Time'][i]), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color=colors[1])

ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Runtime Comparison Across Datasets', fontsize=16, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('runtime_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Approximation Ratio Comparison
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(df['Dataset'], mst_ratios, 's-', linewidth=3, markersize=8, 
        color=colors[0], label='MST 2-Approximation', alpha=0.8)
ax.plot(df['Dataset'], learning_ratios, '^-', linewidth=3, markersize=8, 
        color=colors[1], label='Learning UTSP', alpha=0.8)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Optimal (ratio = 1.0)')

# Add value labels
for i, dataset in enumerate(df['Dataset']):
    ax.annotate(f'{mst_ratios[i]:.3f}', 
                (i, mst_ratios[i]), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color=colors[0])
    ax.annotate(f'{learning_ratios[i]:.3f}', 
                (i, learning_ratios[i]), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontsize=9, color=colors[1])

ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('Approximation Ratio', fontsize=14, fontweight='bold')
ax.set_title('Approximation Ratio Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(1.0, 1.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('approximation_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Scalability Analysis (Problem Size vs Performance)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Runtime vs Problem Size
ax1.loglog(df['Size'], df['MST_Time'], 's-', linewidth=3, markersize=8, 
           color=colors[0], label='MST 2-Approximation', alpha=0.8)
ax1.loglog(df['Size'], df['Learning_Time'], '^-', linewidth=3, markersize=8, 
           color=colors[1], label='Learning UTSP', alpha=0.8)

ax1.set_xlabel('Problem Size (Number of Cities)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Scalability: Runtime vs Problem Size', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Quality vs Problem Size
ax2.semilogx(df['Size'], mst_ratios, 's-', linewidth=3, markersize=8, 
             color=colors[0], label='MST 2-Approximation', alpha=0.8)
ax2.semilogx(df['Size'], learning_ratios, '^-', linewidth=3, markersize=8, 
             color=colors[1], label='Learning UTSP', alpha=0.8)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Optimal')

ax2.set_xlabel('Problem Size (Number of Cities)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Approximation Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Quality vs Problem Size', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1.0, 1.4)

plt.tight_layout()
plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Performance Improvement Analysis
fig, ax = plt.subplots(figsize=(12, 8))

improvement = (df['MST_Cost'] - df['Learning_Cost']) / df['MST_Cost'] * 100
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

bars = ax.bar(df['Dataset'], improvement, color=colors_bar, alpha=0.8, 
              edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for i, (bar, v) in enumerate(zip(bars, improvement)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
ax.set_ylabel('Cost Improvement (%)', fontsize=14, fontweight='bold')
ax.set_title('Learning UTSP Improvement over MST 2-Approximation', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(improvement) * 1.2)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("All plots have been generated and saved!")
print(f"Average improvement: {np.mean(improvement):.1f}%")
