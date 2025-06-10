import json
import matplotlib.pyplot as plt
import numpy as np

# Load actual results
with open('results/experiment_results.json', 'r') as f:
    results = json.load(f)

# Extract data
datasets = []
algorithms = list(results.keys())
data = {alg: {'costs': [], 'times': [], 'cities': []} for alg in algorithms}

# Get common datasets
all_datasets = set()
for alg in algorithms:
    all_datasets.update(results[alg].keys())

datasets = sorted(list(all_datasets))

for ds in datasets:
    for alg in algorithms:
        if ds in results[alg]:
            data[alg]['costs'].append(results[alg][ds]['cost'])
            data[alg]['times'].append(results[alg][ds]['runtime'])
            data[alg]['cities'].append(results[alg][ds]['cities'])
        else:
            data[alg]['costs'].append(None)
            data[alg]['times'].append(None) 
            data[alg]['cities'].append(None)

# Plot 1: Cost comparison
plt.figure(figsize=(12, 8))
for i, alg in enumerate(algorithms):
    valid_costs = [c for c in data[alg]['costs'] if c is not None]
    valid_datasets = [ds for j, ds in enumerate(datasets) if data[alg]['costs'][j] is not None]
    
    plt.plot(valid_datasets, valid_costs, 'o-', linewidth=3, markersize=8, 
             label=alg.upper(), alpha=0.8)
    
    # Add value labels
    for x, y in zip(valid_datasets, valid_costs):
        plt.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)

plt.xlabel('Datasets', fontsize=14, fontweight='bold')
plt.ylabel('Tour Cost', fontsize=14, fontweight='bold')
plt.title('Actual Experimental Results: Cost Comparison', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('real_cost_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Runtime comparison  
plt.figure(figsize=(12, 8))
for alg in algorithms:
    valid_times = [t for t in data[alg]['times'] if t is not None]
    valid_datasets = [ds for j, ds in enumerate(datasets) if data[alg]['times'][j] is not None]
    
    plt.plot(valid_datasets, valid_times, 'o-', linewidth=3, markersize=8, 
             label=alg.upper(), alpha=0.8)

plt.xlabel('Datasets', fontsize=14, fontweight='bold')
plt.ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
plt.title('Actual Experimental Results: Runtime Comparison', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('real_runtime_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Real experimental plots generated!")