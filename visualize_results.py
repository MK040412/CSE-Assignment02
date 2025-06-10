import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_results(filepath='results/experiment_results.json'):
    """Load experimental results"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_comparison_plots(results_data):
    """Create comprehensive comparison plots"""
    
    # Extract data
    algorithms = list(results_data['data'].keys())
    datasets = []
    
    # Find common datasets
    all_datasets = set()
    for alg in algorithms:
        all_datasets.update(results_data['data'][alg].keys())
    datasets = sorted(list(all_datasets))
    
    # Prepare data structure
    plot_data = []
    for ds in datasets:
        row = {'dataset': ds}
        for alg in algorithms:
            if ds in results_data['data'][alg]:
                result = results_data['data'][alg][ds]
                if result['cost'] is not None:
                    row[f'{alg}_cost'] = result['cost']
                    row[f'{alg}_time'] = result['runtime']
                    row[f'{alg}_cities'] = result['cities']
                else:
                    row[f'{alg}_cost'] = None
                    row[f'{alg}_time'] = None
                    row[f'{alg}_cities'] = None
            else:
                row[f'{alg}_cost'] = None
                row[f'{alg}_time'] = None
                row[f'{alg}_cities'] = None
        plot_data.append(row)
    
    df = pd.DataFrame(plot_data)
    
    # Color scheme
    colors = {
        'mst': '#E74C3C',      # Red
        'utsp': '#3498DB',     # Blue  
        'heldkarp': '#2ECC71'  # Green
    }
    
    # 1. Cost Comparison
    plt.figure(figsize=(14, 8))
    
    for alg in algorithms:
        cost_col = f'{alg}_cost'
        if cost_col in df.columns:
            valid_data = df[df[cost_col].notna()]
            if not valid_data.empty:
                plt.plot(valid_data['dataset'], valid_data[cost_col], 
                        'o-', linewidth=3, markersize=8, 
                        color=colors.get(alg, 'gray'), 
                        label=alg.upper(), alpha=0.8)
                
                # Add value labels
                for _, row in valid_data.iterrows():
                    plt.annotate(f'{row[cost_col]:,.0f}', 
                               (row['dataset'], row[cost_col]), 
                               textcoords="offset points", 
                               xytext=(0, 10), ha='center', fontsize=9)
    
    plt.xlabel('Datasets', fontsize=14, fontweight='bold')
    plt.ylabel('Tour Cost', fontsize=14, fontweight='bold')
    plt.title('Algorithm Performance Comparison: Tour Cost', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig('figures/cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Runtime Comparison
    plt.figure(figsize=(14, 8))
    
    for alg in algorithms:
        time_col = f'{alg}_time'
        if time_col in df.columns:
            valid_data = df[df[time_col].notna()]
            if not valid_data.empty:
                plt.plot(valid_data['dataset'], valid_data[time_col], 
                        'o-', linewidth=3, markersize=8, 
                        color=colors.get(alg, 'gray'), 
                        label=alg.upper(), alpha=0.8)
                
                # Add value labels
                for _, row in valid_data.iterrows():
                    plt.annotate(f'{row[time_col]:.2f}s', 
                               (row['dataset'], row[time_col]), 
                               textcoords="offset points", 
                               xytext=(0, 10), ha='center', fontsize=9)
    
    plt.xlabel('Datasets', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
    plt.title('Algorithm Performance Comparison: Runtime', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('figures/runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Scalability Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Runtime vs Problem Size
    for alg in algorithms:
        time_col = f'{alg}_time'
        cities_col = f'{alg}_cities'
        if time_col in df.columns and cities_col in df.columns:
            valid_data = df[df[time_col].notna() & df[cities_col].notna()]
            if not valid_data.empty:
                ax1.loglog(valid_data[cities_col], valid_data[time_col], 
                          'o-', linewidth=3, markersize=8, 
                          color=colors.get(alg, 'gray'), 
                          label=alg.upper(), alpha=0.8)
    
    ax1.set_xlabel('Problem Size (Cities)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Scalability: Runtime vs Problem Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Cost vs Problem Size
    for alg in algorithms:
        cost_col = f'{alg}_cost'
        cities_col = f'{alg}_cities'
        if cost_col in df.columns and cities_col in df.columns:
            valid_data = df[df[cost_col].notna() & df[cities_col].notna()]
            if not valid_data.empty:
                ax2.loglog(valid_data[cities_col], valid_data[cost_col], 
                          'o-', linewidth=3, markersize=8, 
                          color=colors.get(alg, 'gray'), 
                          label=alg.upper(), alpha=0.8)
    
    ax2.set_xlabel('Problem Size (Cities)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tour Cost', fontsize=12, fontweight='bold')
    ax2.set_title('Solution Quality vs Problem Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

if __name__ == '__main__':
    print("Loading experimental results...")
    results = load_results()
    print("Creating visualization plots...")
    df = create_comparison_plots(results)
    print("Plots saved to figures/ directory!")
    
    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))