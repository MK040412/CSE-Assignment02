import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_tsp, compute_dist_matrix
from mst_approx import mst_2_approx
from held_karp import held_karp
from utsp_variant import utsp_variant_tour

def run_algorithm(name, coords):
    """알고리즘 실행"""
    D = compute_dist_matrix(coords)
    start = time.time()
    
    try:
        if name == 'mst':
            tour = mst_2_approx(coords)
            cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        elif name == 'heldkarp':
            # Held-Karp는 작은 데이터셋에서만 실행 (지수 복잡도 때문)
            if len(coords) > 20:
                print(f"    ⚠️  Held-Karp skipped: {len(coords)} cities too large (max 20)")
                return float('inf'), 0, []
            tour, cost = held_karp(D)
        elif name == 'utsp':
            tour = utsp_variant_tour(coords)
            cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        else:
            raise ValueError(f'Unknown algorithm: {name}')
        
        return cost, time.time() - start, tour
    
    except Exception as e:
        print(f"    ❌ Error in {name}: {e}")
        return float('inf'), 0, []

def create_detailed_plots(results, datasets):
    """상세한 시각화 생성"""
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 비용 비교
    plt.subplot(2, 3, 1)
    algs = list(results.keys())
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.25
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, alg in enumerate(algs):
        costs = []
        for ds in datasets:
            if ds in results[alg] and results[alg][ds][0] != float('inf'):
                costs.append(results[alg][ds][0])
            else:
                costs.append(0)  # Skip if not available
        
        plt.bar(x + i * width, costs, width, label=alg, color=colors[i % len(colors)])
    
    plt.xlabel('Datasets')
    plt.ylabel('Tour Cost')
    plt.title('Tour Cost Comparison')
    plt.xticks(x + width, datasets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 실행 시간 비교 (로그 스케일)
    plt.subplot(2, 3, 2)
    for i, alg in enumerate(algs):
        times = []
        for ds in datasets:
            if ds in results[alg] and results[alg][ds][1] > 0:
                times.append(results[alg][ds][1])
            else:
                times.append(0.001)  # 최소값으로 설정
        
        plt.bar(x + i * width, times, width, label=alg, color=colors[i % len(colors)])
    
    plt.xlabel('Datasets')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(x + width, datasets, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 성능 비율 (MST 대비)
    plt.subplot(2, 3, 3)
    if 'mst' in results:
        for i, alg in enumerate(algs):
            if alg != 'mst':
                ratios = []
                for ds in datasets:
                    if (ds in results['mst'] and ds in results[alg] and 
                        results['mst'][ds][0] != float('inf') and results[alg][ds][0] != float('inf')):
                        mst_cost = results['mst'][ds][0]
                        alg_cost = results[alg][ds][0]
                        ratios.append(alg_cost / mst_cost)
                    else:
                        ratios.append(1.0)
                
                plt.bar(x + i * width, ratios, width, label=alg, 
                       color=colors[i % len(colors)])
        
        plt.xlabel('Datasets')
        plt.ylabel('Cost Ratio (vs MST)')
        plt.title('Performance Ratio vs MST')
        plt.xticks(x + width, datasets, rotation=45)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='MST baseline')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 데이터셋 크기별 성능
    plt.subplot(2, 3, 4)
    dataset_sizes = []
    for ds in datasets:
        try:
            coords = load_tsp(f'data/{ds}.tsp')
            dataset_sizes.append(len(coords))
        except:
            dataset_sizes.append(0)
    
    for alg in algs:
        costs = []
        sizes = []
        for i, ds in enumerate(datasets):
            if ds in results[alg] and results[alg][ds][0] != float('inf'):
                costs.append(results[alg][ds][0])
                sizes.append(dataset_sizes[i])
        
        if costs:
            plt.scatter(sizes, costs, label=alg, s=50, alpha=0.7)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Tour Cost')
    plt.title('Scalability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 5: 알고리즘 효율성 (cost/time)
    plt.subplot(2, 3, 5)
    for alg in algs:
        efficiency = []
        for ds in datasets:
            if (ds in results[alg] and results[alg][ds][0] != float('inf') and 
                results[alg][ds][1] > 0):
                # 효율성 = 1 / (정규화된 비용 * 시간)
                normalized_cost = results[alg][ds][0] / 10000  # 스케일링
                efficiency.append(1 / (normalized_cost * results[alg][ds][1] + 1))
            else:
                efficiency.append(0)
        
        plt.bar(x + algs.index(alg) * width, efficiency, width, 
               label=alg, color=colors[algs.index(alg) % len(colors)])
    
    plt.xlabel('Datasets')
    plt.ylabel('Efficiency Score')
    plt.title('Algorithm Efficiency')
    plt.xticks(x + width, datasets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 6: 요약 통계
    plt.subplot(2, 3, 6)
    summary_data = []
    labels = []
    
    for alg in algs:
        valid_costs = []
        valid_times = []
        for ds in datasets:
            if ds in results[alg] and results[alg][ds][0] != float('inf'):
                valid_costs.append(results[alg][ds][0])
                valid_times.append(results[alg][ds][1])
        
        if valid_costs:
            avg_cost = np.mean(valid_costs)
            avg_time = np.mean(valid_times)
            summary_data.append([avg_cost/10000, avg_time*1000])  # 스케일링
            labels.append(alg)
    
    if summary_data:
        summary_data = np.array(summary_data)
        x_pos = np.arange(len(labels))
        
        plt.bar(x_pos - 0.2, summary_data[:, 0], 0.4, label='Avg Cost (×10k)', alpha=0.7)
        plt.bar(x_pos + 0.2, summary_data[:, 1], 0.4, label='Avg Time (×1000s)', alpha=0.7)
        
        plt.xlabel('Algorithms')
        plt.ylabel('Scaled Values')
        plt.title('Average Performance Summary')
        plt.xticks(x_pos, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_basic_plots(results, datasets):
    """기본 시각화 생성"""
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 비용 비교
    plt.subplot(1, 2, 1)
    algs = list(results.keys())
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.25
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, alg in enumerate(algs):
        costs = []
        for ds in datasets:
            if ds in results[alg] and results[alg][ds][0] != float('inf'):
                costs.append(results[alg][ds][0])
            else:
                costs.append(0)
        
        plt.bar(x + i * width, costs, width, label=alg, color=colors[i % len(colors)])
    
    plt.xlabel('Datasets')
    plt.ylabel('Tour Cost')
    plt.title('Tour Cost Comparison')
    plt.xticks(x + width, datasets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 실행 시간 비교
    plt.subplot(1, 2, 2)
    for i, alg in enumerate(algs):
        times = []
        for ds in datasets:
            if ds in results[alg] and results[alg][ds][1] > 0:
                times.append(results[alg][ds][1])
            else:
                times.append(0.001)
        
        plt.bar(x + i * width, times, width, label=alg, color=colors[i % len(colors)])
    
    plt.xlabel('Datasets')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(x + width, datasets, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/basic_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_summary(results, datasets):
    """결과 요약 저장"""
    os.makedirs('results', exist_ok=True)
    
    # 텍스트 리포트 생성
    with open('results/experiment_report.txt', 'w') as f:
        f.write("CSE331 Assignment #2 - TSP Solver Comparison Report\n")
        f.write("=" * 55 + "\n\n")
        
        f.write(f"Datasets tested: {', '.join(datasets)}\n")
        f.write(f"Algorithms compared: {', '.join(results.keys())}\n\n")
        
        # 데이터셋별 결과
        for dataset in datasets:
            f.write(f"\n{dataset} Results:\n")
            f.write("-" * 20 + "\n")
            
            try:
                coords = load_tsp(f'data/{dataset}.tsp')
                f.write(f"Number of cities: {len(coords)}\n")
            except:
                f.write("Number of cities: Unknown\n")
            
            for alg in results:
                if dataset in results[alg]:
                    cost, time_val, _ = results[alg][dataset]
                    if cost != float('inf'):
                        f.write(f"{alg:>10}: Cost={cost:>10.2f}, Time={time_val:>8.4f}s\n")
                    else:
                        f.write(f"{alg:>10}: SKIPPED/FAILED\n")
        
        # 알고리즘 분석
        f.write(f"\n\nAlgorithm Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write("MST 2-Approximation:\n")
        f.write("  - Time Complexity: O(n²)\n")
        f.write("  - Approximation Ratio: ≤ 2\n")
        f.write("  - Pros: Fast, guaranteed approximation\n")
        f.write("  - Cons: Not optimal\n\n")
        
        f.write("Held-Karp Dynamic Programming:\n")
        f.write("  - Time Complexity: O(n²2ⁿ)\n")
        f.write("  - Approximation Ratio: Optimal\n")
        f.write("  - Pros: Exact solution\n")
        f.write("  - Cons: Exponential time, memory intensive\n\n")
 
        f.write("UTSP Variant (Novel Algorithm):\n")
        f.write("  - Time Complexity: O(n³)\n")
        f.write("  - Approximation Ratio: Heuristic\n")
        f.write("  - Pros: Heat-map guided selection\n")
        f.write("  - Cons: No theoretical guarantees\n\n")

def main():
        parser = argparse.ArgumentParser(description='CSE331 Assignment #2 - TSP Solver')
        parser.add_argument('--algorithms', type=str, default='mst,heldkarp,utsp',
                           help='Comma-separated list of algorithms: mst,heldkarp,utsp')
        parser.add_argument('--datasets', type=str, default='a280',
                           help='Comma-separated list of datasets')
        parser.add_argument('--detailed', action='store_true',
                           help='Generate detailed analysis plots')
        args = parser.parse_args()

        algs = [alg.strip() for alg in args.algorithms.split(',')]
        dsets = [ds.strip() for ds in args.datasets.split(',')]
    
        print("=" * 60)
        print("CSE331 Assignment #2 - TSP Solver Experiments")
        print("=" * 60)
        print(f"Algorithms: {', '.join(algs)}")
        print(f"Datasets: {', '.join(dsets)}")
        print("=" * 60)

        results = {alg: {} for alg in algs}

        for ds in dsets:
            print(f"\n🔍 Processing dataset: {ds}")
        
            filepath = f"data/{ds}.tsp"
            try:
                coords = load_tsp(filepath)
                print(f"   📍 Loaded {len(coords)} cities")
            except Exception as e:
                print(f"   ❌ Failed to load {ds}: {e}")
                continue

            for alg in algs:
                print(f"   🚀 Running {alg}...")
                cost, runtime, tour = run_algorithm(alg, coords)
                results[alg][ds] = (cost, runtime, tour)
            
                if cost != float('inf'):
                    print(f"      ✅ Cost: {cost:.2f}, Time: {runtime:.4f}s")
                else:
                    print(f"      ⏩ Skipped or failed")

        print("\n" + "=" * 60)
        print("📊 Generating visualizations...")
    
        if args.detailed:
            create_detailed_plots(results, dsets)
            print("   ✅ Detailed plots saved to figures/comprehensive_comparison.png")
        else:
            create_basic_plots(results, dsets)
            print("   ✅ Basic plots saved to figures/basic_comparison.png")
    
        print("💾 Saving results summary...")
        save_results_summary(results, dsets)
        print("   ✅ Report saved to results/experiment_report.txt")
    
        # 결과 요약 출력
        print("\n" + "=" * 60)
        print("📋 EXPERIMENT SUMMARY")
        print("=" * 60)
    
        for ds in dsets:
            print(f"\n{ds}:")
            for alg in algs:
                if ds in results[alg]:
                    cost, time_val, _ = results[alg][ds]
                    if cost != float('inf'):
                        print(f"  {alg:>10}: {cost:>10.2f} (in {time_val:>6.3f}s)")
                    else:
                        print(f"  {alg:>10}: {'SKIPPED':>10}")
    
        print("\n" + "=" * 60)
        print("✅ All experiments completed!")
        print("📁 Check 'figures/' and 'results/' directories for outputs")
        print("=" * 60)

if __name__ == '__main__':
        main()
