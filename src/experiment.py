import argparse
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import load_tsp, compute_dist_matrix
from mst_approx import mst_2_approx
from held_karp import held_karp
from utsp_variant import utsp_variant_tour

def run_algorithm(name, coords):
    D = compute_dist_matrix(coords)
    start = time.time()
    try:
        if name == 'mst':
            tour = mst_2_approx(coords)
            cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        elif name == 'heldkarp':
            if len(coords) > 20:  # Skip for large instances
                return None, None
            tour, cost = held_karp(D)
        elif name == 'utsp':
            tour = utsp_variant_tour(coords)
            cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        else:
            raise ValueError('Unknown algorithm')
        return cost, time.time() - start
    except Exception as e:
        print(f"Error in {name}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', type=str, default='mst,utsp')
    parser.add_argument('--datasets', type=str, default='a280')
    args = parser.parse_args()

    algs = args.algorithms.split(',')
    dsets = args.datasets.split(',')
    results = {alg: {} for alg in algs}

    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    for ds in dsets:
        print(f"\nProcessing dataset: {ds}")
        filepath = f"data/{ds}.tsp"
        
        if not os.path.exists(filepath):
            print(f"Dataset {ds} not found. Skipping...")
            continue
            
        coords = load_tsp(filepath)
        print(f"Loaded {len(coords)} cities")
        
        for alg in algs:
            print(f"Running {alg}...")
            cost, rt = run_algorithm(alg, coords)
            if cost is not None:
                results[alg][ds] = {
                    'cost': cost, 
                    'runtime': rt,
                    'cities': len(coords)
                }
                print(f"{alg}: Cost={cost:.1f}, Time={rt:.3f}s")
            else:
                print(f"{alg}: Failed or skipped")

    # Save results to JSON
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/experiment_results.json")
    print("Results summary:")
    for alg in results:
        print(f"\n{alg.upper()}:")
        for ds in results[alg]:
            r = results[alg][ds]
            print(f"  {ds}: {r['cost']:.1f} (cities: {r['cities']}, time: {r['runtime']:.3f}s)")

if __name__ == '__main__':
    main()
