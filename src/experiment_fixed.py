import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_tsp, load_tsp_alternative, compute_dist_matrix
from mst_approx import mst_2_approx
from held_karp import held_karp
from utsp_variant import utsp_variant_tour

def smart_load_tsp(filepath):
    """스마트 TSP 로더 - 여러 파서 시도"""
    # 먼저 기본 파서 시도
    coords = load_tsp(filepath)
    
    if not coords:
        print(f"      🔄 Trying alternative parser...")
        coords = load_tsp_alternative(filepath)
    
    if not coords:
        print(f"      🔄 Trying manual parsing...")
        coords = manual_parse_tsp(filepath)
    
    return coords

def manual_parse_tsp(filepath):
    """수동 TSP 파싱 - 특정 형식에 대응"""
    coords = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"      ❌ Encoding error: {e}")
            return []
    
    # 다양한 형식 시도
    data_started = False
    
    for line in lines:
        line = line.strip()
        
        # 섹션 시작 감지
        if any(keyword in line.upper() for keyword in 
               ['NODE_COORD_SECTION', 'DISPLAY_DATA_SECTION', 'DATA_SECTION']):
            data_started = True
            continue
        
        # 섹션 종료 감지
        if line.upper() in ['EOF', 'EDGE_WEIGHT_SECTION', '']:
            if line.upper() == 'EOF':
                break
            continue
        
        # 데이터 파싱
        if data_started and line:
            parts = line.split()
            
            # 최소 2개 숫자 필요 (x, y 좌표)
            if len(parts) >= 2:
                try:
                    # 다양한 형식 시도
                    if len(parts) == 2:  # x y
                        x, y = float(parts[0]), float(parts[1])
                    elif len(parts) == 3:  # id x y
                        _, x, y = parts
                        x, y = float(x), float(y)
                    else:  # 더 많은 필드가 있는 경우 마지막 두 개를 좌표로
                        x, y = float(parts[-2]), float(parts[-1])
                    
                    coords.append((x, y))
                    
                except (ValueError, IndexError):
                    continue
    
    # 여전히 빈 경우, 숫자만 있는 라인들을 찾아보기
    if not coords:
        numeric_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(char.isalpha() for char in line):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        nums = [float(p) for p in parts]
                        if len(nums) >= 2:
                            numeric_lines.append(nums)
                    except ValueError:
                        continue
        
        # 숫자 라인들을 좌표로 해석
        for nums in numeric_lines:
            if len(nums) == 2:
                coords.append((nums[0], nums[1]))
            elif len(nums) >= 3:
                coords.append((nums[1], nums[2]))  # 첫 번째는 ID로 가정
    
    return coords

def run_algorithm_safe(name, coords):
    """안전한 알고리즘 실행"""
    if not coords:
        return float('inf'), 0, []
    
    try:
        D = compute_dist_matrix(coords)
        start = time.time()
        
        if name == 'mst':
            tour = mst_2_approx(coords)
            cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        elif name == 'heldkarp':
            if len(coords) > 15:  # 15개로 제한 완화
                print(f"      ⚠️  Held-Karp skipped: {len(coords)} cities too large (max 15)")
                return float('inf'), 0, []
            tour, cost = held_karp(D)
        elif name == 'utsp':
            tour = utsp_variant_tour(coords)
            cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        else:
            raise ValueError(f'Unknown algorithm: {name}')
        
        return cost, time.time() - start, tour
    
    except Exception as e:
        print(f"      ❌ Error in {name}: {e}")
        return float('inf'), 0, []
def main():
    parser = argparse.ArgumentParser(description='CSE331 Assignment #2 - Fixed TSP Solver')
    parser.add_argument('--algorithms', type=str, default='mst,heldkarp,utsp',
                       help='Algorithms: mst,heldkarp,utsp,sac')
    parser.add_argument('--datasets', type=str, default='a280',
                       help='Datasets: a280,xql662,kz9976,monalisa')
    parser.add_argument('--check-data', action='store_true',
                       help='Check data files first')
    args = parser.parse_args()

    algs = [alg.strip() for alg in args.algorithms.split(',')]
    dsets = [ds.strip() for ds in args.datasets.split(',')]
    
    print("=" * 60)
    print("CSE331 Assignment #2 - Fixed TSP Solver")
    print("=" * 60)
    
    # 데이터 확인
    if args.check_data:
        print("🔍 Checking data files first...")
        for ds in dsets:
            filepath = f"data/{ds}.tsp"
            coords = smart_load_tsp(filepath)
            print(f"   {ds}: {len(coords)} cities loaded")
        print()

    # 실험 실행
    results = {alg: {} for alg in algs}
    
    for ds in dsets:
        print(f"\n🔍 Processing dataset: {ds}")
        
        filepath = f"data/{ds}.tsp"
        coords = smart_load_tsp(filepath)
        
        if coords:
            print(f"   ✅ Loaded {len(coords)} cities")
        else:
            print(f"   ❌ Failed to load data")
            continue

        for alg in algs:
            print(f"   🚀 Running {alg}...")
            cost, runtime, tour = run_algorithm_safe(alg, coords)
            results[alg][ds] = (cost, runtime, tour)
            
            if cost != float('inf'):
                print(f"      ✅ Cost: {cost:.2f}, Time: {runtime:.4f}s")
            else:
                print(f"      ⏩ Skipped or failed")

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)
    
    for ds in dsets:
        print(f"\n{ds}:")
        for alg in algs:
            if ds in results[alg]:
                cost, time_val, _ = results[alg][ds]
                if cost != float('inf'):
                    print(f"  {alg:>10}: {cost:>10.2f} ({time_val:>6.3f}s)")
                else:
                    print(f"  {alg:>10}: {'FAILED':>10}")

if __name__ == '__main__':
    main()