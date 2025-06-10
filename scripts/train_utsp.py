import argparse
import os
import sys
import numpy as np
sys.path.append('src')

from utsp_gnn import UTSPGNNSolver
from utils import load_tsp
from rl_utils import TSPMetrics

def generate_random_instances(n_cities: int, n_instances: int = 100):
    """랜덤 TSP 인스턴스 생성"""
    instances = []
    for _ in range(n_instances):
        coords = np.random.rand(n_cities, 2) * 100  # 0-100 범위의 좌표
        instances.append(coords.tolist())
    return instances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset')
    parser.add_argument('--n_cities', type=int, default=50, help='Number of cities for random instances')
    parser.add_argument('--n_instances', type=int, default=200, help='Number of training instances')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/utsp_gnn_model.pth', help='Model save path')
    args = parser.parse_args()
    
    # 훈련 데이터 준비
    if args.dataset:
        # 특정 데이터셋 사용
        coords = load_tsp(f'data/{args.dataset}.tsp')
        train_coords_list = [coords]
        print(f"Training on dataset: {args.dataset} ({len(coords)} cities)")
    else:
        # 랜덤 인스턴스 생성
        train_coords_list = generate_random_instances(args.n_cities, args.n_instances)
        print(f"Generated {args.n_instances} random instances with {args.n_cities} cities")
    
    # UTSP 솔버 생성 및 학습
    solver = UTSPGNNSolver()
    solver.train(
        train_coords_list=train_coords_list,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path
    )
    
    # 테스트
    if args.dataset:
        test_coords = load_tsp(f'data/{args.dataset}.tsp')
        tour, cost = solver.solve(test_coords)
        print(f"Test result - Tour cost: {cost:.2f}")
        
        # 통계 출력
        stats = TSPMetrics.compute_tour_statistics(test_coords, tour)
        print(f"Tour statistics: {stats}")
    
    print("UTSP training completed!")

if __name__ == '__main__':
    main()