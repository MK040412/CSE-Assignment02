import numpy as np
import torch
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import deque

class ExperienceBuffer:
    """경험 버퍼"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.stack([torch.FloatTensor(x[0]) for x in batch])
        actions = torch.LongTensor([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.stack([torch.FloatTensor(x[3]) for x in batch])
        dones = torch.BoolTensor([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class TSPMetrics:
    """TSP 성능 메트릭"""
    
    @staticmethod
    def compute_optimality_gap(found_cost: float, optimal_cost: float) -> float:
        """최적성 갭 계산"""
        return (found_cost - optimal_cost) / optimal_cost * 100
    
    @staticmethod
    def compute_tour_statistics(coords: List[Tuple], tour: List[int]) -> Dict:
        """투어 통계 계산"""
        n_cities = len(coords)
        
        # 투어 길이
        total_distance = 0
        for i in range(len(tour) - 1):
            city1, city2 = tour[i], tour[i + 1]
            distance = np.linalg.norm(
                np.array(coords[city1]) - np.array(coords[city2])
            )
            total_distance += distance
        
        # 평균 구간 거리
        avg_segment_distance = total_distance / n_cities
        
        # 최장/최단 구간
        segment_distances = []
        for i in range(len(tour) - 1):
            city1, city2 = tour[i], tour[i + 1]
            distance = np.linalg.norm(
                np.array(coords[city1]) - np.array(coords[city2])
            )
            segment_distances.append(distance)
        
        return {
            'total_distance': total_distance,
            'avg_segment_distance': avg_segment_distance,
            'max_segment_distance': max(segment_distances),
            'min_segment_distance': min(segment_distances),
            'std_segment_distance': np.std(segment_distances)
        }
    
    @staticmethod
    def visualize_tour_comparison(coords: List[Tuple], tours: Dict[str, List[int]], 
                                save_path: str = None):
        """투어 비교 시각화"""
        n_methods = len(tours)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        coords_array = np.array(coords)
        
        for idx, (method_name, tour) in enumerate(tours.items()):
            ax = axes[idx]
            
            # 도시들 표시
            ax.scatter(coords_array[:, 0], coords_array[:, 1], 
                      c='red', s=50, zorder=5)
            
            # 투어 경로 표시
            for i in range(len(tour) - 1):
                start_idx, end_idx = tour[i], tour[i + 1]
                ax.plot([coords_array[start_idx, 0], coords_array[end_idx, 0]], 
                       [coords_array[start_idx, 1], coords_array[end_idx, 1]], 
                       'b-', alpha=0.7, linewidth=1.5)
            
            # 시작점 강조
            ax.scatter(coords_array[0, 0], coords_array[0, 1], 
                      c='green', s=100, zorder=6, marker='s')
            
            # 통계 계산
            stats = TSPMetrics.compute_tour_statistics(coords, tour)
            
            ax.set_title(f'{method_name}\nDistance: {stats["total_distance"]:.2f}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tour comparison saved to {save_path}")
        
        plt.show()