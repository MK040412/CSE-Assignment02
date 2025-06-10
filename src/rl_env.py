import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

class TSPEnvironment(gym.Env):
    """TSP를 위한 강화학습 환경"""
    
    def __init__(self, coords: list, max_steps: int = None):
        super().__init__()
        self.coords = np.array(coords)
        self.n_cities = len(coords)
        self.max_steps = max_steps or self.n_cities
        
        # 거리 행렬 미리 계산
        self.dist_matrix = self._compute_distance_matrix()
        
        # Action space: 다음 방문할 도시 선택
        self.action_space = spaces.Discrete(self.n_cities)
        
        # Observation space: [current_pos, unvisited_mask, position_encoding]
        obs_dim = 1 + self.n_cities + self.n_cities * 2  # pos + mask + coords
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """거리 행렬 계산"""
        n = len(self.coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return dist_matrix
    
    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.current_pos = 0  # 시작점은 항상 0
        self.visited = set([0])
        self.tour = [0]
        self.total_distance = 0.0
        self.step_count = 0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """현재 상태 관찰값 반환"""
        # Current position (one-hot)
        current_pos_onehot = np.zeros(self.n_cities)
        current_pos_onehot[self.current_pos] = 1.0
        
        # Unvisited mask
        unvisited_mask = np.array([1.0 if i not in self.visited else 0.0 
                                  for i in range(self.n_cities)])
        
        # Coordinate features (normalized)
        coord_features = self.coords.flatten()
        coord_features = (coord_features - coord_features.mean()) / (coord_features.std() + 1e-8)
        
        obs = np.concatenate([
            [self.current_pos / self.n_cities],  # normalized current position
            unvisited_mask,
            coord_features
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """환경에서 한 스텝 실행"""
        self.step_count += 1
        
        # Invalid action penalty
        if action in self.visited:
            reward = -10.0  # 이미 방문한 도시 선택시 큰 페널티
            done = False
            info = {'invalid_action': True}
        else:
            # Valid action
            prev_pos = self.current_pos
            self.current_pos = action
            self.visited.add(action)
            self.tour.append(action)
            
            # Distance-based reward (negative)
            step_distance = self.dist_matrix[prev_pos, action]
            self.total_distance += step_distance
            reward = -step_distance / 100.0  # 정규화된 음의 보상
            
            # Episode termination
            done = len(self.visited) == self.n_cities
            if done:
                # Return to start
                final_distance = self.dist_matrix[self.current_pos, 0]
                self.total_distance += final_distance
                reward += -final_distance / 100.0
                # Completion bonus
                reward += 10.0
                self.tour.append(0)
            
            info = {'total_distance': self.total_distance, 'tour_length': len(self.tour)}
        
        # Max steps termination
        if self.step_count >= self.max_steps:
            done = True
            reward -= 5.0  # 시간 초과 페널티
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """투어 시각화"""
        if len(self.tour) < 2:
            return
        
        plt.figure(figsize=(10, 8))
        coords = self.coords
        
        # 도시들 표시
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=5)
        
        # 투어 경로 표시
        for i in range(len(self.tour) - 1):
            start_idx, end_idx = self.tour[i], self.tour[i + 1]
            plt.plot([coords[start_idx, 0], coords[end_idx, 0]], 
                    [coords[start_idx, 1], coords[end_idx, 1]], 'b-', alpha=0.7)
        
        # 도시 번호 표시
        for i, (x, y) in enumerate(coords):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'TSP Tour (Distance: {self.total_distance:.2f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.show()