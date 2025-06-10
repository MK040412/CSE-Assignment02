"""
Soft Actor-Critic (SAC) for TSP - Advanced RL approach
This is a bonus implementation for extra credit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Tuple
import os

from .rl_env import TSPEnvironment
from .utils import compute_dist_matrix

class TSPCallback(BaseCallback):
    """SAC 학습 중 성능 모니터링 콜백"""
    
    def __init__(self, eval_coords: list, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_coords = eval_coords
        self.eval_freq = eval_freq
        self.best_distance = float('inf')
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 평가 수행
            env = TSPEnvironment(self.eval_coords)
            model = self.model
            
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
            
            distance = info.get('total_distance', float('inf'))
            if distance < self.best_distance:
                self.best_distance = distance
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: New best distance: {distance:.2f}")
        
        return True

class SACTSPSolver:
    """SAC를 사용한 TSP 솔버"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        
    def train(self, coords: list, total_timesteps: int = 100000, 
              save_path: str = None) -> None:
        """SAC 모델 학습"""
        
        # 환경 생성
        def make_env():
            return TSPEnvironment(coords)
        
        env = make_vec_env(make_env, n_envs=1)
        
        # SAC 모델 생성
        self.model = SAC(
            "MlpPolicy", 
            env,
            learning_rate=3e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/sac_tsp/"
        )
        
        # 콜백 설정
        callback = TSPCallback(coords, eval_freq=5000)
        
        # 학습 실행
        print(f"Starting SAC training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="SAC_TSP"
        )
        
        # 모델 저장
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """학습된 모델 로드"""
        self.model = SAC.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def solve(self, coords: list, deterministic: bool = True) -> Tuple[List[int], float]:
        """TSP 문제 해결"""
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.load_model(self.model_path)
            else:
                raise ValueError("No trained model available. Train first or provide model path.")
        
        env = TSPEnvironment(coords)
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, _, done, info = env.step(action)
        
        tour = env.tour
        total_distance = info.get('total_distance', 0)
        
        return tour, total_distance
    
    def solve_multiple_runs(self, coords: list, n_runs: int = 10) -> Tuple[List[int], float]:
        """여러 번 실행하여 최적 해 찾기"""
        best_tour = None
        best_distance = float('inf')
        
        for _ in range(n_runs):
            tour, distance = self.solve(coords, deterministic=False)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
        
        return best_tour, best_distance

# UTSP + SAC 하이브리드 접근법
class UTSPSACHybrid:
    """UTSP 휴리스틱과 SAC를 결합한 하이브리드 접근법"""
    
    def __init__(self, sac_solver: SACTSPSolver):
        self.sac_solver = sac_solver
    
    def solve(self, coords: list) -> Tuple[List[int], float]:
        """UTSP 초기해 + SAC 개선"""
        from .utsp_variant import utsp_variant_tour
        
        # 1. UTSP로 초기 해 생성
        utsp_tour = utsp_variant_tour(coords)
        D = compute_dist_matrix(coords)
        utsp_cost = sum(D[utsp_tour[i], utsp_tour[i+1]] for i in range(len(utsp_tour)-1))
        
        # 2. SAC로 해 개선 시도
        try:
            sac_tour, sac_cost = self.sac_solver.solve_multiple_runs(coords, n_runs=5)
            
            # 더 좋은 해 선택
            if sac_cost < utsp_cost:
                return sac_tour, sac_cost
            else:
                return utsp_tour, utsp_cost
        except:
            # SAC 실패시 UTSP 해 반환
            return utsp_tour, utsp_cost
