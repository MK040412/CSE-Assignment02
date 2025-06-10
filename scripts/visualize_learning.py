import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.append('src')

from utils import load_tsp, compute_dist_matrix
from utsp_learning import LearningUTSP

def create_learning_animation(coords, dataset_name, n_episodes=50):
    """학습 과정을 애니메이션으로 시각화"""
    
    learner = LearningUTSP(len(coords))
    
    # 학습 과정 기록
    tours_history = []
    costs_history = []
    preferences_history = []
    
    print(f"🎬 Creating learning animation for {dataset_name}...")
    
    best_cost = float('inf')
    for episode in range(n_episodes):
        tour = learner.generate_tour_probabilistic(coords)
        cost = learner.calculate_tour_cost(tour, coords)
        
        if cost < best_cost:
            best_cost = cost
        
        learner.update_learning_parameters(tour, coords, cost, best_cost)
        learner.temperature = max(1.0, learner.temperature * 0.995)
        
        # 기록 저장 (매 5번째 에피소드)
        if episode % 5 == 0:
            tours_history.append(tour.copy())
            costs_history.append(cost)
            preferences_history.append(learner.preference_matrix.copy())
    
    # 애니메이션 생성
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    def animate(frame):
        # 모든 축 클리어
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        if frame >= len(tours_history):
            frame = len(tours_history) - 1
        
        # 1. 현재 투어 시각화
        tour = tours_history[frame]
        x_coords = [coords[i][0] for i in tour]
        y_coords = [coords[i][1] for i in tour]
        
        ax1.plot(x_coords, y_coords, 'b-o', linewidth=2, markersize=4)
        ax1.scatter([coords[i][0] for i in range(len(coords))], 
                   [coords[i][1] for i in range(len(coords))], 
                   c='red', s=50, alpha=0.7)
        ax1.set_title(f'Tour at Episode {frame*5} (Cost: {costs_history[frame]:.2f})')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. 비용 변화
        ax2.plot(range(0, (frame+1)*5, 5), costs_history[:frame+1], 'g-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Tour Cost')
        ax2.set_title('Learning Progress')
        ax2.grid(True, alpha=0.3)
        
        # 3. 선호도 매트릭스
        im = ax3.imshow(preferences_history[frame], cmap='coolwarm', 
                       vmin=-2, vmax=2, aspect='auto')
        ax3.set_title('Learned Preferences')
        ax3.set_xlabel('To City')
        ax3.set_ylabel('From City')
        
        # 4. 온도 변화
        temperatures = [15.0 * (0.995 ** (i*5)) for i in range(frame+1)]
        ax4.plot(range(0, (frame+1)*5, 5), temperatures, 'r-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Temperature')
        ax4.set_title('Temperature Annealing')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    # 애니메이션 생성 및 저장
    anim = FuncAnimation(fig, animate, frames=len(tours_history), 
                        interval=500, repeat=True)
    
    os.makedirs('figures/animations', exist_ok=True)
    anim_path = f'figures/animations/{dataset_name}_learning.gif'
    anim.save(anim_path, writer='pillow', fps=2)
    
    plt.close()
    print(f"🎬 Animation saved to {anim_path}")

def compare_learning_curves(datasets):
    """여러 데이터셋의 학습 곡선 비교"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, dataset in enumerate(datasets[:4]):
        if i >= len(axes):
            break
            
        print(f"📈 Analyzing learning curve for {dataset}...")
        
        filepath = f"data/{dataset}.tsp"
        if not os.path.exists(filepath):
            continue
            
        coords = load_tsp(filepath)
        learner = LearningUTSP(len(coords))
        
        # 학습 실행
        tour, cost = learner.learning_tour(coords, n_episodes=100, verbose=False)
        
        # 학습 곡선 플롯
        ax = axes[i]
        ax.plot(learner.cost_history, alpha=0.7, color=colors[i], linewidth=1)
        
        # 이동 평균
        window = min(10, len(learner.cost_history) // 4)
        if window > 1:
            moving_avg = np.convolve(learner.cost_history, 
                                   np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(learner.cost_history)), moving_avg, 
                   color=colors[i], linewidth=3, label=f'{dataset} (avg)')
        
        ax.set_title(f'{dataset} ({len(coords)} cities)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Tour Cost')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Learning curves comparison saved to figures/learning_curves_comparison.png")

if __name__ == '__main__':
    # 작은 데이터셋들로 학습 시각화
    datasets = ['tiny8', 'small12', 'medium15']
    
    # 개별 애니메이션 생성
    for dataset in datasets:
        filepath = f"data/{dataset}.tsp"
        if os.path.exists(filepath):
            coords = load_tsp(filepath)
            create_learning_animation(coords, dataset, n_episodes=50)
    
    # 학습 곡선 비교
    compare_learning_curves(datasets)
    
    print("🎯 All learning visualizations completed!")