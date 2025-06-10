import numpy as np
import matplotlib.pyplot as plt
from utils import compute_dist_matrix

class LearningUTSP:
    """
    학습 기반 UTSP 알고리즘
    - 강화학습으로 경로 선택 정책 개선
    - 경험을 통한 파라미터 자동 조정
    - 적응형 온도 스케줄링
    """
    
    def __init__(self, n_cities, learning_rate=0.01, initial_temperature=15.0):
        self.n_cities = n_cities
        self.lr = learning_rate
        self.initial_temp = initial_temperature
        self.temperature = initial_temperature
        
        # 학습 가능한 파라미터들
        self.preference_matrix = np.zeros((n_cities, n_cities))  # 경로 선호도
        self.visit_counts = np.zeros((n_cities, n_cities))       # 방문 횟수
        self.success_rates = np.ones((n_cities, n_cities)) * 0.5 # 성공률
        
        # 학습 기록
        self.cost_history = []
        self.temperature_history = []
        
    def compute_learned_heatmap(self, coords):
        """학습된 선호도를 반영한 heat map"""
        n = len(coords)
        D = compute_dist_matrix(coords)
        
        # 기본 거리 기반 가중치
        base_weights = np.exp(-D / self.temperature)
        
        # 학습된 선호도 적용
        learned_bonus = np.exp(self.preference_matrix / 5.0)
        
        # 성공률 기반 조정
        success_factor = 0.5 + self.success_rates
        
        # 종합 가중치
        W = base_weights * learned_bonus * success_factor
        np.fill_diagonal(W, 0)
        
        # 정규화 (Sinkhorn 스타일)
        for _ in range(3):  # 몇 번 반복
            W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
            W = W / (W.sum(axis=0, keepdims=True) + 1e-8)
        
        return W
    
    def generate_tour_probabilistic(self, coords):
        """확률적 tour 생성 (탐험과 활용의 균형)"""
        n = len(coords)
        T = self.compute_learned_heatmap(coords)
        
        visited = set([0])
        tour = [0]
        
        while len(tour) < n:
            current = tour[-1]
            probs = T[current].copy()
            
            # 방문한 도시는 확률 0
            for v in visited:
                probs[v] = 0
            
            # 확률 정규화
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum
                
                # 온도에 따른 확률적 선택
                if self.temperature > 5.0:
                    # 높은 온도: 더 탐험적
                    next_city = np.random.choice(n, p=probs)
                else:
                    # 낮은 온도: 더 탐욕적
                    candidates = np.where(probs > 0)[0]
                    if len(candidates) > 0:
                        # 상위 후보들 중에서 선택
                        top_k = min(3, len(candidates))
                        top_indices = np.argsort(-probs[candidates])[:top_k]
                        next_city = candidates[top_indices[0]]
                    else:
                        next_city = candidates[0]
            else:
                # 방문 가능한 도시 중 아무나
                unvisited = [i for i in range(n) if i not in visited]
                next_city = unvisited[0] if unvisited else 0
            
            tour.append(next_city)
            visited.add(next_city)
        
        tour.append(0)  # 시작점으로 복귀
        return tour
    
    def calculate_tour_cost(self, tour, coords):
        """투어 비용 계산"""
        D = compute_dist_matrix(coords)
        return sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
    
    def update_learning_parameters(self, tour, coords, cost, best_cost):
        """경험을 통한 학습 파라미터 업데이트"""
        # 보상 계산 (좋은 해일수록 높은 보상)
        if best_cost > 0:
            reward = max(0, (best_cost - cost) / best_cost)
        else:
            reward = 0
        
        # 사용된 경로들 업데이트
        for i in range(len(tour) - 1):
            from_city = tour[i]
            to_city = tour[i + 1]
            
            # 방문 횟수 증가
            self.visit_counts[from_city, to_city] += 1
            
            # 선호도 업데이트
            if cost <= best_cost:
                # 좋은 경로 강화
                self.preference_matrix[from_city, to_city] += self.lr * (1 + reward)
                self.success_rates[from_city, to_city] = min(1.0, 
                    self.success_rates[from_city, to_city] + self.lr * 0.1)
            else:
                # 나쁜 경로 약화
                self.preference_matrix[from_city, to_city] -= self.lr * 0.5
                self.success_rates[from_city, to_city] = max(0.1,
                    self.success_rates[from_city, to_city] - self.lr * 0.05)
    
    def learning_tour(self, coords, n_episodes=100, verbose=False):
        """학습 기반 투어 생성"""
        best_tour = None
        best_cost = float('inf')
        no_improvement_count = 0
        
        if verbose:
            print(f"🧠 Starting learning process for {len(coords)} cities...")
        
        for episode in range(n_episodes):
            # 현재 정책으로 투어 생성
            tour = self.generate_tour_probabilistic(coords)
            cost = self.calculate_tour_cost(tour, coords)
            
            # 최고 기록 갱신
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                no_improvement_count = 0
                if verbose and episode % 10 == 0:
                    print(f"  Episode {episode}: New best cost {cost:.2f}")
            else:
                no_improvement_count += 1
            
            # 학습 파라미터 업데이트
            self.update_learning_parameters(tour, coords, cost, best_cost)
            
            # 기록 저장
            self.cost_history.append(cost)
            self.temperature_history.append(self.temperature)
            
            # 온도 감소 (simulated annealing)
            self.temperature = max(1.0, self.temperature * 0.995)
            
            # 조기 종료 (개선이 없으면)
            if no_improvement_count > 20:
                if verbose:
                    print(f"  Early stopping at episode {episode}")
                break
        
        if verbose:
            print(f"🎯 Learning completed! Best cost: {best_cost:.2f}")
        
        return best_tour, best_cost
    
    def plot_learning_progress(self, save_path=None):
        """학습 과정 시각화"""
        if not self.cost_history:
            print("No learning history to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 비용 변화
        ax1.plot(self.cost_history, alpha=0.7, linewidth=1)
        ax1.plot(np.convolve(self.cost_history, np.ones(10)/10, mode='valid'), 
                'r-', linewidth=2, label='Moving Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Tour Cost')
        ax1.set_title('Learning Progress: Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 온도 변화
        ax2.plot(self.temperature_history, 'g-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Annealing')
        ax2.grid(True, alpha=0.3)
        
        # 3. 선호도 매트릭스 히트맵
        im = ax3.imshow(self.preference_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_xlabel('To City')
        ax3.set_ylabel('From City')
        ax3.set_title('Learned Preferences')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning progress plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

# 편의 함수
def utsp_learning_tour(coords, n_episodes=50, verbose=False):
    """학습 UTSP의 간단한 인터페이스"""
    learner = LearningUTSP(len(coords))
    tour, cost = learner.learning_tour(coords, n_episodes, verbose)
    return tour, learner

# 성능 비교 함수
def compare_utsp_variants(coords, n_runs=5):
    """기본 UTSP vs 학습 UTSP 비교"""
    from utsp_variant import utsp_variant_tour
    
    print(f"🆚 Comparing UTSP variants on {len(coords)} cities...")
    
    # 기본 UTSP
    basic_costs = []
    for _ in range(n_runs):
        tour = utsp_variant_tour(coords)
        D = compute_dist_matrix(coords)
        cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        basic_costs.append(cost)
    
    # 학습 UTSP
    learning_costs = []
    for _ in range(n_runs):
        tour, _ = utsp_learning_tour(coords, n_episodes=30)
        D = compute_dist_matrix(coords)
        cost = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
        learning_costs.append(cost)
    
    print(f"📊 Results ({n_runs} runs each):")
    print(f"   Basic UTSP:    {np.mean(basic_costs):.2f} ± {np.std(basic_costs):.2f}")
    print(f"   Learning UTSP: {np.mean(learning_costs):.2f} ± {np.std(learning_costs):.2f}")
    print(f"   Improvement:   {(np.mean(basic_costs) - np.mean(learning_costs)):.2f} ({((np.mean(basic_costs) - np.mean(learning_costs))/np.mean(basic_costs)*100):+.1f}%)")
    
    return {
        'basic': basic_costs,
        'learning': learning_costs
    }
