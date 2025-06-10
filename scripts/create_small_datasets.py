import os
import numpy as np

def create_test_dataset(name, n_cities, pattern='random'):
    """작은 테스트 데이터셋 생성"""
    
    os.makedirs('data', exist_ok=True)
    
    if pattern == 'random':
        # 랜덤 분포
        coords = np.random.uniform(0, 100, (n_cities, 2))
    elif pattern == 'circle':
        # 원형 분포
        angles = np.linspace(0, 2*np.pi, n_cities, endpoint=False)
        radius = 50 + np.random.normal(0, 5, n_cities)  # 약간의 노이즈
        coords = np.column_stack([
            50 + radius * np.cos(angles),
            50 + radius * np.sin(angles)
        ])
    elif pattern == 'cluster':
        # 클러스터 분포
        n_clusters = max(2, n_cities // 4)
        cluster_centers = np.random.uniform(20, 80, (n_clusters, 2))
        
        coords = []
        cities_per_cluster = n_cities // n_clusters
        
        for i, center in enumerate(cluster_centers):
            if i == len(cluster_centers) - 1:
                # 마지막 클러스터는 남은 모든 도시
                n_in_cluster = n_cities - len(coords)
            else:
                n_in_cluster = cities_per_cluster
                
            cluster_coords = np.random.normal(center, 5, (n_in_cluster, 2))
            coords.extend(cluster_coords)
        
        coords = np.array(coords[:n_cities])  # 정확한 개수 맞추기
    
    # TSP 파일 형식으로 저장
    filepath = f'data/{name}.tsp'
    with open(filepath, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: Test dataset with {n_cities} cities ({pattern} pattern)\n")
        f.write(f"DIMENSION: {n_cities}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        
        f.write("EOF\n")
    
    print(f"✅ Created {name}: {n_cities} cities ({pattern}) -> {filepath}")
    return filepath

def main():
    """다양한 크기와 패턴의 테스트 데이터셋 생성"""
    
    print("🏗️ Creating small test datasets...")
    
    # 작은 데이터셋들
    datasets = [
        ('tiny8', 8, 'random'),
        ('small12', 12, 'circle'),
        ('medium15', 15, 'cluster'),
        ('test20', 20, 'random'),
        ('circle16', 16, 'circle'),
        ('cluster25', 25, 'cluster'),
    ]
    
    created_files = []
    
    for name, n_cities, pattern in datasets:
        try:
            filepath = create_test_dataset(name, n_cities, pattern)
            created_files.append(filepath)
        except Exception as e:
            print(f"❌ Error creating {name}: {e}")
    
    print(f"\n📊 Created {len(created_files)} test datasets:")
    for filepath in created_files:
        print(f"   - {filepath}")
    
    print(f"\n🚀 Ready for learning experiments!")

if __name__ == '__main__':
    main()
