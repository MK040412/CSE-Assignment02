import os
import requests
import gzip
import shutil
import numpy as np

# 작동하는 데이터셋 URL들로 교체
DATASETS = {
    'a280': 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/a280.tsp.gz',
    'xql662': 'https://www.math.uwaterloo.ca/tsp/vlsi/xql662.tsp',
    # 문제가 있는 URL들을 대체
    'berlin52': 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp.gz',
    'pr76': 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/pr76.tsp.gz',
    'rat195': 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/rat195.tsp.gz',
    'pcb442': 'http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/pcb442.tsp.gz'
}

def create_synthetic_tsp(name, n_cities):
    """합성 TSP 데이터 생성"""
    print(f"Creating synthetic {name} with {n_cities} cities...")
    
    if name == 'kz9976_synthetic':
        # 카자흐스탄 스타일 (넓은 지역)
        np.random.seed(9976)
        coords = []
        # 주요 도시들 (클러스터)
        centers = [(100, 100), (300, 150), (200, 300), (400, 250), (150, 400)]
        for center in centers:
            cluster_size = n_cities // len(centers)
            for _ in range(cluster_size):
                x = center[0] + np.random.normal(0, 30)
                y = center[1] + np.random.normal(0, 30)
                coords.append((x, y))
        
        # 나머지 랜덤 분포
        remaining = n_cities - len(coords)
        for _ in range(remaining):
            x = np.random.uniform(0, 500)
            y = np.random.uniform(0, 500)
            coords.append((x, y))
    
    elif name == 'monalisa_synthetic':
        # 모나리자 스타일 (예술적 분포)
        np.random.seed(100000)
        coords = []
        
        # 타원형 분포 (얼굴 윤곽)
        for i in range(n_cities):
            angle = 2 * np.pi * i / n_cities
            # 얼굴 모양 근사
            if i < n_cities // 3:  # 얼굴 윤곽
                a, b = 150, 200  # 타원 반지름
                x = a * np.cos(angle) + 200 + np.random.normal(0, 10)
                y = b * np.sin(angle) + 200 + np.random.normal(0, 10)
            elif i < 2 * n_cities // 3:  # 내부 특징
                x = np.random.normal(200, 50)
                y = np.random.normal(200, 60)
            else:  # 배경
                x = np.random.uniform(50, 350)
                y = np.random.uniform(50, 350)
            
            coords.append((max(0, x), max(0, y)))
    
    return coords

def save_tsp_file(coords, filepath, name):
    """TSP 파일 형식으로 저장"""
    with open(filepath, 'w') as f:
        f.write(f"NAME : {name}\n")
        f.write(f"COMMENT : Synthetic TSP data for CSE331 Assignment #2\n")
        f.write(f"TYPE : TSP\n")
        f.write(f"DIMENSION : {len(coords)}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        
        f.write("EOF\n")

def download_with_fallback():
    """데이터 다운로드 (실패 시 대체 데이터 생성)"""
    os.makedirs('data', exist_ok=True)
    
    print("=" * 60)
    print("CSE331 Assignment #2 - TSP Data Downloader")
    print("=" * 60)
    
    for name, url in DATASETS.items():
        out_path = f"data/{name}.tsp"
        
        try:
            print(f"\n🌐 Downloading {name}...")
            
            if url.endswith('.gz'):
                gz_path = out_path + '.gz'
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                
                with open(gz_path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                
                # 압축 해제
                with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
            else:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                
                # HTML 응답 확인
                if '<html>' in r.text.lower() or 'forbidden' in r.text.lower():
                    raise requests.exceptions.HTTPError("Received HTML instead of TSP data")
                
                with open(out_path, 'wb') as f:
                    f.write(r.content)
            
            print(f"   ✅ Successfully saved to {out_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to download {name}: {e}")
            print(f"   🔄 Creating synthetic data...")
            
            # 합성 데이터 생성
            if name in ['kz9976', 'monalisa']:
                if name == 'kz9976':
                    coords = create_synthetic_tsp('kz9976_synthetic', 100)  # 작은 크기로
                else:  # monalisa
                    coords = create_synthetic_tsp('monalisa_synthetic', 200)
                
                save_tsp_file(coords, out_path, name)
                print(f"   ✅ Synthetic {name} created with {len(coords)} cities")
            else:
                print(f"   ⚠️  No fallback available for {name}")

if __name__ == '__main__':
    download_with_fallback()
