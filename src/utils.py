import re
import numpy as np

def load_tsp(filepath):
    """
    TSP 파일을 로드하는 견고한 파서
    다양한 TSPLIB 형식을 지원
    """
    coords = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    node_section = False
    edge_weight_section = False
    dimension = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 차원 정보 추출
        if line.startswith('DIMENSION'):
            try:
                dimension = int(line.split(':')[1].strip())
            except:
                continue
        
        # 노드 좌표 섹션 시작
        if 'NODE_COORD_SECTION' in line:
            node_section = True
            continue
        
        # 섹션 종료 조건들
        if line in ['EOF', 'EDGE_WEIGHT_SECTION', 'DISPLAY_DATA_SECTION']:
            break
        
        # 노드 좌표 파싱
        if node_section and line:
            # 공백으로 분리
            parts = re.split(r'\s+', line.strip())
            
            if len(parts) >= 3:
                try:
                    # 첫 번째는 노드 ID, 나머지는 좌표
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
                except (ValueError, IndexError):
                    continue
            elif len(parts) == 2:
                # ID 없이 바로 좌표인 경우
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append((x, y))
                except ValueError:
                    continue
    
    # 결과 검증
    if not coords:
        print(f"⚠️  Warning: No coordinates found in {filepath}")
        # 파일 내용 일부 출력 (디버깅용)
        print("File preview:")
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1}: {line.strip()}")
        if len(lines) > 10:
            print("  ...")
    
    return coords

def load_tsp_alternative(filepath):
    """
    대안 TSP 파서 - 더 관대한 파싱
    """
    coords = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 숫자로만 이루어진 라인들을 찾아서 좌표로 해석
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('NAME', 'TYPE', 'COMMENT', 'DIMENSION', 'EDGE_WEIGHT')):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # 마지막 두 개를 x, y로 시도
                    x = float(parts[-2])
                    y = float(parts[-1])
                    coords.append((x, y))
                except ValueError:
                    continue
    
    except Exception as e:
        print(f"Error in alternative parser: {e}")
    
    return coords

def compute_dist_matrix(coords):
    """거리 행렬 계산"""
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])
    return D
