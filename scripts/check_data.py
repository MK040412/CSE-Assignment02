"""
TSP 데이터 파일 확인 스크립트
"""

import os
import sys
sys.path.append('src')

from utils import load_tsp, load_tsp_alternative

def check_tsp_file(filepath):
    """TSP 파일 상세 확인"""
    print(f"\n🔍 Checking: {filepath}")
    print("-" * 50)
    
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return False
    
    # 파일 크기 확인
    file_size = os.path.getsize(filepath)
    print(f"📁 File size: {file_size:,} bytes")
    
    # 파일 내용 미리보기
    print("📄 File preview (first 20 lines):")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[:20]):
            print(f"  {i+1:2d}: {line.strip()}")
        
        if len(lines) > 20:
            print(f"  ... (total {len(lines)} lines)")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # 기본 파서로 시도
    print("\n🔧 Testing default parser...")
    coords = load_tsp(filepath)
    print(f"   Loaded {len(coords)} coordinates")
    
    if len(coords) == 0:
        print("⚠️  Default parser failed, trying alternative...")
        coords = load_tsp_alternative(filepath)
        print(f"   Alternative parser loaded {len(coords)} coordinates")
    
    if len(coords) > 0:
        print("✅ Successfully parsed!")
        print(f"   First few coordinates: {coords[:5]}")
        if len(coords) > 5:
            print(f"   Last few coordinates: {coords[-3:]}")
        
        # 좌표 범위 확인
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            print(f"   X range: {min(xs):.2f} to {max(xs):.2f}")
            print(f"   Y range: {min(ys):.2f} to {max(ys):.2f}")
        
        return True
    else:
        print("❌ Failed to parse coordinates!")
        return False

def main():
    """모든 데이터 파일 확인"""
    print("=" * 60)
    print("TSP Data Files Verification")
    print("=" * 60)
    
    datasets = ['a280', 'xql662', 'kz9976', 'monalisa']
    results = {}
    
    for dataset in datasets:
        filepath = f"data/{dataset}.tsp"
        success = check_tsp_file(filepath)
        results[dataset] = success
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    for dataset, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"{dataset:>10}: {status}")
    
    failed_count = sum(1 for success in results.values() if not success)
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} files need attention!")
        print("\n🔧 Suggestions:")
        print("1. Re-download the problematic files")
        print("2. Check if the TSP format is different")
        print("3. Try manual parsing for specific formats")
    else:
        print("\n🎉 All files are ready for experiments!")

if __name__ == '__main__':
    main()
