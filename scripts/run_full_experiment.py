import os
import sys
import subprocess
import argparse

def run_command(cmd):
    """명령어 실행"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='a280', help='Dataset name')
    parser.add_argument('--train_steps', type=int, default=50000, help='SAC training steps')
    parser.add_argument('--utsp_epochs', type=int, default=50, help='UTSP training epochs')
    args = parser.parse_args()
    
    dataset = args.dataset
    
    print("=" * 50)
    print("Full TSP Solver Experiment Pipeline")
    print("=" * 50)
    
    # 1. 데이터 다운로드
    print("\n1. Downloading datasets...")
    run_command("python scripts/download_data.py")
    # 2. UTSP 모델 학습
    print(f"\n2. Training UTSP model on {dataset}...")
    run_command(f"python scripts/train_utsp.py --dataset {dataset} --epochs {args.utsp_epochs}")
    
    # 3. SAC 모델 학습
    print(f"\n3. Training SAC model on {dataset}...")
    run_command(f"python scripts/train_sac.py --dataset {dataset} --timesteps {args.train_steps}")
    
    # 4. 전체 알고리즘 비교 실험
    print(f"\n4. Running comprehensive comparison...")
    algorithms = "mst,utsp,sac,hybrid"
    run_command(f"python src/experiment.py --algorithms {algorithms} --datasets {dataset}")
    
    # 5. 결과 분석
    print(f"\n5. Analyzing results...")
    run_command(f"python scripts/analyze_results.py --dataset {dataset}")
    
    print("\n" + "=" * 50)
    print("Experiment completed! Check the 'figures/' directory for results.")
    print("=" * 50)
    # 3. SAC 모델 학