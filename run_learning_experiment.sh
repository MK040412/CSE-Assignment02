#!/bin/bash

echo "🧠 Learning UTSP Experiment Pipeline"
echo "===================================="

# 1. 작은 테스트 데이터 생성
echo "📊 Creating small test datasets..."
python scripts/create_small_datasets.py

# 2. 학습 UTSP 실험 실행
echo "🚀 Running learning experiments..."
python src/experiment_with_learning.py --algorithms mst,utsp,utsp_learning --datasets tiny8,small12,medium15 --compare-utsp

# 3. 학습 과정 시각화
echo "🎬 Creating learning visualizations..."
python scripts/visualize_learning.py

# 4. 결과 확인
echo "📋 Generated files:"
find figures/ results/ -name "*learning*" -o -name "*animation*" | sort

echo ""
echo "✅ Learning UTSP experiment completed!"
echo "🧠 Learning reports: results/learning_utsp_report.md"
echo "📊 Performance plots: figures/learning_comparison.png"
echo "🎬 Learning animations: figures/animations/"
