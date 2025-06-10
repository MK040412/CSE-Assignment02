#!/bin/bash

echo "🎯 CSE331 Assignment #2 - Final Execution Pipeline"
echo "=================================================="

# 환경 확인
echo "📋 Checking environment..."
python --version
pip list | grep -E "(numpy|matplotlib|networkx|requests)"

# 전체 파이프라인 실행
echo "🚀 Starting complete pipeline..."

# 1단계: 데이터 준비
echo "1️⃣ Data preparation phase..."
python scripts/download_data.py
python scripts/check_data.py

# 2단계: 실험 실행
echo "2️⃣ Experiment execution phase..."
python src/experiment_fixed.py --algorithms mst,heldkarp,utsp --datasets a280,berlin52 --check-data
python src/experiment_fixed.py --algorithms mst,utsp --datasets xql662,pcb442,kz9976,monalisa --check-data

# 3단계: 결과 분석
echo "3️⃣ Results analysis phase..."
python scripts/analyze_results.py

# 4단계: 최종 검증
echo "4️⃣ Final verification..."
echo "📁 Generated files:"
find figures/ results/ -name "*.png" -o -name "*.txt" -o -name "*.md" | sort

echo ""
echo "✅ Assignment #2 Complete!"
echo "📊 Performance plots: figures/"
echo "📋 Detailed reports: results/"
echo "🎯 Ready for submission!"
