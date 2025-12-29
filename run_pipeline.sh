#!/usr/bin/env bash
#
# run_pipeline.sh - Simple end-to-end evaluation pipeline example
#
# Usage: ./run_pipeline.sh

set -e

RESULTS_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
TASKS_DIR="eval/tasks"
MODELS="openai:gpt-5.1 openai:gpt-4o openai:o4-mini anthropic:claude-opus-4-20250514"

echo "=== Step 1: Collect Tasks ==="
python collect_tasks.py --output "$TASKS_DIR" --max-prs 2000

echo "=== Step 2: E2E Test ==="
python test_e2e.py --task "$TASKS_DIR/scikit-learn__scikit-learn-30022.json" --max-steps 20

echo "=== Step 3: Benchmark ==="
python benchmark.py --tasks "$TASKS_DIR" --models $MODELS -o "$RESULTS_DIR" --detailed-metrics

echo "=== Step 4: Visualize ==="
python visualize_results.py --results "$RESULTS_DIR" --output "$RESULTS_DIR/report.html"

echo "=== Step 5: Analyze ==="
python analyze_results.py --results "$RESULTS_DIR" --compare-methods --output "$RESULTS_DIR/analysis.json"

echo "=== Done ==="
echo "Results: $RESULTS_DIR"