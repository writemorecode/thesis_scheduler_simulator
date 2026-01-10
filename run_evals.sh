#!/bin/bash

SCHEDULERS="ffd_l2,ffd,ffd_sum,ffd_max,ffd_prod,peak_demand,ffd_new,bfd"
SEED="5000"

mkdir -p evaluation
mkdir -p evaluation/eval_results_1
mkdir -p evaluation/raw_eval_data_1

uv run src/generate_dataset.py --output-dir "evaluation/dataset_1" --seed "${SEED}" --K-min 3 --K-max 6 --J-min 15 --J-max 20 --M-min 5 --M-max 8 --T-min 100 --T-max 200

# 1. Run schedulers and write raw per-instance results
echo "Evaluating schedulers..."
uv run src/eval.py --dataset "evaluation/dataset_1" --schedulers "${SCHEDULERS}" --output-dir "evaluation/raw_eval_data_1" --seed "${SEED}"

# 2. Summarize per-scheduler performance
echo "Running per-scheduler summary..."
uv run src/eval_multi_summary.py --results-dir "evaluation/raw_eval_data_1" --output "evaluation/eval_results_1/eval_summary_results_1.csv " --verbose

# 3. Statistical analysis on a subset of schedulers:
echo "Running scheduler statistical summary..."
uv run src/analysis.py --results-dir "evaluation/raw_eval_data_1" --schedulers "${SCHEDULERS}" --export-summary "evaluation/eval_results_1/eval_log_ratio_summary_1.csv"

# 3. Compute performance profiles (win rates) for a subset of schedulers:
echo "Running performance profiles for schedulers..."
uv run src/performance_profile.py --results-dir "evaluation/raw_eval_data_1/" --schedulers "${SCHEDULERS}" --output "evaluation/eval_results_1/eval_performance_profiles_1.csv" --verbose
