#!/bin/bash
#SBATCH --job-name=specmpc-h4567
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/mechanism_h4567_%j.out
#SBATCH --error=logs/mechanism_h4567_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJ="/scratch/qzp4ta/speculative-mpc"
cd "$PROJ"
source .venv/bin/activate

echo "=== H4 ===" && python experiments/h4_stochastic_vs_deterministic.py 2>&1 | tee experiment_results/h4_log.txt
echo "=== H5 ===" && python experiments/h5_acceptance_metrics.py 2>&1 | tee experiment_results/h5_log.txt
echo "=== H6 ===" && python experiments/h6_pareto_frontier.py 2>&1 | tee experiment_results/h6_log.txt
echo "=== H7 ===" && python experiments/h7_outlier_aware.py 2>&1 | tee experiment_results/h7_log.txt
echo "=== Done ==="
