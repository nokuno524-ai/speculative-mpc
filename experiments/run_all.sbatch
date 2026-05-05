#!/bin/bash
#SBATCH --job-name=specmpc-mechanism
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/mechanism_%j.out
#SBATCH --error=logs/mechanism_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Speculative MPC Mechanism Investigation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

PROJ="/scratch/qzp4ta/speculative-mpc"
cd "$PROJ"

# Activate venv
source .venv/bin/activate

# Create output dirs
mkdir -p experiment_results logs

echo "=== H1: Verification Cost ==="
python experiments/h1_verification_cost.py 2>&1 | tee experiment_results/h1_log.txt
echo ""

echo "=== H2: Error Compounding ==="
python experiments/h2_error_compounding.py 2>&1 | tee experiment_results/h2_log.txt
echo ""

echo "=== H3: Transformer Transition ==="
python experiments/h3_transformer_transition.py 2>&1 | tee experiment_results/h3_log.txt
echo ""

echo "=== H4: Stochastic vs Deterministic ==="
python experiments/h4_stochastic_vs_deterministic.py 2>&1 | tee experiment_results/h4_log.txt
echo ""

echo "=== H5: Acceptance Metrics ==="
python experiments/h5_acceptance_metrics.py 2>&1 | tee experiment_results/h5_log.txt
echo ""

echo "=== H6: Pareto Frontier ==="
python experiments/h6_pareto_frontier.py 2>&1 | tee experiment_results/h6_log.txt
echo ""

echo "=== H7: Outlier-Aware Draft ==="
python experiments/h7_outlier_aware.py 2>&1 | tee experiment_results/h7_log.txt
echo ""

echo "=== All experiments complete: $(date) ==="
echo "Results in: $PROJ/experiment_results/"
