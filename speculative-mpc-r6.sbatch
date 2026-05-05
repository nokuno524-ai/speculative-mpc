#!/bin/bash
#SBATCH --job-name=specmpc-r6
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/qzp4ta/speculative-mpc/logs/r6_%j.out
#SBATCH --error=/scratch/qzp4ta/speculative-mpc/logs/r6_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /scratch/qzp4ta/speculative-mpc
mkdir -p logs results experiment_results checkpoints

echo "=== R6: Complete Sweep ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu')"
echo "Start: $(date)"

module load python/3.11.0 2>/dev/null || true
source .venv/bin/activate

echo "=== HalfCheetah ==="
python3 speculative-mpc-r6.py --env HalfCheetah-v4 --train-steps 20000

echo ""
echo "=== CartPole (remaining configs) ==="
python3 speculative-mpc-r6.py --env CartPole-v1 --train-steps 20000

echo ""
echo "=== Done: $(date) ==="
