#!/bin/bash
#SBATCH --job-name=specmpc-jump
#SBATCH --partition=gpu-a6000
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/qzp4ta/speculative-mpc/logs/jump_%j.out
#SBATCH --error=/scratch/qzp4ta/speculative-mpc/logs/jump_%j.err

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/speculative-mpc

source .venv/bin/activate

echo "=== Jump-Ahead Pivot ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu')"
echo "Time: $(date)"

python src/jump_ahead.py --env CartPole-v1 --n-episodes 10000 --target-epochs 500 --jump-epochs 500 --jump-k 5

echo "=== Done at $(date) ==="
