#!/bin/bash
#SBATCH --job-name=specmpc-cartpole
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/qzp4ta/speculative-mpc/logs/%j.out
#SBATCH --error=/scratch/qzp4ta/speculative-mpc/logs/%j.err

export PYTHONUNBUFFERED=1

cd /scratch/qzp4ta/speculative-mpc
mkdir -p logs results

source .venv/bin/activate

# Install deps if needed
pip install gymnasium[classic-control] -q 2>/dev/null

echo "=== Speculative MPC: CartPole Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Start: $(date)"

python -m src.train_cartpole

echo "End: $(date)"
