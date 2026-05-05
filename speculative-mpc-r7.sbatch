#!/bin/bash
#SBATCH --job-name=specmpc-r7
#SBATCH --output=/scratch/qzp4ta/speculative-mpc/logs/r7_%j.out
#SBATCH --error=/scratch/qzp4ta/speculative-mpc/logs/r7_%j.err
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G

export PYTHONUNBUFFERED=1
source /scratch/qzp4ta/speculative-mpc/.venv/bin/activate

echo "=== R7: E2E CEM Benchmark ==="
echo "Start: $(date)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python /scratch/qzp4ta/speculative-mpc/speculative-mpc-r7.py --envs CartPole-v1 HalfCheetah-v4 --n-episodes 10 --horizon 12

echo "Done: $(date)"
