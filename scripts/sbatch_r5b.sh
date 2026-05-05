#!/bin/bash
#SBATCH --job-name=specmpc-r5b
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/qzp4ta/speculative-mpc/logs/r5b_%j.out
#SBATCH --error=/scratch/qzp4ta/speculative-mpc/logs/r5b_%j.err

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/speculative-mpc

source .venv/bin/activate

echo "=== Running HalfCheetah-v4 ==="
python src/r5b_improved_draft.py --env HalfCheetah-v4 --n-data-eps 15000 --target-epochs 600 --draft-epochs 800

echo "=== Running CartPole-v1 for comparison ==="
python src/r5b_improved_draft.py --env CartPole-v1 --n-data-eps 10000 --target-epochs 400 --draft-epochs 600
