#!/bin/bash
#SBATCH --job-name=specmpc-cheetah
#SBATCH --partition=gpu-a6000
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/qzp4ta/speculative-mpc/logs/cheetah_%j.out
#SBATCH --error=/scratch/qzp4ta/speculative-mpc/logs/cheetah_%j.err

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/speculative-mpc

source .venv/bin/activate

echo "=== HalfCheetah Round 2 ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

python -m src.main_v2 --env HalfCheetah-v4 --n-episodes 1000 --epsilon 0.1 --target-epochs 300 --draft-epochs 300

echo "=== HalfCheetah done ==="
