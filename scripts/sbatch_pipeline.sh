#!/bin/bash
#SBATCH --job-name=specmpc-pipeline
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/specmpc-%j.out

set -e
export PYTHONUNBUFFERED=1

echo "=== Speculative MPC on $(hostname) ==="
echo "Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

cd /scratch/qzp4ta/speculative-mpc

# Activate venv
source .venv/bin/activate

# Check deps
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import gymnasium; print(f'Gymnasium {gymnasium.__version__}')"

# Run pipeline
bash scripts/run_all.sh

echo "End: $(date)"
