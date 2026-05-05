#!/bin/bash
#SBATCH --job-name=specmpc-r4
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/r4_%j.out
#SBATCH --error=logs/r4_%j.err

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/speculative-mpc
mkdir -p logs

echo "=== Round 4: Hybrid Coarse-to-Fine CEM ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu')"
echo "Start: $(date)"

module load python/3.11.0 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || { python3 -m venv .venv && source .venv/bin/activate && pip install torch gymnasium numpy; }

python src/run_round4.py \
    --env CartPole-v1 \
    --n-data-eps 15000 \
    --target-epochs 600 \
    --jump-epochs 800 \
    --jump-k 5 \
    --n-mpc-eps 20 \
    --horizon 12 \
    --n-samples 512 \
    --n-iterations 5 \
    --top-k-sweep 5,10,20,50 \
    --seed 42

echo "End: $(date)"
