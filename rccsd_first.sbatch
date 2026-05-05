#!/bin/bash
#SBATCH --job-name=rccsd-first
#SBATCH --partition=gpu
#SBATCH --account=zhangmlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rccsd_first_%j.out
#SBATCH --error=logs/rccsd_first_%j.err

set -e
echo "=== RCCSD First Experiment ==="
echo "Start: $(date)"

cd /scratch/qzp4ta/speculative-mpc

# Use existing venv if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "/p/realai/bohan/speculative-mpc/.venv" ]; then
    source /p/realai/bohan/speculative-mpc/.venv/bin/activate
else
    echo "No venv found, creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch numpy matplotlib
fi

export PYTHONUNBUFFERED=1
python experiments/rccsd_experiment.py

echo "End: $(date)"
