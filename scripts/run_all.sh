#!/bin/bash
# Full pipeline: train target RSSM → distill MLP draft → benchmark
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Speculative MPC Pipeline ==="
echo "Starting at $(date)"

python -m src.main

echo "Pipeline complete at $(date)"
