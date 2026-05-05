# Speculative Decoding for World Model Rollouts

Speculative decoding adapted for continuous latent spaces in Model Predictive Control (MPC) with RSSM world models.

## Overview

This project adapts speculative decoding (originally for LLMs) to accelerate planning in world model-based reinforcement learning. By using a small draft RSSM to propose trajectory rollouts and a large target RSSM to verify them in parallel, we achieve significant wall-clock speedups while maintaining planning quality.

### Key Innovation: Adaptive KL Threshold

In continuous spaces, we use an adaptive acceptance criterion: ε_t = ε₀ + α·t, which relaxes as horizon increases to account for natural uncertainty growth.

## Status

🚧 In Development

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch gymnasium dm_control numpy matplotlib wandb
```

## Usage

```bash
# Train RSSM models and run speculative decoding benchmarks
python train.py --env CartPole-v1
python benchmark.py --horizons 5 10 20 40
```
