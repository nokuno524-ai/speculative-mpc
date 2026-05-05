# Speculative Decoding for World Model Rollouts

Adapting speculative decoding from LLMs to continuous latent spaces for Model Predictive Control (MPC) with Recurrent State-Space Models (RSSMs).

## Key Innovation

**Adaptive KL Threshold**: In continuous latent spaces, distributional errors compound over the planning horizon. We use an adaptive threshold:

```
ε_t = ε₀ + α·t
```

This prevents exponential acceptance rate drop-off that occurs with fixed thresholds.

## Architecture

| Component | Description |
|-----------|-------------|
| **Target RSSM** | Full-size RSSM (GRU + Gaussian stochastic states, ~50K params) |
| **Draft RSSM** | 4x smaller distilled RSSM (~12K params) |
| **Parallel Verifier** | Batched target verification using draft's projected states |
| **Adaptive Acceptance** | KL-based acceptance with cumprod prefix mask |
| **CEM Planner** | Cross-Entropy Method with speculative rollout benchmarking |

## Pipeline

```
1. Draft Model generates trajectory (fast, autoregressive, deterministic mean)
2. Target Model verifies ALL steps (sequential GRU but using draft's projected stochs)
3. Adaptive KL acceptance determines valid prefix length
4. Rejected steps use Target's mean to correct trajectory
5. CEM planner uses corrected rollouts for action optimization
```

## Quick Start

```bash
# On Rivanna
cd /scratch/qzp4ta/speculative-mpc
source .venv/bin/activate

# Run directly
python -m src.train_cartpole

# Or submit to Slurm
sbatch scripts/train_cartpole.sh
```

## Replication Steps

1. **Train Target RSSM** on CartPole-v1 (100 epochs, sequence training)
2. **Distill Draft RSSM** via state alignment MSE (80 epochs)
3. **Benchmark**: Compare target-only vs speculative rollout wall-clock time

## Expected Results

| Metric | Target |
|--------|--------|
| Speedup | ≥2x |
| Acceptance rate | Varies with ε₀, α |
| Draft size | ~25% of target params |

## File Structure

```
src/
├── rssm.py           # Target RSSM with GRU
├── draft_model.py    # 4x smaller draft RSSM with projection layers
├── parallel_verify.py # Target verification using draft states
├── acceptance.py     # Adaptive KL acceptance + speculative rollout
├── cem_planner.py    # CEM planner with benchmarking
└── train_cartpole.py # End-to-end CartPole experiment
scripts/
└── train_cartpole.sh # Slurm sbatch script
```

## References

- Chen et al. (2023) - "Accelerating Large Language Model Decoding with Speculative Sampling"
- Hafner et al. (2023) - "Mastering Diverse Domains through World Models" (DreamerV3)
