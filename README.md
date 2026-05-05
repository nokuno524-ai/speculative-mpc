# Speculative Decoding for World Model Rollouts

Adapting speculative decoding from LLMs to continuous latent spaces for Model Predictive Control (MPC) with Recurrent State-Space Models (RSSMs).

## Key Innovation

**Adaptive KL Threshold**: In discrete token spaces, a token is either right or wrong. In continuous latent spaces, small distributional errors compound over the planning horizon. We use an adaptive threshold that relaxes with time:

```
ε_t = ε₀ + α·t
```

This prevents exponential acceptance rate drop-off that would occur with a fixed threshold.

## Architecture

- **Target RSSM**: Full-size recurrent state-space model with GRU transition
- **Draft RSSM**: 4x smaller distilled model for fast speculative rollouts
- **Parallel Verifier**: Batched target verification of drafted trajectories
- **Adaptive Acceptance**: KL-based acceptance with horizon-relaxing threshold
- **CEM Planner**: Cross-Entropy Method integrated with speculative decoding

## Pipeline

```
1. Draft Model generates trajectory s^{draft}_{1:H} (fast, autoregressive)
2. Target Model verifies ALL steps in parallel (batched)
3. Adaptive KL acceptance determines valid prefix length
4. Rejected steps use Target's distribution to correct trajectory
5. CEM planner uses corrected rollouts for action optimization
```

## Installation

```bash
cd /scratch/qzp4ta/speculative-mpc
source .venv/bin/activate
pip install gymnasium[classic-control]
```

## Training

```bash
# Submit to Slurm
sbatch scripts/train_cartpole.sh

# Or run directly
python -m src.train_cartpole
```

## Replication

1. **Train Target RSSM** on CartPole-v1 (50 epochs, ~2 min on GPU)
2. **Distill Draft RSSM** via KL matching (30 epochs, ~1 min)
3. **Benchmark**: Compare target-only vs speculative rollouts
   - Measure: wall-clock time, acceptance rate, reward retention, KL divergence

## Expected Results

| Metric | Target |
|--------|--------|
| Speedup | ≥2x |
| Reward retention | ≥98% |
| Acceptance rate | >60% (varies with ε₀, α) |
| Draft size | 25% of target params |

## File Structure

```
src/
├── rssm.py           # Target RSSM with GRU
├── draft_model.py    # 4x smaller draft RSSM  
├── parallel_verify.py # Batched target verification
├── acceptance.py     # Adaptive KL acceptance + full speculative rollout
├── cem_planner.py    # CEM with speculative benchmarking
├── train.py          # Training loop for target + distillation
└── train_cartpole.py # CartPole experiment entry point
scripts/
└── train_cartpole.sh # Slurm sbatch script
```

## References

- Chen et al. (2023) - "Accelerating Large Language Model Decoding with Speculative Sampling"
- Hafner et al. (2023) - "Mastering Diverse Domains through World Models" (DreamerV3)
- Zhang et al. (2023) - "Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding"
