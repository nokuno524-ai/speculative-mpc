# Speculative Decoding for World Model Rollouts

Adapting speculative decoding to accelerate MPC planning with RSSM world models.

## Key Idea

Instead of sequentially rolling out a large GRU-based RSSM for H timesteps (O(H)), we use a **non-autoregressive draft MLP** that predicts all H future states in a single forward pass (O(1)). The target model then verifies the draft's predictions in parallel, accepting a prefix where distributions match.

## Architecture

| Component | Type | Rollout | Params |
|-----------|------|---------|--------|
| **Target** | Deep GRU-based RSSM | Sequential O(H) | ~2M |
| **Draft** | Non-autoregressive MLP | Single pass O(1) | ~200K |

### Adaptive Acceptance
- KL threshold relaxes over horizon: `ε_t = ε₀ + α·t`
- Accept contiguous prefix via `cumprod` mask
- On rejection at step k: resample from target, re-draft remaining steps

## Files

```
src/
├── rssm.py           # Target RSSM (GRU + stochastic states)
├── ml_draft.py       # Non-autoregressive draft MLP
├── acceptance.py     # Adaptive KL acceptance criteria
├── parallel_verify.py # Target verification of draft predictions
├── cem_planner.py    # CEM planner (target-only + speculative modes)
├── utils.py          # Env wrappers
└── main.py           # Full pipeline: train → distill → benchmark
scripts/
├── run_all.sh        # Run pipeline
└── sbatch_pipeline.sh # Slurm submission script
```

## Quick Start

```bash
# Local (CPU)
pip install torch gymnasium numpy
python -m src.main

# Rivanna (GPU)
sbatch scripts/sbatch_pipeline.sh
```

## Results

See `results/results.json` after training. Target metrics:
- Draft speedup: >10x (single pass vs sequential)
- Speculative speedup (draft + verify): >2x
- Reward retention: ≥98% of target-only CEM
- Acceptance rate: monitored via adaptive KL threshold

## Training Pipeline

1. **Collect data**: 500 random CartPole-v1 episodes
2. **Train target**: 300 epochs, GRU-based RSSM (obs→posterior, transition prior, reward)
3. **Distill draft**: 300 epochs, MSE + KL matching to target's stochastic states
4. **Benchmark**: Speed comparison (target vs draft vs speculative)
5. **MPC evaluation**: CEM planning on actual CartPole episodes
