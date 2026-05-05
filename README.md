# Speculative Decoding for World Model Rollouts

Adapting speculative decoding to accelerate MPC planning with RSSM world models.

## Key Result

**22.6x speedup** in rollout evaluation using a non-autoregressive draft MLP vs sequential GRU-based target, with 100% acceptance rate and 87% reward retention.

## Architecture

| Component | Type | Params | Rollout |
|-----------|------|--------|---------|
| **Target** | Deep GRU-based RSSM | 1.1M (100%) | Sequential O(H) |
| **Draft** | Non-autoregressive MLP | 45K (4%) | Single pass O(1) |

### How It Works

1. **Draft model** takes `(stoch_0, actions[0:H])` → predicts all H stochastic states in **one forward pass**
2. **Target reward head** evaluates draft's predicted states (no GRU needed during proposal)
3. **Adaptive KL acceptance**: `ε_t = ε₀ + α·t` relaxes threshold over horizon
4. **Cumprod prefix mask** finds contiguous accepted prefix

### Why Non-Autoregressive?

Previous attempts with GRU-based draft models showed that Python loop overhead kills the speedup. The MLP draft avoids any sequential computation — it's a single matrix multiply that predicts all H states at once.

## Results (A6000 GPU, CartPole-v1, H=30)

| Metric | Value |
|--------|-------|
| Target rollout (sequential) | 14.9 ms |
| Speculative (draft+reward) | 0.66 ms |
| **Speedup** | **22.6x** |
| Acceptance rate | 100% |
| Mean KL divergence | 0.0014 |
| MPC reward retention | 87% |

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
# Local (CPU/GPU)
pip install torch gymnasium numpy
python -m src.main

# Rivanna (GPU)
sbatch scripts/sbatch_pipeline.sh
```

## Training Pipeline

1. **Collect data**: 500 random CartPole-v1 episodes (~11K transitions)
2. **Train target**: 300 epochs, GRU-based RSSM (posterior, prior, reward)
3. **Distill draft**: 300 epochs, MSE + KL matching to target's stochastic states
4. **Benchmark**: Speed comparison + acceptance rate measurement
5. **MPC evaluation**: CEM planning on actual CartPole episodes
