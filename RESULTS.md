# Speculative MPC — Round 2 Results

## Overview
Addressing Gemini review feedback on the speculative decoding world model approach. Key issues identified: 100% acceptance rate (red flag), 87% reward retention, both models trained on random data.

## Changes Made

### 1. Epsilon-Greedy Data Collection
- **Before**: 500 episodes, purely random actions
- **After**: 10,000 episodes, ε=0.1 (greedy heuristic + 10% random)
- **Result**: 200K transitions (vs 11K before), much better training data quality
- Average episode reward: ~40 (vs random baseline ~20)

### 2. Improved Draft Architecture (CausalConvDraft)
- **Before**: Simple MLP, each timestep sees only `stoch_0 + action_t` (no temporal context)
- **After**: Causal 1D convolution over full action sequence + positional encoding
- **Result**: Draft MSE improved from **0.6 (stuck)** → **0.000065** (actually learning!)
- Key insight: Draft must predict *prior* states (not posterior), since it lacks observations

### 3. KL Threshold Sweep
- **Before**: Used fixed threshold (ε₀=5.0, α=0.5), always got 100% acceptance
- **After**: Adaptive sweep based on measured KL divergence
- Sweep range automatically calibrated to actual KL (mean KL ≈ 1.6)

**CartPole KL Sweep Results:**

| ε₀    | α     | Acceptance Rate | Median Acc. Length |
|-------|-------|----------------|-------------------|
| 0.001 | 0.000 | 0.0%           | 0                 |
| 0.050 | 0.000 | 0.0%           | 0                 |
| 0.324 | 0.005 | 3.5%           | 0                 |
| 0.811 | 0.000 | 5.0%           | 0                 |
| 1.298 | 0.010 | 10.2%          | 1                 |
| **1.622** | **0.000** | **30.0%** | **9**         |
| **2.433** | **0.001** | **72.5%** | **19**        |
| 3.244 | 0.000 | 100.0%         | 30                |

**Best operating point**: ε₀=2.43, α=0.001 → 72.5% acceptance (in target 70-90% range) ✅

### 4. HalfCheetah-v4 Support
- Continuous action space (6-dim), more complex dynamics
- obs_dim=17, act_dim=6
- Data collection in progress (1K episodes)

## CartPole Results Summary

| Metric                    | Round 1 (v1)  | Round 2 (v2)   | Target    |
|---------------------------|---------------|----------------|-----------|
| Data collection           | 500 eps random| 10K eps ε=0.1  | Expert    |
| Draft MSE                 | 0.60 (stuck)  | 0.000065       | Low       |
| Acceptance rate           | 100% ⚠️       | 72.5% ✅       | 70-90%   |
| Mean KL divergence        | 0.0013        | 1.62           | Non-trivial |
| Draft-only speedup        | 39x           | 17x            | High      |
| Speculative speedup       | 22x*          | 0.9x           | >1x       |
| Reward retention          | 87%           | 55% ⚠️         | >95%      |
| MPC target reward         | 20.0          | 21.5           | Higher    |
| MPC speculative reward    | 17.4          | 11.8           | Higher    |

*v1 speedup was misleading — measured draft-only, not end-to-end with verification

## Key Findings

### What worked ✅
1. **KL sweep is meaningful** — we can now measure a real acceptance boundary
2. **Draft actually learns** — MSE dropped from 0.6 → 0.000065 by training on prior states
3. **Acceptance rate in target range** — 72.5% (was 100%, now realistic)
4. **Draft model is 17x faster** per forward pass than target

### What didn't work ⚠️
1. **End-to-end speedup is 0.9x** (slower!) — the `parallel_verify` step still runs the target GRU sequentially, negating the draft speedup
2. **Reward retention dropped to 55%** — the speculative planner uses draft states with approximated (zero) det states, leading to poor reward predictions
3. **No real planning improvement** — the CEM planner with speculative rollouts doesn't help because verification cost dominates

### Root Cause Analysis
The speculative decoding framework from LLMs doesn't directly translate to world models because:
- In LLMs, verification is cheap (single forward pass to compare token distributions)
- In RSSMs, verification requires running the target GRU sequentially (O(H)), which is the same cost as the original rollout
- The draft's speedup (parallel prediction) is negated by the verification cost

### Potential Solutions (Future Work)
1. **Accept unverfied draft rollouts for CEM proposal** — use draft for fast candidate generation, then only verify the top-k candidates with target
2. **Asynchronous verification** — overlap draft prediction with target verification
3. **Draft-based reward head** — train a separate lightweight reward predictor on draft states
4. **Diffusion-style verification** — multi-step refinement instead of accept/reject

## Files Changed
- `src/main_v2.py` — Full rewrite with epsilon-greedy collection, KL sweep, multi-env support
- `src/ml_draft_v2.py` — CausalConvDraft architecture with temporal context
- `scripts/run_cartpole.sh` — A6000 partition, 1h timeout
- `scripts/run_cheetah.sh` — HalfCheetah-v4 experiment
