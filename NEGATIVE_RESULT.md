# Negative Result: LLM Speculative Decoding Does Not Transfer to World Model Planning

## TL;DR

We attempted to apply LLM-style speculative decoding to accelerate model-predictive control (MPC) with learned world models (RSSM). Despite achieving 17x speedup in the draft model's forward pass, the end-to-end speedup was **0.9x** (i.e., slower than baseline) across two environments (CartPole-v1, HalfCheetah-v4). The fundamental reason: verification in continuous latent spaces is as expensive as the original computation.

## Background

**Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) accelerates LLM inference by:
1. A small **draft model** generates K tokens autoregressively (fast but less accurate)
2. The large **target model** verifies all K tokens in a **single forward pass** (parallel attention)
3. Accepted tokens are kept; rejected tokens are re-sampled from the target distribution

The key insight is that **verification is O(1)** — the target processes all K draft tokens in parallel because transformer attention has no sequential dependency during inference.

## Our Setup

We applied this framework to RSSM-based world model planning:

- **Target model**: DeepRSSM with 6-layer MLPs, GRU recurrence (det_dim=200, stoch_dim=30)
- **Draft model**: CausalConvDraft — 1D causal convolution with positional encoding, ~5% of target params
- **Task**: CEM planning with horizon H=12-30, comparing target-only rollouts vs. speculative rollouts

## Results

| Metric | CartPole-v1 | HalfCheetah-v4 | LLM (reference) |
|--------|-------------|----------------|-----------------|
| Draft speedup | 17x | 17.1x | 3-10x |
| **End-to-end speedup** | **0.9x** | **0.9x** | **2-3x** |
| Acceptance rate | 72.5% | 74.9% | 70-90% |
| Reward retention | 55% | N/A | N/A (token match) |
| Draft MSE | 6.5e-5 | 2.8e-3 | N/A |

## Why It Fails: The Fundamental Asymmetry

### LLM: Discrete tokens, cheap verification
- Verification = single forward pass of target model over K tokens
- The target's transformer attention processes all positions simultaneously
- Cost: O(1) target forward passes regardless of K

### RSSM: Continuous latents, expensive verification
- Verification requires running the target GRU **sequentially** through H steps
- Each GRU step depends on the previous hidden state: h_{t+1} = GRU(h_t, x_t)
- Even if we know the draft's proposed latent states, we must still compute h_1, h_2, ..., h_H sequentially
- **Cost: O(H) sequential steps — identical to the original rollout**

The draft model's 17x speedup comes from predicting all H latent states in parallel (no GRU). But verification requires the target's GRU to run sequentially, taking just as long as the original. The draft speedup is completely negated.

### Reward Degradation

Even if verification were free, acceptance rate of ~73% means ~27% of states are rejected. These rejections cascade: a rejected state at step t invalidates all subsequent predictions. With only 55% reward retention, the speculative planner makes noticeably worse decisions.

In LLMs, rejected tokens are re-sampled from the target distribution — a clean correction. In RSSMs, rejected latents require re-running the GRU from the rejection point, and the resulting state trajectory diverges from the draft's predictions.

## Quantitative Evidence

The 0.9x end-to-end speedup (actually a slowdown) decomposes as:
- Draft prediction: ~1/17 of target time
- Verification (sequential GRU): ~1x of target time
- Overhead (KL computation, acceptance logic): ~0.1x
- Total: ~1.06x → net slowdown

This was confirmed across:
1. **CartPole-v1**: Simple dynamics, discrete actions, easy to model → still 0.9x
2. **HalfCheetah-v4**: Complex dynamics, continuous 6-dim actions → also 0.9x

The result is environment-agnostic because the bottleneck is architectural, not task-specific.

## Implications

1. **Speculative decoding is specific to parallelizable architectures** (transformers with causal attention). It does not generalize to inherently sequential models (RNNs, GRUs, LSTMs).

2. **The "draft-then-verify" paradigm requires cheap verification.** For discrete token spaces, this is natural (probability comparison). For continuous latent spaces, verification requires reproducing the full sequential computation.

3. **Speeding up world model rollouts requires a fundamentally different approach** — not faster prediction of the same computation, but reducing the computation itself.

## Lessons

1. **Measure end-to-end, not component speedup.** The 17x draft speedup was real but irrelevant because it was in the wrong place.

2. **Acceptance rate alone is misleading.** 72.5% sounds good, but in a sequential context, each rejection cascades.

3. **Analogies between NLP and RL architectures break down at the verification step.** The sequential nature of recurrent models is the irreducible bottleneck.

## Files
- `RESULTS.md` — Full Round 2 results with KL sweep
- `src/main_v2.py` — Pipeline with ε-greedy collection, sweep
- `src/ml_draft_v2.py` — CausalConvDraft architecture
- `src/parallel_verify.py` — Sequential verification (the bottleneck)
