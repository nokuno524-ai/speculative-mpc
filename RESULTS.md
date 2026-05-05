# Round 4: Hybrid Coarse-to-Fine CEM — CartPole-v1

## Approach
Combine jump-ahead (fast but inaccurate) with target model (slow but accurate):
1. Evaluate all N=512 candidates with jump-ahead model (coarse)
2. Select top-K candidates
3. Re-evaluate top-K with target model (fine)
4. CEM update from fine-evaluated elites

## Jump-Ahead Model Quality (Persistent Problem)
- **MSE: 0.687** (barely improved from r3's 0.60)
- **Cosine similarity: 0.056** (near zero — state prediction essentially random)
- **Spearman ranking correlation: 0.39** (weak but non-trivial)
- Top-50 overlap with target's top-100: 92% (reasonable at large K)
- Architecture: Transformer encoder + residual connections, 178% of target params
- Tried: normalized targets, cosine loss, AdamW, cosine LR schedule, gradient clipping
- **State prediction for CartPole RSSM latent space appears fundamentally hard**

## Results

| Method | Reward | Time (ms) | Retention | Speedup | Rollouts Saved |
|--------|--------|-----------|-----------|---------|----------------|
| Target-only | 16.9 ± 6.9 | 690 | 100.0% | 1.00x | 0% |
| Jump-only | 13.9 ± 3.9 | 206 | 82.5% | 3.35x | 100% |
| Hybrid(K=5) | 17.1 ± 4.6 | 933 | 101.2% | 0.74x | 99% |
| Hybrid(K=10) | 16.4 ± 7.7 | 901 | 97.6% | 0.77x | 98% |
| Hybrid(K=20) | 15.6 ± 5.4 | 900 | 92.6% | 0.77x | 96% |
| Hybrid(K=50) | 19.1 ± 7.9 | 1114 | 113.6% | 0.62x | 90% |

## Analysis

### ✅ Reward retention solved
Hybrid CEM achieves 93-114% retention across all K values. The coarse-to-fine approach works as intended — target model re-evaluation ensures action quality.

### ❌ No end-to-end speedup
All hybrid variants are **slower** than target-only (0.62-0.77x). Root cause:
1. **Coarse step adds ~250ms** overhead for evaluating 512 candidates with jump-ahead
2. **Fine step still costs ~660-830ms** for K target rollouts
3. **Target rollouts on GPU are batch-efficient** — even K=5 takes 679ms (batched GPU compute)
4. The GPU is not the bottleneck — it's the CEM iteration count and overhead

### Why the jump-ahead model can't learn state prediction
- MSE stuck at ~0.68 regardless of architecture (MLP, transformer, residual, normalization)
- Cosine sim ~0.06 means predicted states point in random directions
- **Hypothesis**: The RSSM stochastic state space is nearly isotropic (std ≈ 0.75-0.83 across all dims) — states are spread uniformly and k=5 step transitions are hard to predict from state+actions alone (would need the deterministic GRU state too)

### The real bottleneck
On CartPole with a small model, GPU-batched target rollouts are already fast (~690ms for 512×5 rollouts). The overhead of maintaining a separate model and doing two evaluation passes outweighs any savings from fewer target rollouts.

## Verdict
**⚠️ Another negative result.** The hybrid coarse-to-fine approach correctly preserves reward quality but doesn't achieve speedup because:
1. Jump-ahead state prediction quality is too poor for meaningful coarse filtering
2. GPU-batched target rollouts leave little room for improvement on small problems

**Possible paths forward:**
- Test on larger problems (HalfCheetah, humanoid) where target rollouts are genuinely expensive
- Use deterministic state from target GRU alongside jump-ahead stochastic state
- Skip state prediction entirely — just train a direct Q-function / return predictor
