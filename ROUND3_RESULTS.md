# Round 3 Results: Negative Result + Jump-Ahead Pivot

## Summary

Documented the negative result (speculative decoding doesn't transfer) and explored a pivot: **Jump-Ahead Prediction**. The pivot achieves real speedup (6-12x) but at the cost of planning quality (44.5% reward retention).

## Negative Result: Why Speculative Decoding Doesn't Transfer

See `NEGATIVE_RESULT.md` for the full write-up. Key insight:

- **LLM verification**: O(1) — single parallel forward pass
- **RSSM verification**: O(H) — sequential GRU, same as original rollout
- **Result**: 0.9x end-to-end (slower!) despite 17x draft speedup

## Pivot: Jump-Ahead Prediction

Instead of predicting s_{t+1} from s_t (H sequential steps), predict **s_{t+k}** directly from s_t + [a_t, ..., a_{t+k-1}] using a feedforward MLP. This reduces sequential steps from H to H/k.

### Architecture
- MLP takes current stochastic state + flattened action sequence → predicts state at t+k
- No recurrence → each jump is a single forward pass
- Also predicts k rewards per jump step

### Speedup Results (CartPole-v1, H=30, A6000)

| Jump k | Sequential Steps | Time (ms) | Speedup |
|--------|-----------------|-----------|---------|
| 1 (target) | 30 | 19.57 | 1.0x |
| 3 | 10 | 5.20 | 3.8x |
| 5 | 6 | 3.16 | **6.2x** |
| 10 | 3 | 1.61 | **12.1x** |

### Planning Quality (CartPole-v1, CEM H=12)

| Metric | Target-only | Jump-ahead (k=5) |
|--------|------------|-------------------|
| Mean reward | 21.1 ± 8.2 | 9.4 ± 0.7 |
| Reward retention | — | **44.5%** |

### Analysis

**Speedup is real** (6.2x with k=5, 12.1x with k=10), unlike speculative decoding's 0.9x.

**But planning quality is poor** (44.5% retention). The jump-ahead state MSE is stuck at ~0.6 — the model isn't learning accurate multi-step predictions. This is likely because:
1. The stochastic state space is high-dimensional (30-dim) and the jump is large
2. The model sees the full action sequence but loses the intermediate dynamics
3. CEM proposals based on inaccurate jump predictions are poor

### Comparison

| Approach | Speedup | Reward Retention | Verdict |
|----------|---------|-----------------|---------|
| Speculative Decoding | 0.9x ❌ | 55% | Verification kills speedup |
| Jump-Ahead (k=5) | 6.2x ✅ | 44% ❌ | Fast but inaccurate |
| Jump-Ahead (k=10) | 12.1x ✅ | — | Even faster, likely worse |

### Conclusion

Jump-ahead prediction genuinely speeds up the rollout (unlike speculative decoding), but the state prediction accuracy is too low for useful planning. The speedup–accuracy tradeoff is unfavorable:

- Small k (3-5): modest speedup, still poor accuracy
- Large k (10): great speedup, likely terrible accuracy

The fundamental issue is that **skipping intermediate states loses too much information** about the dynamics. Each intermediate state encodes the effect of one action, and compressing k actions into a single prediction discards the sequential structure that matters for planning.

### Future Directions

1. **Hybrid approach**: Use jump-ahead for coarse CEM proposals (fast), then refine top-k candidates with full target rollouts
2. **Residual jump-ahead**: Predict only the *delta* from a linear dynamics model
3. **Multi-scale**: Train jump models at k=1,2,4,8 and adaptively choose granularity
4. **Accept that world model rollouts are inherently sequential** and focus on reducing the number of CEM iterations needed

## Files
- `NEGATIVE_RESULT.md` — Full negative result write-up
- `src/jump_ahead.py` — Jump-ahead implementation + evaluation
- `scripts/run_jump.sh` — Slurm script
