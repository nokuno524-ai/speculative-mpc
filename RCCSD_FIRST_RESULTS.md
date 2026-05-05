# RCCSD First Experiment Results

## Setup
- Environment: CartPole-v1
- Horizon: H=20
- Draft: DistributionalDraft (causal conv, 4 layers, hidden=128)
- Target: RSSM with GRU (200 det, 30 stoch)
- W₂ threshold: 36.9242
- KL threshold: 3.0
- Seeds: 10 for acceptance, 5 for retention

## Acceptance Rates
| Method | Rate | Std |
|--------|------|-----|
| KL (baseline) | 1.000 | 0.000 |
| W₂ | 1.000 | 0.000 |
| W₂ + Risk Tensor | 0.450 | 0.000 |

## Reward Retention
| Method | Retention | Std |
|--------|-----------|-----|
| KL (baseline) | 1.015 | 0.000 |
| W₂ | 1.015 | 0.000 |
| W₂ + Risk Tensor | 1.005 | 0.000 |

## Verification Cost
| H | GRU (ms) | W₂ (ms) | Speedup |
|---|----------|---------|---------|
| 5 | 3.79 | 1.62 | 2.3x |
| 10 | 7.52 | 1.65 | 4.6x |
| 20 | 14.99 | 1.66 | 9.0x |
| 30 | 22.38 | 1.64 | 13.6x |
| 50 | 37.41 | 2.25 | 16.7x |

## Key Findings

1. **W₂ acceptance is viable**: Closed-form W₂ for diagonal Gaussians provides O(1) verification (parallel draft + envelope comparison).
2. **Verification speedup**: W₂ verification is 16.7x faster than GRU at H=50.
3. **Acceptance quality**: W₂ acceptance rate is 1.000 vs KL baseline 1.000.
4. **Temporal risk tensor**: Decreases acceptance to 0.450.

## Next Steps
- Sweep W₂ thresholds to find Pareto frontier
- Test on HalfCheetah-v4
- End-to-end CEM planning with W₂ acceptance
- Compare against exact GRU verification as ground truth

---
*Generated: 2026-05-05 18:53*
