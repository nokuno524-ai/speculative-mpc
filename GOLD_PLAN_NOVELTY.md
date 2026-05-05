# GOLD_PLAN_NOVELTY.md — RCCSD (Risk-Calibrated Continuous Speculative Decoding) Novelty & Feasibility Assessment

## 1. Literature Review

### 1.1 SPECS (arxiv 2506.15733) — Reward-Guided Soft Verification
SPECS integrates a Process Reward Model (PRM) into speculative decoding. Its acceptance combines target log-likelihood with PRM reward: `S(y) = α·log P_target(y|x) + (1-α)·r(y)`. A dynamic switching mechanism falls back to the target model when PRM reward drops below threshold τ. **Limitation**: Step-wise PRM thresholds suffer from compounding uncertainty over long horizons — no trajectory-level distributional reasoning.

### 1.2 Spec-VLA (arxiv 2507.22424) — Relaxed Acceptance for Continuous Actions
Spec-VLA defines "top-k synonyms" as tokens whose dequantized continuous actions fall within k-nearest neighbors in action-distance space (Euclidean/Manhattan). Acceptance: `dist(dequantize(â), dequantize(a*)) < ε`. Achieves 44% increase in acceptance length, 1.42x speedup on Libero. **Limitation**: Heuristic distance threshold with no formal divergence guarantees; no trajectory-level reasoning.

### 1.3 SSD (arxiv 2508.17739) — Safety-Aware Speculative Decoding
SSD monitors match-ratio between draft and target. Binary toggle: high match-ratio → "Intersection" mode (prioritize utility), low match-ratio → "Union" mode (prioritize safety). **Limitation**: Binary toggle is brittle — threshold sensitivity means utility can degrade sharply. No continuous risk modeling.

### 1.4 SSS (arxiv 2508.15044) — Reward-Shifted Speculative Sampling
SSS modifies acceptance: `Acc(x) = min(1, π_t(x)·exp(r(x)/β) / π_d(x))`. Achieves test-time weak-to-strong alignment by shifting acceptance toward high-reward regions of the target distribution. **Limitation**: Operates per-token in discrete space; no trajectory distribution reasoning.

### 1.5 CDSL (arxiv 2412.10418) — Constrained Decoding with Speculative Lookaheads
CDSL uses draft model to generate k lookahead tokens, verified against both target distribution and grammar/constraint satisfaction. State-machine acceptance. 2.2–12x speedup on constrained tasks. **Limitation**: Discrete constraint satisfaction; no probabilistic risk modeling.

## 2. Novelty Assessment

### ✅ Genuinely Novel Elements

1. **Wasserstein distance over trajectory distributions**: All existing methods (SPECS, Spec-VLA, SSD, SSS, CDSL) operate at the single-token or single-step level. None formulate acceptance as a distributional comparison between draft and target trajectory distributions using W_2 distance. This is a real gap.

2. **Temporal risk tensor modulation**: No existing work modulates acceptance thresholds by a cumulative temporal risk function that grows exponentially near safety envelope boundaries. SSD's binary match-ratio toggle is the closest, but it's categorically different — binary vs. continuous, reactive vs. predictive.

3. **Dual-critic (utility + safety) verification in speculative decoding**: While dual-critic RL exists (constrained MDP literature), applying it to speculative decoding acceptance is novel. SPECS has a single PRM; SSD has a binary safety toggle; RCCSD proposes a continuous joint acceptance integrating both.

4. **Application to continuous latent trajectory spaces**: All speculative decoding work targets either discrete tokens (LLMs) or discretized continuous actions (VLA). RCCSD targets latent state distributions in RSSM-style world models — directly relevant to our speculative-MPC project.

### ⚠️ Caveats on Novelty

- **Wasserstein distance in RL/Control is well-studied** — distributional RL (Bellemare et al.), Wasserstein robust MDPs, etc. The novelty is specifically in *applying it to speculative decoding acceptance*, not in the mathematical tool itself.
- **Temporal risk functions** echo risk-sensitive control (Whittle, 1990; Basu & Borkar, 2008). The novelty is the synthesis with speculative decoding.
- The claim of "genuinely novel" depends on execution quality. If the formulation reduces to "use W_2 instead of KL with a time-varying threshold," reviewers may see it as incremental.

### 📊 Novelty Score: 7/10
- Strong synthesis novelty (Wasserstein + temporal risk + speculative decoding)
- No direct prior art on trajectory-distribution acceptance in speculative decoding
- Vulnerable to "incremental combination of known techniques" criticism if mathematical depth is insufficient

## 3. Feasibility Assessment

### 3.1 Is the Dual-Critic Approach Sound?
**Partially.** The concept is sound — separate utility and safety critics provide orthogonal signals. However:
- In our speculative-MPC setup, we don't have a separate safety critic. We'd need to train one or derive safety from the RSSM's uncertainty estimates.
- The joint acceptance probability formula: `P_accept = σ(-β·W_2(π_draft, π_target) - Risk(A_{1:k}))` requires estimating W_2 between two distributions during inference. This is non-trivial.

### 3.2 Can Wasserstein Bounds Be Computed Efficiently During Inference?
**This is the critical bottleneck.** Computing exact W_2 between trajectory distributions is O(n^3 log n) in general. Feasible approaches:
- **Closed-form for Gaussians**: If both draft and target trajectory distributions are modeled as Gaussians in latent space, W_2 has a closed form: `W_2^2(N(m_1, Σ_1), N(m_2, Σ_2)) = ||m_1-m_2||^2 + Tr(Σ_1 + Σ_2 - 2(Σ_1^{1/2}Σ_2Σ_1^{1/2})^{1/2})`. This is O(d^3) per step due to the matrix square root, but d is the latent dimension (~30-200), so it's feasible.
- **Sinkhorn approximation**: O(n^2) with entropic regularization, well-suited to GPU.
- **Sliced Wasserstein**: Random 1D projections, O(n·d·L) where L is number of projections. Very fast, reasonable approximation.

**Verdict**: Yes, but only with Gaussian or sliced approximations. Exact W_2 is too expensive.

### 3.3 Connection to Our Empirical Finding (GRU is the Bottleneck)
Our core negative result: **GRU sequential verification makes speculative decoding O(H) regardless of draft speedup**. RCCSD doesn't directly solve this — if acceptance still requires running the target GRU to compute W_2, we're back to O(H).

**Key insight for RCCSD feasibility**: The Wasserstein acceptance must be computable *without* running the target GRU. This means:
- The draft produces a trajectory distribution `P_draft(A_{1:k})` directly (our MLP draft already does this)
- The target's "permissible" distribution must be pre-computed or approximated (e.g., via a learned safety boundary in latent space)
- W_2 is computed between draft's predicted distribution and a pre-computed target envelope — **no sequential GRU needed**

This is the crucial design constraint that determines whether RCCSD is viable or just another negative result.

## 4. Recommended First Experiment

### Experiment: Wasserstein-Bounded Draft Acceptance Without Target Verification

**Goal**: Test whether W_2 distance between draft trajectory and a pre-computed target distribution envelope can predict acceptance quality, replacing sequential GRU verification.

**Setup**:
1. Train RSSM target model on CartPole-v1 (existing)
2. Train MLP draft model to predict trajectory distributions (Gaussian params: mean + covariance) rather than point estimates
3. Pre-compute target distribution statistics: collect 1000 target rollouts, fit Gaussian mixture model (or just mean+var per step) — this is the "permissible envelope"
4. At inference: draft predicts Gaussian trajectory → compute W_2^2 against target envelope (closed-form) → accept if below threshold
5. Compare: (a) speed vs target-only, (b) reward retention, (c) acceptance accuracy vs actual GRU verification

**Why this experiment first**:
- Tests the core feasibility question (can W_2 replace GRU verification?)
- Uses existing CartPole infrastructure
- If W_2 acceptance correlates well with actual GRU acceptance (>80% agreement), RCCSD has legs
- If W_2 acceptance doesn't predict GRU acceptance, the entire RCCSD approach needs rethinking

**Expected outcome**: W_2 acceptance will be a noisy but reasonable proxy (~60-70% agreement with GRU), because our draft model's state predictions are already poor (MSE ~0.6, cosine sim ~0.06). The experiment will reveal whether distributional prediction is fundamentally better than point prediction.

**Fallback if experiment fails**: Abandon trajectory-distribution acceptance. Instead, focus on the temporal risk tensor idea alone — use it to modulate jump-ahead step size k in our existing framework (larger k = faster but riskier; smaller k = safer but slower).

## 5. Summary Verdict

| Aspect | Assessment |
|--------|-----------|
| **Novelty** | Genuine (7/10). No prior work combines Wasserstein trajectory bounds with temporal risk in speculative decoding. |
| **Feasibility** | Uncertain (4/10). Core risk: W_2 computation requires target distribution stats that may need the GRU we're trying to avoid. |
| **Connection to our work** | Direct. Our negative result (GRU bottleneck) is the exact problem RCCSD must solve. |
| **Biggest risk** | Becoming a third negative result — Wasserstein acceptance doesn't correlate with actual quality, or requires GRU computation anyway. |
| **Recommended path** | Run the W_2 draft acceptance experiment first. If it works → full RCCSD paper. If it fails → pivot to temporal-risk-modulated jump-ahead (more practical, still publishable). |

---
*Generated: 2026-05-05 | Based on literature review via Gemini, gold-plan strategy document, and empirical results from /scratch/qzp4ta/speculative-mpc/*
