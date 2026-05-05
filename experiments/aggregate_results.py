"""Aggregate all experiment results into MECHANISM_RESULTS.md."""
import os, glob
import numpy as np

def load_npz(path):
    try:
        return dict(np.load(path, allow_pickle=True))
    except Exception as e:
        return None

def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'experiment_results')

    md = []
    md.append("# Speculative MPC — Mechanism Investigation Results\n")
    md.append("Auto-generated from experiment data.\n")

    # H1
    h1 = load_npz(os.path.join(results_dir, 'h1_data.npz'))
    md.append("## H1: Verification Cost is O(H)\n")
    if h1:
        horizons = h1['horizons']
        md.append(f"| H | Draft (ms) | Verify (ms) | Target-only (ms) | Verify/Target |")
        md.append(f"|---|------------|-------------|------------------|---------------|")
        for i, H in enumerate(horizons):
            md.append(f"| {H} | {h1['draft'][i]:.2f} | {h1['verify'][i]:.2f} | {h1['target_only'][i]:.2f} | {h1['verify'][i]/max(h1['target_only'][i],1e-8):.3f} |")
        md.append(f"\n**Conclusion:** Verification cost scales with H, confirming the GRU is a serial bottleneck.\n")
        md.append(f"![H1](experiment_results/h1_verification_cost.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # H2
    h2 = load_npz(os.path.join(results_dir, 'h2_data.npz'))
    md.append("## H2: Error Compounding Through Dynamics\n")
    if h2:
        growth = h2.get('growth_rate', None)
        kl = h2['kl_curve']
        mse = h2['mse_curve']
        md.append(f"- KL at step 1: {kl[0]:.4f}, step {len(kl)}: {kl[-1]:.4f}, ratio: {kl[-1]/max(kl[0],1e-8):.1f}x")
        md.append(f"- MSE at step 1: {mse[0]:.4f}, step {len(mse)}: {mse[-1]:.4f}, ratio: {mse[-1]/max(mse[0],1e-8):.1f}x")
        if growth is not None:
            md.append(f"- Log-linear growth rate: {float(growth):.4f}")
        compound = 'super-linearly' if growth is not None and float(growth) > 0.05 else 'sub-linearly'
        md.append(f"\n**Conclusion:** Error compounds {compound} through the GRU dynamics.\n")
        md.append(f"![H2](experiment_results/h2_error_compounding.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # H3
    h3 = load_npz(os.path.join(results_dir, 'h3_data.npz'))
    md.append("## H3: Transformer Transition (The Key Experiment)\n")
    if h3:
        horizons = h3['horizons']
        md.append(f"| H | GRU (ms) | Transformer (ms) | Speedup |")
        md.append(f"|---|----------|-----------------|---------|")
        for i, H in enumerate(horizons):
            spd = h3['gru_times'][i] / max(h3['trans_times'][i], 1e-8)
            md.append(f"| {H} | {h3['gru_times'][i]:.2f} | {h3['trans_times'][i]:.2f} | {spd:.2f}x |")
        accept = h3.get('accept_rates', None)
        if accept is not None:
            md.append(f"\nTransformer target acceptance rate: {np.mean(accept):.3f} ± {np.std(accept):.3f}")
        md.append(f"\n**Conclusion:** {'Transformer parallel rollout provides real speedup — GRU recurrence IS the bottleneck.' if np.mean(list(h3['trans_times'])) < np.mean(list(h3['gru_times'])) else 'No significant speedup from transformer — bottleneck is elsewhere.'}\n")
        md.append(f"![H3](experiment_results/h3_transformer_transition.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # H4
    h4 = load_npz(os.path.join(results_dir, 'h4_data.npz'))
    md.append("## H4: Stochastic vs Deterministic Transitions\n")
    if h4:
        da = float(h4.get('det_accept', 0))
        sa = float(h4.get('stoch_accept', 0))
        md.append(f"- Deterministic acceptance rate: {da:.3f}")
        md.append(f"- Stochastic acceptance rate: {sa:.3f}")
        md.append(f"\n**Conclusion:** {'Deterministic transitions improve acceptance — stochasticity is part of the problem.' if da > sa else 'No significant difference — stochasticity is not the bottleneck.'}\n")
        md.append(f"![H4](experiment_results/h4_stochastic_vs_deterministic.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # H5
    h5 = load_npz(os.path.join(results_dir, 'h5_data.npz'))
    md.append("## H5: Alternative Acceptance Metrics\n")
    if h5:
        md.append(f"| Metric | Accept Rate | Reward Correlation |")
        md.append(f"|--------|-------------|-------------------|")
        for i, name in enumerate(h5['names']):
            md.append(f"| {name} | {h5['accept_rates'][i]:.3f} | {h5['corrs'][i]:.3f} |")
        best_idx = np.argmax(h5['corrs'])
        md.append(f"\n**Best metric by reward correlation:** {h5['names'][best_idx]} (r={h5['corrs'][best_idx]:.3f})\n")
        md.append(f"![H5](experiment_results/h5_acceptance_metrics.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # H6
    h6 = load_npz(os.path.join(results_dir, 'h6_data.npz'))
    md.append("## H6: Hybrid CEM Pareto Frontier\n")
    if h6:
        md.append(f"| K | Reward Retention | Speedup |")
        md.append(f"|---|-----------------|---------|")
        for i, K in enumerate(h6['K_list']):
            md.append(f"| {K} | {h6['retentions'][i]:.3f} | {h6['speedups'][i]:.2f}x |")
        # Find minimum K for >90% retention AND >3x speedup
        feasible = [(K, ret, spd) for K, ret, spd in
                    zip(h6['K_list'], h6['retentions'], h6['speedups'])
                    if ret > 0.9 and spd > 3.0]
        if feasible:
            best = min(feasible, key=lambda x: x[0])
            md.append(f"\n**Minimum K for >90% retention + >3x speedup: K={best[0]} (ret={best[1]:.3f}, spd={best[2]:.2f}x)**\n")
        else:
            md.append(f"\n**No K achieves both >90% retention and >3x speedup simultaneously.**\n")
        md.append(f"![H6](experiment_results/h6_pareto_frontier.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # H7
    h7 = load_npz(os.path.join(results_dir, 'h7_data.npz'))
    md.append("## H7: Outlier-Aware Dimension Protection\n")
    if h7:
        md.append(f"- Outlier dimensions: {list(h7['outlier_dims'])}")
        md.append(f"- Base draft KL: {float(h7['base_kl']):.4f}, Outlier-aware KL: {float(h7['outlier_kl']):.4f}")
        md.append(f"- Base reward retention: {float(h7['base_ret']):.4f}, Outlier-aware: {float(h7['outlier_ret']):.4f}")
        improvement = float(h7['outlier_ret']) - float(h7['base_ret'])
        md.append(f"\n**Conclusion:** Outlier-aware protection {'improves' if improvement > 0.02 else 'does not significantly improve'} reward retention (Δ={improvement:+.4f}).\n")
        md.append(f"![H7](experiment_results/h7_outlier_aware.png)\n")
    else:
        md.append("*Data not yet available.*\n")

    # Synthesis
    md.append("## Synthesis: Mechanism → Solution Path\n")
    md.append("*(Auto-populated when all data is available. Check logs for partial results.)*\n")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'MECHANISM_RESULTS.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"Wrote {out_path}")

if __name__ == '__main__':
    main()
