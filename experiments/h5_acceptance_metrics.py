"""H5: Alternative acceptance metrics for speculative decoding."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributions as td
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_utils import load_target, load_draft, det_dim, stoch_dim, act_dim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    target = load_target(device)
    draft = load_draft(device)

    H, B, n_seeds = 20, 256, 10
    metrics = {n: {'accept': [], 'corr': []} for n in ['KL', 'MSE', 'Wasserstein', 'Overlap']}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        d0 = torch.randn(B, det_dim, device=device)
        s0 = torch.randn(B, stoch_dim, device=device)
        a = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

        with torch.no_grad():
            priors, td_arr, ts = target.unroll_imagine(d0, s0, a, deterministic=True)
            tm = torch.stack([p.mean for p in priors], 1)
            ts_std = torch.stack([p.stddev for p in priors], 1)
            t_dists = td.Independent(td.Normal(tm, ts_std), 1)
            t_ret = target.get_reward(td_arr.reshape(-1, det_dim), ts.reshape(-1, stoch_dim)).reshape(B, H).sum(1)

            _, d_dists = draft(s0, a)
            dm, ds = d_dists.mean, d_dists.stddev
            det_exp = d0.unsqueeze(1).expand(-1, H, -1)
            d_ret = target.get_reward(det_exp.reshape(-1, det_dim), dm.reshape(-1, stoch_dim)).reshape(B, H).sum(1)

            kl = td.kl.kl_divergence(t_dists, d_dists)
            mse = ((tm - dm)**2).sum(-1)
            wass = ((tm - dm)**2 + (ts_std - ds)**2).sum(-1)
            overlap = 1 - torch.exp(-0.25 * (tm - dm)**2 / (ts_std**2 + ds**2 + 1e-8)).mean(-1)

            for name, per_step in [('KL', kl), ('MSE', mse), ('Wasserstein', wass), ('Overlap', overlap)]:
                thresh = per_step.median()
                accepted = (per_step < thresh).int()
                rate = torch.cumprod(accepted, 1).sum().float() / (B * H)
                avg = per_step.mean(1)
                r_err = (d_ret - t_ret).abs()
                corr = torch.corrcoef(torch.stack([avg, r_err]))[0, 1].item()
                metrics[name]['accept'].append(rate.item())
                metrics[name]['corr'].append(corr)

    names = list(metrics.keys())
    ar = [np.mean(metrics[n]['accept']) for n in names]
    co = [np.mean(metrics[n]['corr']) for n in names]

    print(f"{'Metric':<15} {'Accept':>8} {'Corr':>8}")
    for n, a, c in zip(names, ar, co):
        print(f"{n:<15} {a:>8.3f} {c:>8.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax1.bar(names, ar, color=colors)
    ax1.set_ylabel('Accept Rate'); ax1.set_title('H5: Acceptance Rate'); ax1.grid(True, alpha=0.3, axis='y')
    ax2.bar(names, co, color=colors)
    ax2.set_ylabel('Correlation w/ Reward Error'); ax2.set_title('H5: Metric Quality'); ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h5_acceptance_metrics.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h5_data.npz'), names=names, accept_rates=ar, corrs=co)
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
