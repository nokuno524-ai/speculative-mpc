"""H4: Stochastic transitions make verification noisy."""
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

    # Part 1: Rollout variance
    H, N, n_seeds = 30, 100, 5
    stoch_vars, det_vars = [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed + 100)
        d0 = torch.randn(1, det_dim, device=device)
        s0 = torch.randn(1, stoch_dim, device=device)
        acts = torch.randn(1, H, act_dim, device=device).clamp(-1, 1)
        all_s, all_d = [], []
        with torch.no_grad():
            for _ in range(N):
                det, stoch = d0, s0
                ss, ds = [], []
                for t in range(H):
                    det, stoch, _ = target.imagine(det, stoch, acts[:, t])
                    ss.append(stoch); ds.append(det)
                all_s.append(torch.stack(ss, 1)); all_d.append(torch.stack(ds, 1))
        all_s = torch.cat(all_s, 0); all_d = torch.cat(all_d, 0)  # [N, H, dim]
        stoch_vars.append(all_s.var(0).mean(-1).cpu().numpy())  # [H]
        det_vars.append(all_d.var(0).mean(-1).cpu().numpy())

    sv = np.mean(stoch_vars, 0); dv = np.mean(det_vars, 0)
    steps = np.arange(1, H + 1)

    # Part 2: Deterministic vs stochastic acceptance
    H2, B = 20, 256
    det_accept, stoch_accept = [], []
    for seed in range(10):
        torch.manual_seed(seed)
        d0 = torch.randn(B, det_dim, device=device)
        s0 = torch.randn(B, stoch_dim, device=device)
        a = torch.randn(B, H2, act_dim, device=device).clamp(-1, 1)
        with torch.no_grad():
            _, d_dists = draft(s0, a)
            # Deterministic target
            det_priors, _, _ = target.unroll_imagine(d0, s0, a, deterministic=True)
            dm = torch.stack([p.mean for p in det_priors], 1)
            ds = torch.stack([p.stddev for p in det_priors], 1)
            det_td = td.Independent(td.Normal(dm, ds), 1)
            kl_det = td.kl.kl_divergence(det_td, d_dists)
            r1 = ((kl_det < 5.0).int().cumprod(1).sum(1).float().mean() / H2).item()
            det_accept.append(r1)
            # Stochastic target (avg over K)
            K = 10
            means_list, stds_list = [], []
            for _ in range(K):
                det, stoch = d0, s0
                priors = []
                for t in range(H2):
                    det, stoch, p = target.imagine(det, stoch, a[:, t])
                    priors.append(p)
                means_list.append(torch.stack([p.mean for p in priors], 1))
                stds_list.append(torch.stack([p.stddev for p in priors], 1))
            sm = torch.stack(means_list).mean(0)
            ss = torch.stack(stds_list).mean(0)
            st_td = td.Independent(td.Normal(sm, ss), 1)
            kl_st = td.kl.kl_divergence(st_td, d_dists)
            r2 = ((kl_st < 5.0).int().cumprod(1).sum(1).float().mean() / H2).item()
            stoch_accept.append(r2)

    print(f"Det acceptance: {np.mean(det_accept):.3f}, Stoch: {np.mean(stoch_accept):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, sv, 'b-', label='Stochastic state var', linewidth=2)
    ax1.plot(steps, dv, 'r-', label='Deterministic state var', linewidth=2)
    ax1.set_xlabel('Timestep'); ax1.set_ylabel('Variance'); ax1.set_title('H4: Rollout Variance')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.bar(['Deterministic', 'Stochastic'], [np.mean(det_accept), np.mean(stoch_accept)],
            color=['green', 'orange'])
    ax2.set_ylabel('Acceptance Rate'); ax2.set_title('H4: Acceptance'); ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h4_stochastic_vs_deterministic.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h4_data.npz'), steps=steps, sv=sv, dv=dv,
             det_accept=np.mean(det_accept), stoch_accept=np.mean(stoch_accept))
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
