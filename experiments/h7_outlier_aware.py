"""H7: Outlier-aware dimension protection for draft models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_utils import load_target, load_draft, det_dim, stoch_dim, act_dim


class OutlierDraft(nn.Module):
    """Draft with separate heads for outlier vs normal dims."""
    def __init__(self, outlier_dims, stoch_dim=30, act_dim=1, hidden=256):
        super().__init__()
        self.outlier_dims = sorted(outlier_dims)
        self.n_outlier = len(outlier_dims)
        self.n_normal = stoch_dim - self.n_outlier
        self.normal_idx = [i for i in range(stoch_dim) if i not in outlier_dims]

        self.trunk = nn.Sequential(
            nn.Linear(stoch_dim + act_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.normal_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2 * self.n_normal))
        self.outlier_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden // 2), nn.ELU(),
            nn.Linear(hidden // 2, 2 * self.n_outlier))

    def forward(self, stoch_0, actions):
        B, H, _ = actions.shape
        inp = torch.cat([stoch_0.unsqueeze(1).expand(-1, H, -1), actions], -1)
        f = self.trunk(inp.reshape(B * H, -1))
        nm, ns = self.normal_head(f).chunk(2, -1)
        om, os_ = self.outlier_head(f).chunk(2, -1)
        ns = F.softplus(ns) + 0.1; os_ = F.softplus(os_) + 0.1

        mean = torch.zeros(B * H, 30, device=stoch_0.device)
        std = torch.zeros_like(mean)
        mean[:, self.normal_idx] = nm; std[:, self.normal_idx] = ns
        mean[:, self.outlier_dims] = om; std[:, self.outlier_dims] = os_
        mean = mean.reshape(B, H, -1); std = std.reshape(B, H, -1)
        return mean, td.Independent(td.Normal(mean, std), 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    target = load_target(device)
    draft = load_draft(device)

    # Identify outlier dims
    print("=== Finding outlier dims ===")
    with torch.no_grad():
        all_s = []
        for _ in range(200):
            d = torch.randn(1, det_dim, device=device)
            s = torch.randn(1, stoch_dim, device=device)
            a = torch.randn(1, 30, act_dim, device=device).clamp(-1, 1)
            _, _, stochs = target.unroll_imagine(d, s, a, deterministic=False)
            all_s.append(stochs.cpu())
        all_s = torch.cat(all_s).reshape(-1, stoch_dim)
        dim_var = all_s.var(0).numpy()

    n_outlier = max(1, stoch_dim // 5)
    outlier_dims = sorted(np.argsort(dim_var)[-n_outlier:].tolist())
    normal_dims = [i for i in range(stoch_dim) if i not in outlier_dims]
    print(f"Outlier dims: {outlier_dims}, var ratio: {dim_var[outlier_dims].mean()/max(dim_var[normal_dims].mean(),1e-8):.1f}x")

    # Train outlier draft
    print("=== Training outlier-aware draft ===")
    od = OutlierDraft(outlier_dims).to(device)
    opt = torch.optim.Adam(od.parameters(), lr=1e-3)
    H = 20; B = 64
    for step in range(3000):
        d0 = torch.randn(B, det_dim, device=device)
        s0 = torch.randn(B, stoch_dim, device=device)
        a = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)
        with torch.no_grad():
            priors, _, _ = target.unroll_imagine(d0, s0, a, deterministic=True)
            tm = torch.stack([p.mean for p in priors], 1)
        _, od_dists = od(s0, a)
        n_mse = ((od_dists.mean[:, :, normal_dims] - tm[:, :, normal_dims])**2).mean()
        o_mse = ((od_dists.mean[:, :, outlier_dims] - tm[:, :, outlier_dims])**2).mean()
        loss = n_mse + 5.0 * o_mse
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"  Step {step}: {loss.item():.4f}")
    od.eval()

    # Compare
    print("=== Comparing ===")
    H2, B2 = 20, 256
    base_kl, out_kl, base_ret, out_ret = [], [], [], []
    for seed in range(10):
        torch.manual_seed(seed)
        d0 = torch.randn(B2, det_dim, device=device)
        s0 = torch.randn(B2, stoch_dim, device=device)
        a = torch.randn(B2, H2, act_dim, device=device).clamp(-1, 1)
        with torch.no_grad():
            priors, td_arr, ts = target.unroll_imagine(d0, s0, a, deterministic=True)
            tm = torch.stack([p.mean for p in priors], 1)
            ts_ = torch.stack([p.stddev for p in priors], 1)
            t_dist = td.Independent(td.Normal(tm, ts_), 1)
            t_ret = target.get_reward(td_arr.reshape(-1, det_dim), ts.reshape(-1, stoch_dim)).reshape(B2, H2).sum(1)

            _, bd = draft(s0, a)
            _, od_d = od(s0, a)

            base_kl.append(td.kl.kl_divergence(t_dist, bd).mean().item())
            out_kl.append(td.kl.kl_divergence(t_dist, od_d).mean().item())

            det_exp = d0.unsqueeze(1).expand(-1, H2, -1)
            br = target.get_reward(det_exp.reshape(-1, det_dim), bd.mean.reshape(-1, stoch_dim)).reshape(B2, H2).sum(1)
            or_ = target.get_reward(det_exp.reshape(-1, det_dim), od_d.mean.reshape(-1, stoch_dim)).reshape(B2, H2).sum(1)
            base_ret.append((br / t_ret).mean().item())
            out_ret.append((or_ / t_ret).mean().item())

    print(f"{'':20} {'Base':>10} {'Outlier':>10}")
    print(f"{'KL':20} {np.mean(base_kl):>10.4f} {np.mean(out_kl):>10.4f}")
    print(f"{'Reward Ret':20} {np.mean(base_ret):>10.4f} {np.mean(out_ret):>10.4f}")
    print(f"{'Δ Retention':20} {'':>10} {np.mean(out_ret)-np.mean(base_ret):>+10.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    si = np.argsort(dim_var)[::-1]
    colors = ['red' if i in outlier_dims else 'blue' for i in si]
    axes[0].bar(range(stoch_dim), dim_var[si], color=colors, alpha=0.7)
    axes[0].set_title('H7: Dim Variance Spectrum'); axes[0].set_xlabel('Dim (sorted)'); axes[0].set_ylabel('Variance')

    names = ['KL', 'Reward Ret']
    bv = [np.mean(base_kl), np.mean(base_ret)]
    ov = [np.mean(out_kl), np.mean(out_ret)]
    x = np.arange(2); w = 0.35
    axes[1].bar(x - w/2, bv, w, label='Base', color='#1f77b4')
    axes[1].bar(x + w/2, ov, w, label='Outlier-Aware', color='#ff7f0e')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_title('H7: Base vs Outlier'); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].text(0.5, 0.5, f"Δ Retention: {np.mean(out_ret)-np.mean(base_ret):+.4f}",
                 ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
    axes[2].set_title('H7: Summary')

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h7_outlier_aware.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h7_data.npz'), dim_var=dim_var, outlier_dims=outlier_dims,
             base_kl=np.mean(base_kl), outlier_kl=np.mean(out_kl),
             base_ret=np.mean(base_ret), outlier_ret=np.mean(out_ret))
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
