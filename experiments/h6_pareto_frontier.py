"""H6: Pareto frontier for hybrid CEM — sweep K candidates for target rescoring."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_utils import load_target, load_draft, det_dim, stoch_dim, act_dim

def eval_target(target, det, stoch, actions):
    _, dets, stochs = target.unroll_imagine(det, stoch, actions, deterministic=True)
    return target.get_reward(dets.reshape(-1, det_dim), stochs.reshape(-1, stoch_dim)).reshape(actions.shape[0], -1).sum(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    target = load_target(device)
    draft = load_draft(device)

    H, n_samples, n_iter, n_elite = 12, 512, 5, 51
    K_values = [1, 2, 3, 5, 10, 20]
    n_trials = 20
    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None

    results = {K: {'rewards': [], 'times': []} for K in K_values}
    base_rewards, base_times = [], []

    for trial in range(n_trials):
        torch.manual_seed(trial)
        d0 = torch.randn(1, det_dim, device=device).expand(n_samples, -1).clone()
        s0 = torch.randn(1, stoch_dim, device=device).expand(n_samples, -1).clone()

        # Target-only baseline
        mean = torch.zeros(H, act_dim, device=device)
        std = torch.ones(H, act_dim, device=device) * 0.5
        sync(); t0 = time.perf_counter()
        for _ in range(n_iter):
            acts = (mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_samples, H, act_dim, device=device)).clamp(-1, 1)
            rets = eval_target(target, d0, s0, acts)
            idx = rets.topk(n_elite).indices
            mean = acts[idx].mean(0); std = acts[idx].std(0) + 1e-6
        sync(); bt = time.perf_counter() - t0
        base_rewards.append(eval_target(target, d0[:1], s0[:1], mean.unsqueeze(0)).item())
        base_times.append(bt)

        # Hybrid CEM with K
        for K in K_values:
            m2 = torch.zeros(H, act_dim, device=device)
            s2 = torch.ones(H, act_dim, device=device) * 0.5
            sync(); t0 = time.perf_counter()
            for _ in range(n_iter):
                acts = (m2.unsqueeze(0) + s2.unsqueeze(0) * torch.randn(n_samples, H, act_dim, device=device)).clamp(-1, 1)
                # Draft scoring
                ds, _ = draft(s0[:1].expand(n_samples, -1), acts)
                det_exp = d0[:1].unsqueeze(1).expand(n_samples, H, -1).clone()
                dr = target.get_reward(det_exp.reshape(-1, det_dim), ds.reshape(-1, stoch_dim)).reshape(n_samples, H).sum(1)
                # Rescore top-K
                topk = dr.topk(K).indices
                topk_acts = acts[topk]
                tr = eval_target(target, d0[:K].clone(), s0[:K].clone(), topk_acts)
                combined = dr.clone(); combined[topk] = tr
                idx = combined.topk(n_elite).indices
                m2 = acts[idx].mean(0); s2 = acts[idx].std(0) + 1e-6
            sync(); ht = time.perf_counter() - t0
            results[K]['rewards'].append(eval_target(target, d0[:1], s0[:1], m2.unsqueeze(0)).item())
            results[K]['times'].append(ht)

        if trial % 5 == 0: print(f"Trial {trial}/{n_trials}")

    bm = np.mean(base_rewards); btime = np.mean(base_times)
    print(f"\nBaseline: reward={bm:.2f}, time={btime*1000:.1f}ms")
    print(f"{'K':>4} {'Ret':>8} {'Spd':>8}")

    Ks, rets, spds = [], [], []
    for K in K_values:
        r = np.mean(results[K]['rewards'])
        t = np.mean(results[K]['times'])
        ret = r / bm; spd = btime / t
        Ks.append(K); rets.append(ret); spds.append(spd)
        print(f"{K:>4} {ret:>8.3f} {spd:>8.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(spds, rets, s=100, c=Ks, cmap='viridis', zorder=5)
    for s, r, K in zip(spds, rets, Ks): ax1.annotate(f'K={K}', (s, r), xytext=(5, 5), textcoords='offset points')
    ax1.axhline(0.9, color='r', ls='--', alpha=0.5, label='90% retention')
    ax1.axvline(3.0, color='g', ls='--', alpha=0.5, label='3x speedup')
    ax1.set_xlabel('Speedup'); ax1.set_ylabel('Reward Retention'); ax1.set_title('H6: Pareto Frontier')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(Ks, rets, 'bo-', label='Retention')
    ax2b = ax2.twinx(); ax2b.plot(Ks, spds, 'rs-', label='Speedup')
    ax2.axhline(0.9, color='b', ls='--', alpha=0.3)
    ax2b.axhline(3.0, color='r', ls='--', alpha=0.3)
    ax2.set_xlabel('K'); ax2.set_ylabel('Retention', color='b'); ax2b.set_ylabel('Speedup', color='r')
    ax2.set_title('H6: K vs Performance'); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h6_pareto_frontier.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h6_data.npz'), K_list=Ks, retentions=rets, speedups=spds,
             baseline_reward=bm, baseline_time=btime)
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
