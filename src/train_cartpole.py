"""Quick CartPole training script for speculative decoding experiments."""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import os

from src.rssm import RSSM
from src.draft_model import DraftRSSM
from src.train import train_rssm, distill_draft
from src.acceptance import speculative_rollout


def collect_data(env_name='CartPole-v1', n_episodes=200, max_steps=500):
    """Collect trajectories from random policy."""
    env = gym.make(env_name)
    all_obs, all_actions, all_rewards, all_next_obs = [], [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for t in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            all_obs.append(obs.astype(np.float32))
            all_actions.append(np.array([action], dtype=np.float32))
            all_rewards.append(np.array([reward], dtype=np.float32))
            all_next_obs.append(next_obs.astype(np.float32))
            
            obs = next_obs
            if terminated or truncated:
                break
    
    env.close()
    return (
        torch.tensor(np.array(all_obs)),
        torch.tensor(np.array(all_actions)),
        torch.tensor(np.array(all_rewards)),
        torch.tensor(np.array(all_next_obs)),
    )


def make_sequences(obs, actions, rewards, next_obs, seq_len=20):
    """Convert flat data into sequences for RNN training."""
    n = len(obs) - seq_len
    obs_seq = torch.stack([obs[i:i+seq_len] for i in range(0, n, seq_len)])
    act_seq = torch.stack([actions[i:i+seq_len] for i in range(0, n, seq_len)])
    rew_seq = torch.stack([rewards[i:i+seq_len] for i in range(0, n, seq_len)])
    nxt_seq = torch.stack([next_obs[i:i+seq_len] for i in range(0, n, seq_len)])
    return obs_seq, act_seq, rew_seq, nxt_seq


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Hyperparameters
    OBS_DIM = 4      # CartPole observation dim
    ACT_DIM = 1      # CartPole action dim (discrete encoded as scalar)
    DET_DIM = 64
    STOCH_DIM = 16
    HIDDEN_DIM = 64
    
    # Step 1: Collect data
    print("Collecting CartPole data...")
    obs, actions, rewards, next_obs = collect_data(n_episodes=100)
    print(f"Collected {len(obs)} transitions")
    
    # Make sequences
    obs_seq, act_seq, rew_seq, nxt_seq = make_sequences(obs, actions, rewards, next_obs, seq_len=20)
    dataset = TensorDataset(obs_seq, act_seq, rew_seq, nxt_seq)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Step 2: Train target RSSM
    print("\n=== Training Target RSSM ===")
    target = RSSM(OBS_DIM, ACT_DIM, det_dim=DET_DIM, stoch_dim=STOCH_DIM, hidden_dim=HIDDEN_DIM)
    n_params = sum(p.numel() for p in target.parameters())
    print(f"Target params: {n_params:,}")
    train_rssm(target, dataloader, n_epochs=50, lr=3e-4, device=device)
    
    # Step 3: Train draft RSSM (4x smaller)
    print("\n=== Distilling Draft RSSM ===")
    draft = DraftRSSM(OBS_DIM, ACT_DIM, det_dim=DET_DIM//4, stoch_dim=STOCH_DIM//4, hidden_dim=HIDDEN_DIM//4)
    n_params_draft = sum(p.numel() for p in draft.parameters())
    print(f"Draft params: {n_params_draft:,} ({n_params_draft/n_params:.1%} of target)")
    distill_draft(target, draft, dataloader, n_epochs=30, lr=1e-3, device=device)
    
    # Step 4: Benchmark speculative decoding
    print("\n=== Benchmarking Speculative Decoding ===")
    target.eval()
    draft.eval()
    
    HORIZON = 12
    N_BATCH = 64
    
    # Create test data
    test_obs = obs[:N_BATCH].to(device)
    test_actions = torch.randn(N_BATCH, HORIZON, ACT_DIM, device=device)
    
    # Initialize states
    det_0, stoch_0 = target.initial_state(N_BATCH, device)
    
    # Target-only baseline timing
    times_target = []
    for _ in range(20):
        t0 = time.time()
        det, stoch = det_0.clone(), stoch_0.clone()
        for t in range(HORIZON):
            det, stoch, _ = target.imagine(det, stoch, test_actions[:, t])
        torch.cuda.synchronize() if device == 'cuda' else None
        times_target.append(time.time() - t0)
    
    target_time = np.median(times_target)
    
    # Speculative timing
    times_spec = []
    for _ in range(20):
        t0 = time.time()
        with torch.no_grad():
            final_dets, final_stochs, stats = speculative_rollout(
                target, draft, det_0, stoch_0, test_actions,
                eps_base=0.5, alpha=0.02
            )
        torch.cuda.synchronize() if device == 'cuda' else None
        times_spec.append(time.time() - t0)
    
    spec_time = np.median(times_spec)
    speedup = target_time / max(spec_time, 1e-8)
    
    # Reward comparison
    with torch.no_grad():
        # Target-only rewards
        det, stoch = det_0.clone(), stoch_0.clone()
        target_rewards = []
        for t in range(HORIZON):
            det, stoch, _ = target.imagine(det, stoch, test_actions[:, t])
            target_rewards.append(target.get_reward(det, stoch))
        target_returns = torch.stack(target_rewards, dim=1).sum(dim=1)
        
        # Speculative rewards (re-evaluate accepted trajectory)
        spec_dets, spec_stochs, stats = speculative_rollout(
            target, draft, det_0, stoch_0, test_actions, eps_base=0.5, alpha=0.02
        )
        spec_rewards = []
        for t in range(HORIZON):
            r = target.get_reward(spec_dets[:, t], spec_stochs[:, t])
            spec_rewards.append(r)
        spec_returns = torch.stack(spec_rewards, dim=1).sum(dim=1)
    
    reward_ratio = (spec_returns / (target_returns.abs() + 1e-8)).mean().item()
    reward_corr = torch.corrcoef(torch.stack([target_returns, spec_returns]))[0, 1].item()
    
    results = {
        'target_params': n_params,
        'draft_params': n_params_draft,
        'size_ratio': n_params_draft / n_params,
        'target_time_ms': target_time * 1000,
        'speculative_time_ms': spec_time * 1000,
        'speedup': speedup,
        'acceptance_rate': stats['avg_acceptance_rate'],
        'mean_kl': stats['mean_kl'],
        'reward_ratio': reward_ratio,
        'reward_correlation': reward_corr,
        'horizon': HORIZON,
        'n_batch': N_BATCH,
    }
    
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Target params: {n_params:,}")
    print(f"Draft params:  {n_params_draft:,} ({n_params_draft/n_params:.1%} of target)")
    print(f"Target-only time:    {target_time*1000:.2f} ms")
    print(f"Speculative time:    {spec_time*1000:.2f} ms")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Acceptance rate:     {stats['avg_acceptance_rate']:.2%}")
    print(f"Mean KL:             {stats['mean_kl']:.4f}")
    print(f"Reward ratio:        {reward_ratio:.4f}")
    print(f"Reward correlation:  {reward_corr:.4f}")
    print(f"{'='*50}")
    
    # Save results
    os.makedirs('/scratch/qzp4ta/speculative-mpc/results', exist_ok=True)
    with open('/scratch/qzp4ta/speculative-mpc/results/benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/benchmark.json")
    
    # Save models
    torch.save(target.state_dict(), '/scratch/qzp4ta/speculative-mpc/results/target_rssm.pt')
    torch.save(draft.state_dict(), '/scratch/qzp4ta/speculative-mpc/results/draft_rssm.pt')
    print("Models saved.")


if __name__ == '__main__':
    main()
