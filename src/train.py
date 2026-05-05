"""Training loop for target and draft RSSMs."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


def train_rssm(model, dataloader, n_epochs=50, lr=3e-4, device='cpu', 
               kl_weight=1.0, reward_weight=1.0, recon_weight=1.0):
    """Train RSSM model on collected trajectories.
    
    Loss = reconstruction + KL(posterior || prior) + reward prediction
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    metrics_history = []
    
    for epoch in range(n_epochs):
        epoch_losses = {'total': [], 'kl': [], 'reward': [], 'recon': []}
        
        for batch in dataloader:
            obs, actions, rewards, next_obs = [x.to(device) for x in batch]
            B, T, _ = obs.shape
            
            optimizer.zero_grad()
            
            # Forward pass through time
            det, stoch = model.initial_state(B, device)
            kl_loss = 0
            recon_loss = 0
            reward_loss = 0
            n_steps = 0
            
            for t in range(T):
                # Posterior step (with observation)
                det, stoch, posterior = model.observe(obs[:, t], det, stoch, actions[:, t])
                
                # Prior (without observation)
                _, _, prior = model.imagine(det, stoch, torch.zeros_like(actions[:, t]))
                
                # KL divergence
                kl = torch.distributions.kl.kl_divergence(posterior, prior).mean()
                kl_loss += kl
                
                # Reward prediction
                pred_reward = model.get_reward(det, stoch)
                reward_loss += F.mse_loss(pred_reward, rewards[:, t])
                
                n_steps += 1
            
            total_loss = (kl_weight * kl_loss / n_steps + 
                         reward_weight * reward_loss / n_steps)
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['kl'].append((kl_loss / n_steps).item())
            epoch_losses['reward'].append((reward_loss / n_steps).item())
        
        avg = {k: np.mean(v) for k, v in epoch_losses.items()}
        metrics_history.append(avg)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg['total']:.4f} "
                  f"| KL: {avg['kl']:.4f} | Reward: {avg['reward']:.4f}")
    
    return metrics_history


def distill_draft(target_model, draft_model, dataloader, n_epochs=30, lr=1e-3,
                  device='cpu'):
    """Distill target RSSM into draft model via KL matching.
    
    The draft learns to produce similar stochastic states as the target.
    """
    optimizer = torch.optim.Adam(draft_model.parameters(), lr=lr)
    
    # Set projection layers
    draft_model.set_target_stoch_dim(target_model.stoch_dim)
    draft_model.to(device)
    target_model.to(device)
    target_model.eval()
    
    for epoch in range(n_epochs):
        epoch_kl = []
        epoch_align = []
        
        for batch in dataloader:
            obs, actions, rewards, next_obs = [x.to(device) for x in batch]
            B, T, _ = obs.shape
            
            optimizer.zero_grad()
            
            # Run target to get reference stochastic states
            with torch.no_grad():
                tgt_det, tgt_stoch = target_model.initial_state(B, device)
                target_stochs = []
                for t in range(T):
                    tgt_det, tgt_stoch, _ = target_model.observe(
                        obs[:, t], tgt_det, tgt_stoch, actions[:, t])
                    target_stochs.append(tgt_stoch)
                target_stochs = torch.stack(target_stochs, dim=1)  # [B, T, stoch_dim]
            
            # Run draft
            drf_det, drf_stoch = draft_model.initial_state(B, device)
            draft_stochs = []
            kl_loss = 0
            
            for t in range(T):
                drf_det, drf_stoch, prior = draft_model.imagine(
                    drf_det, drf_stoch, actions[:, t])
                
                # Project to target space
                projected = draft_model.project_to_target(drf_stoch)
                draft_stochs.append(projected)
                
                # Alignment loss: match target's stochastic state
                kl_loss += F.mse_loss(projected, target_stochs[:, t])
            
            kl_loss = kl_loss / T
            kl_loss.backward()
            nn.utils.clip_grad_norm_(draft_model.parameters(), 100.0)
            optimizer.step()
            
            epoch_kl.append(kl_loss.item())
        
        if (epoch + 1) % 10 == 0:
            avg_kl = np.mean(epoch_kl)
            print(f"Distill Epoch {epoch+1}/{n_epochs} | Alignment Loss: {avg_kl:.4f}")
    
    return draft_model
