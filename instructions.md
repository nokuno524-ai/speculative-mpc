Deep Dive Project Plan: Speculative Decoding for World Model RolloutsWelcome to the Planning and Robotics Team! 🚀This document is your comprehensive technical roadmap. We are adapting Speculative Decoding for continuous control inside Model Predictive Control (MPC) and Recurrent State-Space Models (RSSMs).Because we operate in continuous, probabilistic latent spaces rather than discrete token spaces, this project requires novel algorithmic design, particularly in Adaptive Acceptance Criteria and Parallel State Verification.1. Technical Context: RSSMs and MPCTo understand the bottleneck we are solving, we must first look at the math underlying our agent.The RSSM ArchitectureIn an RSSM (like Dreamer), the environment state is compressed into a stochastic latent state $s_t$. The world model consists of:Representation Model: $s_t \sim q(s_t | s_{t-1}, a_{t-1}, x_t)$ (encodes pixels $x_t$ into latents).Transition Model (Prior): $\hat{s}_t \sim p(\hat{s}_t | \hat{s}_{t-1}, a_{t-1})$ (predicts the future without seeing pixels).Reward Model: $r_t \sim p(r_t | \hat{s}_t)$.The Planning BottleneckDuring MPC (specifically using the Cross-Entropy Method - CEM), the agent imagines $N$ trajectories of horizon $H$. It relies entirely on the Transition Model to autoregressively unroll states:$$\hat{s}_1 = f(\hat{s}_0, a_0) \rightarrow \hat{s}_2 = f(\hat{s}_1, a_1) \rightarrow \dots \rightarrow \hat{s}_H = f(\hat{s}_{H-1}, a_{H-1})$$For a massive target model with millions of parameters, this sequential $O(H)$ process is too slow for real-time robotic control frequencies (e.g., 50Hz).2. The Speculative Decoding SolutionWe bypass the $O(H)$ sequential bottleneck by introducing a Draft Model (a highly compressed RSSM transition model).The AlgorithmAutoregressive Drafting: The Draft Model quickly generates a trajectory of latents $\hat{s}^{draft}_{1:H}$ and actions $a_{1:H}$.Parallel Verification: The Target Model receives the entire drafted sequence of actions and latents. Through PyTorch batching, it computes its own transition distributions $p_{target}(s_t | \hat{s}^{draft}_{t-1}, a_{t-1})$ for all $t \in [1, H]$ in $O(1)$ wall-clock time.Continuous Acceptance: We compute the Kullback-Leibler (KL) divergence between the Target and Draft distributions at each step.🌟 New Concept: Adaptive Horizon ThresholdingIn discrete text, a token is either right or wrong. In continuous spaces, small distributional errors compound. If we use a fixed KL threshold $\epsilon$, the acceptance rate will drop exponentially as $t$ approaches $H$.Your solution: Implement an Adaptive Threshold $\epsilon_t$ that relaxes slightly as the horizon progresses, acknowledging the natural increase in aleatoric uncertainty.$$\epsilon_t = \epsilon_0 + \alpha \cdot t$$(Where $\epsilon_0$ is the base strictness and $\alpha$ is the relaxation coefficient).3. Implementation Guide & Core Code SnippetsHere are the structural paradigms and PyTorch snippets you will need to build the engine.A. The Parallel VerifierTo verify in parallel, the Target model must take the drafted states and predict the next state, shifted by one timestep.import torch
import torch.distributions as td

def parallel_verify(target_model, draft_states, actions):
    """
    target_model: The large RSSM.
    draft_states: Tensor of shape [Batch, Horizon, Latent_Dim]
    actions: Tensor of shape [Batch, Horizon, Action_Dim]
    """
    B, H, _ = draft_states.shape
    
    # We need to predict state t using draft state t-1.
    # We pad the initial state (s_0) to the beginning of the draft states.
    # Assuming s_0 is known and stored in `initial_states` [B, 1, Latent_Dim]
    # inputs = torch.cat([initial_states, draft_states[:, :-1, :]], dim=1)
    
    # The target model processes the sequence in parallel (requires batched GRU/Linear layers)
    # Output is the parameters of the Normal distribution (mean, std)
    target_means, target_stds = target_model.transition_network(draft_states, actions)
    
    # Create the probability distributions
    target_dists = td.Independent(td.Normal(target_means, target_stds), 1)
    
    return target_dists
B. Continuous Acceptance LogicOnce we have the Target distributions, we compare them against the Draft distributions that generated the states. We use torch.cumprod to find the exact step where the draft diverged too far.def evaluate_and_accept(target_dists, draft_dists, draft_states, eps_base=0.1, alpha=0.01):
    """
    Calculates KL divergence and determines the valid prefix length.
    """
    B, H = draft_states.shape[:2]
    
    # 1. Compute KL Divergence across the batch and horizon
    # Shape: [Batch, Horizon]
    kl_divs = td.kl.kl_divergence(target_dists, draft_dists)
    
    # 2. Create the Adaptive Threshold Tensor
    # e.g., [0.10, 0.11, 0.12, 0.13 ...]
    time_steps = torch.arange(H, device=draft_states.device).float()
    adaptive_thresholds = eps_base + (alpha * time_steps)
    adaptive_thresholds = adaptive_thresholds.unsqueeze(0).expand(B, H)
    
    # 3. Acceptance Mask (True if KL is below threshold)
    accepted_mask = kl_divs < adaptive_thresholds
    
    # 4. Find the contiguous valid prefix using cumulative product
    # If step 2 is False, step 3 onwards must also be False for that batch item
    valid_prefix_mask = torch.cumprod(accepted_mask.int(), dim=1).bool()
    
    # Get the number of accepted steps per batch item
    accepted_lengths = valid_prefix_mask.sum(dim=1)
    
    return valid_prefix_mask, accepted_lengths
C. The CEM Planning Loop IntegrationIn the actual MPC loop, if a sequence is rejected at step $k$, we take the Target model's prediction for step $k$, and trigger the draft model to finish the rest of the sequence $k \rightarrow H$.# Pseudo-code for the outer planning loop
def speculative_rollout(s_0, actions_proposed, H):
    current_states = s_0
    accepted_trajectory = []
    
    step = 0
    while step < H:
        remaining_H = H - step
        
        # 1. Draft the rest of the sequence
        draft_states, draft_dists = draft_model.unroll(current_states, actions_proposed[:, step:], remaining_H)
        
        # 2. Verify in parallel
        target_dists = parallel_verify(target_model, draft_states, actions_proposed[:, step:])
        
        # 3. Accept/Reject using KL logic
        valid_mask, lengths = evaluate_and_accept(target_dists, draft_dists, draft_states)
        
        # 4. Update trajectory and state (Simplified for single batch logic)
        k = lengths[0].item() # Number of accepted steps
        
        if k == remaining_H:
            # Entire draft accepted! We are done.
            accepted_trajectory.append(draft_states)
            break
        else:
            # Accepted up to k. We append the good parts.
            accepted_trajectory.append(draft_states[:, :k])
            
            # For the rejected step k+1, we MUST use the Target model's distribution
            # to resample and correct the trajectory, establishing the new current_state
            corrected_state = target_dists.sample()[:, k] 
            current_states = corrected_state
            step += (k + 1)
            
    return torch.cat(accepted_trajectory, dim=1)
4. Staged Execution PlanPhase 1: Infrastructure & Baselines (Weeks 1-2)Goal: Set up environments and establish target benchmarks.Tasks:Initialize standard WalkerWalk from DeepMind Control Suite.Train an unmodified DreamerV3/RSSM agent (Target).Train a separate, extremely shallow RSSM (Draft) using a distillation loss (KL matching to the pre-trained Target) rather than pure environment loss, to ensure the latents align.Phase 2: Speculative Engine & Batching (Weeks 3-4)Goal: Implement the PyTorch logic for parallel verification.Tasks:Implement the parallel_verify function.Critical: Ensure the Target model's GRU/RNN cells can handle padded sequences in parallel without state-leakage from future timesteps. Use PyTorch torch.nn.utils.rnn.pad_sequence or custom attention masking if using a Transformer-based RSSM.Phase 3: Adaptive Acceptance & Tuning (Weeks 5-6)Goal: Refine the thresholding math.Tasks:Implement the evaluate_and_accept KL logic.Experiment with the $\alpha$ parameter for the Adaptive Horizon Thresholding.Tip: To prevent high variance in drafting, force the Draft Model to output the mean of its distribution rather than sampling during the proposal phase.Phase 4: CEM Integration & Profiling (Weeks 7-8)Goal: Embed the engine into the MPC controller and benchmark.Tasks:Integrate the rollout function into your Cross-Entropy Method action optimizer.Generate the final plot: Wall-Clock Time vs. Planning Horizon ($H$) comparing Target-only vs. Speculative.Target Metric: Maintain 98%+ of baseline reward while achieving $>2.5\times$ latency reduction.5. Recommended ReadingAccelerating Large Language Model Decoding with Speculative Sampling (Chen et al., 2023) - Core algorithm structure.Mastering Diverse Domains through World Models (Hafner et al., 2023) - Understand the specific RSSM Gaussian transition math.Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding (Zhang et al., 2023) - For ideas on skipping the training of a separate draft model and instead skipping layers in the target model.