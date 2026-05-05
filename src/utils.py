"""Utility functions: env wrappers, logging."""
import torch
import numpy as np
import gymnasium as gym


class CartPoleWrapper:
    """Wraps CartPole-v1 for continuous-action world model training.

    Discrete actions (0,1) mapped to continuous (-1, 1).
    """

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = 1  # continuous

    def random_episode(self):
        """Collect one episode with random actions."""
        obs_list, act_list, rew_list = [self.env.reset()[0].astype(np.float32)], [], []
        obs = obs_list[0]
        done = False
        while not done:
            action = int(self.env.action_space.sample())
            cont_action = np.float32(action * 2.0 - 1.0)
            next_obs, reward, term, trunc, _ = self.env.step(action)
            act_list.append(cont_action)
            rew_list.append(np.float32(reward))
            obs_list.append(next_obs.astype(np.float32))
            obs = next_obs
            done = term or trunc
        return np.array(obs_list), np.array(act_list), np.array(rew_list)

    @staticmethod
    def cont_to_discrete(action):
        """Convert continuous action to discrete CartPole action."""
        return int(action > 0)

    def close(self):
        self.env.close()
