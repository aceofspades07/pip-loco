"""
RolloutStorage: GPU-accelerated replay buffer for PIP-Loco.
Acts as the data bridge between the simulator and the HybridTrainer.
"""

import torch
import numpy as np
from typing import Tuple, Generator


class RolloutStorage:
    """
    High-performance memory buffer for the PIP-Loco RL pipeline.
    Manages data lifecycle: ingestion, GAE computation, and mini-batch serving.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: Tuple[int, ...],
        privileged_obs_shape: Tuple[int, ...],
        actions_shape: Tuple[int, ...],
        history_len: int = 50,
        device: str = "cuda:0",
    ) -> None:
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.history_len = history_len
        self.device = device

        self.step = 0

        # Standard PPO Buffers
        self.obs = torch.zeros(
            num_transitions_per_env, num_envs, obs_shape[0],
            dtype=torch.float32, device=self.device
        )
        self.privileged_obs = torch.zeros(
            num_transitions_per_env, num_envs, privileged_obs_shape[0],
            dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, actions_shape[0],
            dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            dtype=torch.float32, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            dtype=torch.float32, device=self.device
        )
        self.actions_log_probs = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            dtype=torch.float32, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, actions_shape[0],
            dtype=torch.float32, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, actions_shape[0],
            dtype=torch.float32, device=self.device
        )

        # PIP-LOCO Specific Buffers
        # Rolling history window for Velocity Estimator (TCN input)
        self.obs_history = torch.zeros(
            num_transitions_per_env, num_envs, history_len, obs_shape[0],
            dtype=torch.float32, device=self.device
        )
        # Next observation for Dreamer dynamics loss
        self.next_obs = torch.zeros(
            num_transitions_per_env, num_envs, obs_shape[0],
            dtype=torch.float32, device=self.device
        )

        # Computed Targets (filled by compute_returns)
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            dtype=torch.float32, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1,
            dtype=torch.float32, device=self.device
        )

    def add_transitions(
        self,
        obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        obs_history: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> None:
        """
        Ingests a single timestep of experience from all parallel environments.
        Uses in-place copy to prevent memory leaks and graph attachment.
        """
        assert self.step < self.num_transitions_per_env, "Rollout buffer overflow"

        self.obs[self.step].copy_(obs)
        self.privileged_obs[self.step].copy_(privileged_obs)
        self.obs_history[self.step].copy_(obs_history)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.next_obs[self.step].copy_(next_obs)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.actions_log_probs[self.step].copy_(actions_log_prob)
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.step += 1

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> None:
        """
        Computes returns and advantages using Generalized Advantage Estimation (GAE).
        Iterates backwards through the buffer for temporal difference bootstrapping.
        """
        last_gae_lam = 0

        # GAE: backward pass from T-1 to 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            # TD residual: r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = (
                self.rewards[step]
                + gamma * next_values * (1 - self.dones[step])
                - self.values[step]
            )

            # GAE (generalized advantage estimation) recursive formula
            last_gae_lam = (
                delta + gamma * lam * (1 - self.dones[step]) * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

            # Return = Advantage + Value
            self.returns[step] = self.advantages[step] + self.values[step]

    def generate_minibatch(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Generator[
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor,
        ],
        None,
        None,
    ]:
        """
        Yields shuffled mini-batches for training.
        Flattens (num_transitions, num_envs) -> (total_samples) for all buffers.
        """
        assert self.step == self.num_transitions_per_env, "Buffer not full"

        total_samples = self.num_transitions_per_env * self.num_envs
        batch_size = total_samples // num_mini_batches

        # Flatten all buffers: merge (T, N) -> (T*N)
        obs_flat = self.obs.view(total_samples, -1)
        privileged_obs_flat = self.privileged_obs.view(total_samples, -1)
        # obs_history: (T, N, H, D) -> (T*N, H, D)
        obs_history_flat = self.obs_history.view(
            total_samples, self.history_len, self.obs_shape[0]
        )
        actions_flat = self.actions.view(total_samples, -1)
        next_obs_flat = self.next_obs.view(total_samples, -1)
        rewards_flat = self.rewards.view(total_samples, -1)
        returns_flat = self.returns.view(total_samples, -1)
        dones_flat = self.dones.view(total_samples, -1)
        values_flat = self.values.view(total_samples, -1)
        actions_log_probs_flat = self.actions_log_probs.view(total_samples, -1)
        advantages_flat = self.advantages.view(total_samples, -1)
        mu_flat = self.mu.view(total_samples, -1)
        sigma_flat = self.sigma.view(total_samples, -1)

        for _ in range(num_epochs):
            # Shuffle indices for each epoch
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                yield (
                    obs_flat[batch_indices],
                    privileged_obs_flat[batch_indices],
                    obs_history_flat[batch_indices],
                    actions_flat[batch_indices],
                    next_obs_flat[batch_indices],
                    rewards_flat[batch_indices],
                    returns_flat[batch_indices],
                    dones_flat[batch_indices],
                    values_flat[batch_indices],
                    actions_log_probs_flat[batch_indices],
                    advantages_flat[batch_indices],
                    mu_flat[batch_indices],
                    sigma_flat[batch_indices],
                )

    def clear(self) -> None:
        """Resets the step counter for the next rollout collection."""
        self.step = 0
