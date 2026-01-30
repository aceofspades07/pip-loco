"""
NoLatentModel: Markovian world model with four MLPs for MPPI planning.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Type


class NoLatentModel(nn.Module):
    """
    Four independent MLPs: dynamics (obs, action → next_obs), reward (obs, action → r),
    policy (obs → action), value (obs → v). Minimal learned simulator for MPPI.
    """
    
    def __init__(
        self,
        obs_dim: int = 48,
        action_dim: int = 12,
        hidden_dims: List[int] = None,
        activation: Type[nn.Module] = nn.ELU
    ) -> None:
        """
        Initialize sub-networks. Args: obs_dim, action_dim, hidden_dims, activation.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        state_action_dim = obs_dim + action_dim
        self.dynamics = self._build_mlp(
            input_dim=state_action_dim,
            output_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        
        self.reward = self._build_mlp(
            input_dim=state_action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation
        )
        
        self.policy = self._build_mlp(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        
        self.value = self._build_mlp(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation
        )
    
    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Type[nn.Module]
    ) -> nn.Sequential:
        """
        Linear-activation stacks; final layer is linear. Returns nn.Sequential.
        """
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def predict_next_state(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Concatenate along last dim. Input: obs (B, O), action (B, A). Output: (B, O).
        """
        state_action = torch.cat([obs, action], dim=-1)
        next_obs = self.dynamics(state_action)
        return next_obs
    
    def predict_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Concatenate along last dim. Input: obs (B, O), action (B, A). Output: (B, 1).
        """
        state_action = torch.cat([obs, action], dim=-1)
        reward = self.reward(state_action)
        return reward
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Input: obs (B, O). Output: action (B, A).
        """
        action = self.policy(obs)
        return action
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Input: obs (B, O). Output: value (B, 1).
        """
        value = self.value(obs)
        return value
    
    def generate_dreams(self, obs: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Rollout H steps: a_hat = policy(obs); next_obs = dynamics([obs, a_hat]).
        Stack to (B, H, O) and flatten to (B, H*O) for Actor input.
        """
        collected_obs: List[torch.Tensor] = []
        current_obs = obs
        
        for _ in range(horizon):
            a_hat = self.get_action(current_obs)
            next_obs = self.predict_next_state(current_obs, a_hat)
            collected_obs.append(next_obs)
            current_obs = next_obs
        stacked = torch.stack(collected_obs, dim=1)
        dreams = stacked.flatten(start_dim=1)
        return dreams
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        horizon: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predictions plus optional dreams when horizon is provided.
        Inputs: obs (B, O), action (B, A). Outputs: next_obs (B, O), reward (B, 1),
        action_pred (B, A), value_pred (B, 1), dreams (B, H*O) or None.
        """
        next_obs = self.predict_next_state(obs, action)
        reward = self.predict_reward(obs, action)
        action_pred = self.get_action(obs)
        value_pred = self.get_value(obs)
        
        dreams = self.generate_dreams(obs, horizon) if horizon is not None else None
        
        return next_obs, reward, action_pred, value_pred, dreams
