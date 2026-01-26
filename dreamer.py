"""
NoLatentModel: A learned world model for MPPI-based locomotion planning.

This module implements a Markovian world model consisting of four independent MLPs
that together enable "dreaming" future trajectories without running a full physics engine.
Used as a lightweight dynamics simulator for model-predictive path integral (MPPI) control.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Type


class NoLatentModel(nn.Module):
    """
    A compound neural network module containing four independent MLPs for world modeling.
    
    This class acts as a learned simulator that predicts state transitions, rewards,
    actions, and values. It is designed to be called inside an MPPI planning loop
    where external action candidates are passed in for dynamics prediction.
    
    Sub-networks:
        - dynamics: Maps (obs, action) -> next_obs
        - reward: Maps (obs, action) -> reward
        - policy: Maps obs -> action
        - value: Maps obs -> value
    
    Attributes:
        obs_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        dynamics: MLP for next-state prediction.
        reward: MLP for reward prediction.
        policy: MLP for action prediction (behavior cloning).
        value: MLP for value estimation.
    """
    
    def __init__(
        self,
        obs_dim: int = 48,
        action_dim: int = 12,
        hidden_dims: List[int] = None,
        activation: Type[nn.Module] = nn.ELU
    ) -> None:
        """
        Initialize the NoLatentModel with four independent MLPs.
        
        Args:
            obs_dim: Size of the observation vector. Default: 48.
            action_dim: Size of the action vector (number of motors). Default: 12.
            hidden_dims: List of hidden layer sizes for all MLPs. Default: [512, 256, 128].
            activation: Activation function class to use between layers. Default: nn.ELU.
        """
        super().__init__()
        
        # Handle mutable default argument
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # Input size for dynamics and reward models (concatenated obs and action)
        state_action_dim = obs_dim + action_dim
        
        # Build the four independent MLPs
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
        Build an MLP with the specified architecture.
        
        Constructs a feedforward network with linear layers and activations.
        The final layer has no activation (raw linear output).
        
        Args:
            input_dim: Size of the input features.
            output_dim: Size of the output features.
            hidden_dims: List of hidden layer sizes.
            activation: Activation function class to instantiate between layers.
        
        Returns:
            nn.Sequential: The constructed MLP.
        """
        layers: List[nn.Module] = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers with activations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def predict_next_state(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the next observation given current observation and action.
        
        Concatenates obs and action along the feature dimension, then passes
        through the dynamics network to predict the next state.
        
        Args:
            obs: Current observation tensor.
            action: Action tensor.
        
        Returns:
            Predicted next observation tensor.
        
        Input Shapes:
            obs: (Batch, obs_dim)
            action: (Batch, action_dim)
        
        Output Shape:
            next_obs: (Batch, obs_dim)
        """
        # Concatenate state and action along feature dimension
        state_action = torch.cat([obs, action], dim=-1)
        next_obs = self.dynamics(state_action)
        return next_obs
    
    def predict_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the immediate reward given current observation and action.
        
        Concatenates obs and action along the feature dimension, then passes
        through the reward network to predict a scalar reward.
        
        Args:
            obs: Current observation tensor.
            action: Action tensor.
        
        Returns:
            Predicted reward tensor.
        
        Input Shapes:
            obs: (Batch, obs_dim)
            action: (Batch, action_dim)
        
        Output Shape:
            reward: (Batch, 1)
        """
        # Concatenate state and action along feature dimension
        state_action = torch.cat([obs, action], dim=-1)
        reward = self.reward(state_action)
        return reward
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the policy action given the current observation.
        
        Passes the observation through the policy network to predict
        the action (behavior cloning of the trained actor).
        
        Args:
            obs: Current observation tensor.
        
        Returns:
            Predicted action tensor.
        
        Input Shape:
            obs: (Batch, obs_dim)
        
        Output Shape:
            action: (Batch, action_dim)
        """
        action = self.policy(obs)
        return action
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the value estimate given the current observation.
        
        Passes the observation through the value network to estimate
        the state value (distilled from the trained critic).
        
        Args:
            obs: Current observation tensor.
        
        Returns:
            Predicted value tensor.
        
        Input Shape:
            obs: (Batch, obs_dim)
        
        Output Shape:
            value: (Batch, 1)
        """
        value = self.value(obs)
        return value
    
    def generate_dreams(self, obs: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Simulate a trajectory of H steps into the future using internal networks.
        Returns a flattened vector to be fed as input to the Actor.
        
        Input Shape:
            obs: (Batch, obs_dim)
        
        Output Shape:
            dreams: (Batch, horizon * obs_dim)
        """
        collected_obs: List[torch.Tensor] = []
        current_obs = obs
        
        for _ in range(horizon):
            a_hat = self.get_action(current_obs)
            next_obs = self.predict_next_state(current_obs, a_hat)
            collected_obs.append(next_obs)
            current_obs = next_obs
        
        # Stack along new dimension and flatten: (Batch, H, Obs_Dim) -> (Batch, H * Obs_Dim)
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
        Forward pass through all four sub-networks.
        
        Computes next state, reward, policy action, and value predictions
        in a single call. If horizon is provided, also generates dreams.
        
        Args:
            obs: Current observation tensor.
            action: Action tensor (used for dynamics and reward prediction).
            horizon: Optional number of steps to dream into the future.
        
        Returns:
            Tuple containing:
                - next_obs: Predicted next observation.
                - reward: Predicted reward.
                - action_pred: Predicted action from policy.
                - value_pred: Predicted value.
                - dreams: Flattened future trajectory or None if horizon not provided.
        
        Input Shapes:
            obs: (Batch, obs_dim)
            action: (Batch, action_dim)
        
        Output Shapes:
            next_obs: (Batch, obs_dim)
            reward: (Batch, 1)
            action_pred: (Batch, action_dim)
            value_pred: (Batch, 1)
            dreams: (Batch, horizon * obs_dim) or None
        """
        next_obs = self.predict_next_state(obs, action)
        reward = self.predict_reward(obs, action)
        action_pred = self.get_action(obs)
        value_pred = self.get_value(obs)
        
        dreams = self.generate_dreams(obs, horizon) if horizon is not None else None
        
        return next_obs, reward, action_pred, value_pred, dreams
