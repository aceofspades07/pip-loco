"""
PIP-Loco Asymmetric Actor-Critic Network
PPO-based architecture with blind actor and privileged critic.
"""

import math
from typing import List, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Asymmetric Actor-Critic for blind quadruped locomotion.
    
    Actor: Processes partial observations (proprioception + velocity estimate + dreamed trajectory)
    Critic: Processes privileged simulation state (friction, terrain, true velocity, etc.)
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: List[int] = [512, 256, 128],
        critic_hidden_dims: List[int] = [512, 256, 128],
        activation: Type[nn.Module] = nn.ELU,
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        # Build actor MLP (outputs action mean)
        self.actor = self._build_mlp(
            input_dim=num_actor_obs,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        # Build critic MLP (outputs scalar value)
        self.critic = self._build_mlp(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # Learnable log standard deviation (initialize so exp(log_std) = init_noise_std)
        self.log_std = nn.Parameter(
            torch.log(torch.ones(num_actions) * init_noise_std)
        )

        # Apply weight initialization
        self._init_weights()

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Type[nn.Module],
    ) -> nn.Sequential:
        """Construct MLP with specified architecture. Final layer has no activation."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """
        Orthogonal initialization with gain=sqrt(2) for hidden layers,
        gain=0.01 for final layers to keep initial outputs near zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Final layer initialization with small gain
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=0.01)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic action selection for training.
        Returns sampled action and its log probability.
        """
        mean = self.actor(obs)
        std = self.log_std.exp()
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions under current policy.
        Used for PPO loss computation.
        """
        mean = self.actor(obs)
        std = self.log_std.exp()
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1).mean()
        return log_prob, entropy
    
    
    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from privileged observations."""
        return self.critic(critic_obs)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action selection for deployment (returns mean)."""
        return self.actor(obs)
