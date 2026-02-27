"""
HybridTrainer: A specialized trainer for PIP-Loco architecture - consisting of a Velocity Estimator, a Dreamer/World Model, and an Actor-Critic.
Enforces strict Gradient Hygiene by using separate optimizers for each component,
ensuring gradients from PPO never flow back into the Estimator or Dreamer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules.pip_actor_critic import ActorCritic
from algorithms.storage import RolloutStorage


class HybridTrainer:

    def __init__(
        self,
        actor_critic: ActorCritic,
        device: str,
        velocity_indices: list = [0, 1, 2],
        lr_encoder: float = 1e-4,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        num_epochs: int = 5,
        mini_batch_size: int = 256,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        self.actor_critic = actor_critic
        self.device = device
        self.velocity_indices = velocity_indices
        self.lr_encoder = lr_encoder
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Three separate optimizers enforce gradient isolation between modules
        self.optimizer_est = optim.Adam(
            self.actor_critic.estimator.parameters(),
            lr=self.lr_encoder
        )
        self.optimizer_dream = optim.Adam(
            self.actor_critic.dreamer.parameters(),
            lr=self.lr_encoder
        )
        self.optimizer_ppo = optim.Adam(
            list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters()),
            lr=self.lr_actor
        )

    def update(self, storage: RolloutStorage) -> dict:
        """
        Performs a complete training update using data from the rollout storage.
        
        Args: storage (RolloutStorage containing collected trajectory data).
        Returns: dict (a dictionary containing mean losses for logging).
        """
        self.actor_critic.train()
        
        total_vel_loss = 0.0
        total_dream_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        
        for epoch in range(self.num_epochs):
            minibatch_generator = storage.generate_minibatch(self.mini_batch_size)
            # 3 updates per minibatch 
            for minibatch in minibatch_generator:
                (
                    obs,
                    privileged_obs,
                    obs_history,
                    actions,
                    next_obs,
                    rewards,
                    returns,
                    dones,
                    values,
                    actions_log_probs,
                    advantages,
                    mu,
                    sigma
                ) = minibatch
                
                obs = obs.to(self.device)
                privileged_obs = privileged_obs.to(self.device)
                obs_history = obs_history.to(self.device)
                actions = actions.to(self.device)
                next_obs = next_obs.to(self.device)
                rewards = rewards.to(self.device)
                returns = returns.to(self.device)
                dones = dones.to(self.device)
                values = values.to(self.device)
                actions_log_probs = actions_log_probs.to(self.device)
                advantages = advantages.to(self.device)
                mu = mu.to(self.device)
                sigma = sigma.to(self.device)
                
                # Estimator update
                true_vel = privileged_obs[:, self.velocity_indices]
                pred_vel = self.actor_critic.estimator(obs_history)
                loss_est = F.mse_loss(pred_vel, true_vel)
                
                self.optimizer_est.zero_grad()
                loss_est.backward()
                self.optimizer_est.step()
                
                # Dreamer update
                pred_next_obs, pred_rewards, pred_actions, pred_values, _ = self.actor_critic.dreamer(obs, actions)
                
                loss_dynamics = F.mse_loss(pred_next_obs, next_obs)
                loss_reward = F.mse_loss(pred_rewards, rewards)
                loss_cloning = F.mse_loss(pred_actions, actions)
                loss_distillation = F.mse_loss(pred_values, values)
                loss_dream = loss_dynamics + loss_reward + loss_cloning + loss_distillation
                
                self.optimizer_dream.zero_grad()
                loss_dream.backward()
                self.optimizer_dream.step()
                
                # PPO Update
                log_probs, entropy = self.actor_critic.evaluate_actions(obs, obs_history, actions)
                value_pred = self.actor_critic.evaluate(privileged_obs)
                
                # Importance sampling ratio
                log_probs = log_probs.view(-1, 1) 
                ratio = torch.exp(log_probs - actions_log_probs)
                
                # Approximate KL divergence
                approx_kl = 0.5 * ((actions_log_probs - log_probs).pow(2).mean())
                
                # Clipped surrogate objective - prevents large policy updates
                surr1 = -advantages * ratio
                surr2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                policy_loss = torch.max(surr1, surr2).mean()
                
                value_loss = (value_pred - returns).pow(2).mean()
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer_ppo.zero_grad()
                total_loss.backward()
                
                # Gradient clipping 
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
                
                self.optimizer_ppo.step()
                
                total_vel_loss += loss_est.item()
                total_dream_loss += loss_dream.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                total_kl += approx_kl.item()
                num_updates += 1
                 
        
        mean_vel_loss = total_vel_loss / num_updates
        mean_dream_loss = total_dream_loss / num_updates
        mean_policy_loss = total_policy_loss / num_updates
        mean_value_loss = total_value_loss / num_updates
        mean_entropy = total_entropy_loss / num_updates
        mean_kl = total_kl / num_updates

        return {
            "loss/velocity": mean_vel_loss,
            "loss/dreamer": mean_dream_loss,
            "loss/policy": mean_policy_loss,
            "loss/value": mean_value_loss,
            "loss/entropy": mean_entropy,
            "loss/kl": mean_kl,
        }
