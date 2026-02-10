"""
PIP-Loco Training Script: Orchestrates hybrid RL training for blind quadruped locomotion.

This script is the "Conductor" that coordinates:
  1. Data collection via rollouts in Genesis simulation
  2. Buffer management with GAE-based return computation
  3. Hybrid training updates (Velocity Estimator + Dreamer + PPO)

Architecture Overview:
  - Actor (blind):  proprioception + velocity estimate (TCN) + dreamed futures → actions
  - Critic (privileged): ground-truth physics + terrain heights → value estimate

Author: PIP-Loco Team
"""

import os
import sys
import time
import torch
import numpy as np
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of launch directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.pip_config import PIPGO2Cfg, PIPTrainCfg
from envs.genesis_wrapper import GenesisWrapper
from modules.velocity_estimator import VelocityEstimator
from modules.dreamer import NoLatentModel
from modules.pip_actor_critic import ActorCritic
from algorithms.storage import RolloutStorage
from algorithms.hybrid_trainer import HybridTrainer

import genesis as gs


def train() -> None:
    """
    Main training entry-point for PIP-Loco blind locomotion.

    Execution flow per iteration:
        Phase A – Rollout:   collect experience in parallel Genesis envs
        Phase B – Accounting: compute GAE returns and advantages
        Phase C – Update:     hybrid gradient step (Estimator / Dreamer / PPO)
    """

    # ==================================================================
    # 1. CONFIGURATION
    # ==================================================================
    env_cfg = PIPGO2Cfg()
    train_cfg = PIPTrainCfg()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PIP-Loco] Using device: {device}")

    # Reproducibility
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # Logging directory: logs/<experiment>_<timestamp>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        PROJECT_ROOT, "logs", f"{train_cfg.runner.experiment_name}_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[PIP-Loco] Logging to: {log_dir}")

    # ==================================================================
    # 2. ENVIRONMENT
    # ==================================================================
    # Genesis must be initialized before creating any scene
    gs.init(logging_level="warning")

    # GenesisSimulator reads dt and substeps from sim_params dict
    sim_params = {
        "dt": env_cfg.sim.dt,
        "substeps": env_cfg.sim.substeps,
    }

    env = GenesisWrapper(
        cfg=env_cfg,
        sim_params=sim_params,
        sim_device=device,
        headless=True,
    )

    # Observation / action dimensions
    num_actor_obs = env_cfg.env.num_observations          # 45 (blind proprioception)
    num_actions = env_cfg.env.num_actions                  # 12 (joint position targets)
    # num_privileged_obs is computed inside GenesisWrapper, never trust the config default
    num_critic_obs = env.num_privileged_obs
    print(
        f"[PIP-Loco] Obs dims – actor: {num_actor_obs}, "
        f"critic: {num_critic_obs}, actions: {num_actions}"
    )

    # ==================================================================
    # 3. NEURAL NETWORKS  (the "Hybrid Brain")
    # ==================================================================

    # 3A. Velocity Estimator – TCN that infers body velocity from history
    estimator = VelocityEstimator(
        input_dim=num_actor_obs,
        history_length=env_cfg.env.history_len,
        hidden_dims=train_cfg.estimator.hidden_dims,
        output_dim=train_cfg.estimator.output_dim,
    ).to(device)

    # 3B. Dreamer (World Model) – learned simulator for imagined rollouts
    dreamer = NoLatentModel(
        obs_dim=num_actor_obs,
        action_dim=num_actions,
        hidden_dims=train_cfg.dreamer.hidden_dims,
        activation=torch.nn.ELU,
    ).to(device)

    # 3C. Actor-Critic – owns Estimator & Dreamer, handles input concatenation
    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        estimator=estimator,
        dreamer=dreamer,
        horizon=train_cfg.policy.dreamer_horizon,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=torch.nn.ELU,
        init_noise_std=train_cfg.policy.init_noise_std,
    ).to(device)

    # ==================================================================
    # 4. TRAINER  (gradient-isolated optimizers for each module)
    # ==================================================================
    trainer = HybridTrainer(
        actor_critic=actor_critic,
        device=device,
        velocity_indices=train_cfg.algorithm.velocity_indices,
        lr_encoder=train_cfg.algorithm.lr_encoder,
        lr_actor=train_cfg.algorithm.lr_actor,
        lr_critic=train_cfg.algorithm.lr_critic,
        num_epochs=train_cfg.algorithm.num_learning_epochs,
        mini_batch_size=train_cfg.algorithm.num_mini_batches,
        clip_param=train_cfg.algorithm.clip_param,
        entropy_coef=train_cfg.algorithm.entropy_coef,
        value_loss_coef=train_cfg.algorithm.value_loss_coef,
        max_grad_norm=train_cfg.algorithm.max_grad_norm,
    )

    # ==================================================================
    # 5. ROLLOUT STORAGE  (GPU-resident replay buffer)
    # ==================================================================
    num_steps_per_env = train_cfg.runner.num_steps_per_env

    storage = RolloutStorage(
        num_envs=env_cfg.env.num_envs,
        num_transitions_per_env=num_steps_per_env,
        obs_shape=(num_actor_obs,),
        privileged_obs_shape=(num_critic_obs,),
        actions_shape=(num_actions,),
        history_len=env_cfg.env.history_len,
        device=device,
    )

    # ==================================================================
    # 6. TRAINING LOOP
    # ==================================================================
    max_iterations = train_cfg.runner.max_iterations
    save_interval = train_cfg.runner.save_interval

    # Episode tracking (rewards are summed per-env across an episode)
    ep_reward_buf = deque(maxlen=100)
    current_rewards = torch.zeros(env_cfg.env.num_envs, 1, device=device)

    # Initial environment reset – populates obs buffers via step()
    obs, privileged_obs = env.reset()
    obs = obs.to(device)
    privileged_obs = privileged_obs.to(device)
    obs_history = env.obs_history_buf.clone().to(device)

    total_timesteps = 0
    start_time = time.time()

    print(f"[PIP-Loco] Starting training for {max_iterations} iterations")
    print(f"[PIP-Loco] Rollout length: {num_steps_per_env} steps × {env_cfg.env.num_envs} envs")
    print("=" * 70)

    for iteration in range(max_iterations):
        iter_start = time.time()

        # ==============================================================
        # Phase A: DATA COLLECTION  (rollout in parallel environments)
        # ==============================================================
        actor_critic.eval()

        with torch.no_grad():
            for step in range(num_steps_per_env):
                # --- Inference ---
                actions, log_probs = actor_critic.act(obs, obs_history)
                values = actor_critic.evaluate(privileged_obs)

                # --- Physics step ---
                next_obs, next_privileged_obs, rewards, dones, extras = env.step(actions)

                next_obs = next_obs.to(device)
                next_privileged_obs = next_privileged_obs.to(device)
                rewards = rewards.unsqueeze(-1).to(device) if rewards.dim() == 1 else rewards.to(device)
                dones = dones.unsqueeze(-1).to(device) if dones.dim() == 1 else dones.to(device)

                # Extract updated history from the environment
                next_history = extras["observations_history"].to(device)

                # --- Store transition ---
                # mu/sigma placeholders (not used for importance weighting in this setup)
                mu_placeholder = actions.clone()
                sigma_placeholder = actions.clone()

                storage.add_transitions(
                    obs=obs,
                    privileged_obs=privileged_obs,
                    obs_history=obs_history,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=dones,
                    values=values,
                    actions_log_prob=log_probs,
                    mu=mu_placeholder,
                    sigma=sigma_placeholder,
                )

                # --- Episode reward tracking ---
                current_rewards += rewards
                # Log completed episodes and reset their counters
                done_mask = dones.squeeze(-1).bool()
                if done_mask.any():
                    finished_rewards = current_rewards[done_mask].cpu().numpy().flatten()
                    for r in finished_rewards:
                        ep_reward_buf.append(float(r))
                    current_rewards[done_mask] = 0.0

                # --- Handover: shift state for next step ---
                obs = next_obs
                privileged_obs = next_privileged_obs
                obs_history = next_history

        total_timesteps += num_steps_per_env * env_cfg.env.num_envs

        # ==============================================================
        # Phase B: RETURN COMPUTATION  (GAE advantage estimation)
        # ==============================================================
        with torch.no_grad():
            last_values = actor_critic.evaluate(privileged_obs)

        storage.compute_returns(
            last_values=last_values,
            gamma=train_cfg.algorithm.gamma,
            lam=train_cfg.algorithm.lam,
        )

        # ==============================================================
        # Phase C: HYBRID TRAINING UPDATE
        # ==============================================================
        losses = trainer.update(storage)

        # Clear the buffer immediately so the next rollout starts clean
        storage.clear()

        # ==============================================================
        # 7. LOGGING
        # ==============================================================
        iter_time = time.time() - iter_start
        fps = int(num_steps_per_env * env_cfg.env.num_envs / iter_time)

        if iteration % 10 == 0:
            mean_reward = np.mean(ep_reward_buf) if len(ep_reward_buf) > 0 else 0.0
            elapsed = time.time() - start_time
            print(
                f"Iter {iteration:5d}/{max_iterations} | "
                f"FPS {fps:6d} | "
                f"Reward {mean_reward:8.2f} | "
                f"VelLoss {losses['loss/velocity']:.4f} | "
                f"DreamLoss {losses['loss/dreamer']:.4f} | "
                f"PolicyLoss {losses['loss/policy']:.4f} | "
                f"ValueLoss {losses['loss/value']:.4f} | "
                f"Entropy {losses['loss/entropy']:.4f} | "
                f"KL {losses['loss/kl']:.5f} | "
                f"Time {elapsed:.0f}s"
            )

        # ==============================================================
        # 8. CHECKPOINTING
        # ==============================================================
        if (iteration + 1) % save_interval == 0:
            checkpoint = {
                "iteration": iteration,
                "model_state_dict": actor_critic.state_dict(),
                "optimizer_est": trainer.optimizer_est.state_dict(),
                "optimizer_dream": trainer.optimizer_dream.state_dict(),
                "optimizer_ppo": trainer.optimizer_ppo.state_dict(),
                "total_timesteps": total_timesteps,
            }
            ckpt_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"[PIP-Loco] Checkpoint saved → {ckpt_path}")

    # ==================================================================
    # 9. FINAL SAVE
    # ==================================================================
    final_checkpoint = {
        "iteration": max_iterations - 1,
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_est": trainer.optimizer_est.state_dict(),
        "optimizer_dream": trainer.optimizer_dream.state_dict(),
        "optimizer_ppo": trainer.optimizer_ppo.state_dict(),
        "total_timesteps": total_timesteps,
    }
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(final_checkpoint, final_path)

    elapsed_total = time.time() - start_time
    mean_reward = np.mean(ep_reward_buf) if len(ep_reward_buf) > 0 else 0.0
    print("=" * 70)
    print(f"[PIP-Loco] Training complete.")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Final mean reward: {mean_reward:.2f}")
    print(f"  Wall-clock time : {elapsed_total / 3600:.2f} h")
    print(f"  Model saved to  : {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()
