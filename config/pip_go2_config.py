"""
PIP-Loco Configuration for Unitree Go2 Blind Locomotion.

This config defines hyperparameters for asymmetric actor-critic training
with quadratic barrier constraints for hardware safety.

Author: PIP-Loco Team
"""

import numpy as np
from genesis_lr.legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

class PIPGO2Cfg(GO2Cfg):
    """Environment and physics configuration for PIP-Loco blind locomotion."""

    class env(GO2Cfg.env):
        num_envs = 4096           # Passed to RolloutStorage
        num_observations = 45     # Passed to GenesisWrapper & Modules
        num_privileged_obs = None # Auto-calculated by Wrapper
        episode_length_s = 20.0
        history_len = 50          # Passed to RolloutStorage & VelocityEstimator

    class terrain(GO2Cfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        curriculum = True
        measured_points_x = np.linspace(-0.8, 0.8, 17).tolist()
        measured_points_y = np.linspace(-0.5, 0.5, 11).tolist()

    class commands(GO2Cfg.commands):
        curriculum = False
        resampling_time = 4.0
        class ranges(GO2Cfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]

    class rewards(GO2Cfg.rewards):
        class scales(GO2Cfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.8
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -5.0
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            # PIP-Loco Barriers
            torque_limits = -0.1
            dof_vel_limits = -1.0
            feet_slip = -0.05

    class domain_rand(GO2Cfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0


class PIPGO2CfgPPO(GO2CfgPPO):
    """
    The Master Configuration for PIP-Loco Training.
    Contains ALL hyperparameters for PPO, Dreamer, and Estimator.
    """
    
    class runner(GO2CfgPPO.runner):
        run_name = 'pip-loco_blind'
        experiment_name = 'go2_blind_locomotion'
        max_iterations = 5000     # Total training iterations
        save_interval = 100       # Save model every N iters

    class algorithm(GO2CfgPPO.algorithm):
        # --- Learning Rates (HybridTrainer) ---
        learning_rate_actor = 1.0e-3      # For Actor (PPO)
        learning_rate_critic = 1.0e-3     # For Critic (PPO)
        lr_encoder = 1.0e-4               # For Dreamer & Estimator (Representation)
        
        # --- PPO Hyperparameters (HybridTrainer) ---
        value_loss_coef = 1.0     
        clip_param = 0.2          
        entropy_coef = 0.01       
        num_learning_epochs = 5   # 'num_epochs' in HybridTrainer
        num_mini_batches = 4      # Used to calc 'mini_batch_size' in train.py
        
        # --- GAE Parameters (RolloutStorage.compute_returns) ---
        gamma = 0.99              # Discount factor
        lam = 0.95                # GAE Lambda

        # --- Aux (HybridTrainer) ---
        velocity_indices = [0, 1, 2] # Indices of linear vel in Privileged Obs
        
    class policy(GO2CfgPPO.policy):
        # --- Architecture Hyperparams (ActorCritic / Dreamer / Estimator) ---
        dream_horizon = 5
        estimator_hidden_dims = [128, 64, 32]
        dreamer_hidden_dims = [512, 256, 128]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 1.0
        
    class runner_params:
        # --- Storage Parameters (RolloutStorage) ---
        num_transitions_per_env = 24  # Steps per environment before PPO update