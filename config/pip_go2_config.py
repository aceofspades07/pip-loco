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
        num_envs = 4096  # High throughput for GPU-accelerated training
        num_observations = 45  # Blind proprioceptive obs (no velocity/terrain)
        num_privileged_obs = None  # Auto-calculated by GenesisWrapper
        episode_length_s = 20.0  # Episode duration in seconds
        history_len = 50  # Observation history buffer length for velocity estimation

    class terrain(GO2Cfg.terrain):
        mesh_type = 'trimesh'  # Use heightfield mesh for rough terrain
        measure_heights = True  # Enable height scanner for privileged critic obs
        curriculum = True  # Progressive terrain difficulty
        measured_points_x = np.linspace(-0.8, 0.8, 17).tolist()  # 1.6m scan range X
        measured_points_y = np.linspace(-0.5, 0.5, 11).tolist()  # 1.0m scan range Y

    class commands(GO2Cfg.commands):
        curriculum = False  # Randomize commands immediately (no curriculum)
        resampling_time = 4.0  # New velocity command every 4 seconds

        class ranges(GO2Cfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # Forward/backward velocity [m/s]
            lin_vel_y = [-1.0, 1.0]  # Lateral velocity [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # Yaw rate [rad/s]

    class rewards(GO2Cfg.rewards):
        """Reward scales for locomotion training with hardware safety barriers."""

        class scales(GO2Cfg.rewards.scales):
            # === Standard Locomotion Rewards ===
            tracking_lin_vel = 1.5  # Primary: track commanded linear velocity
            tracking_ang_vel = 0.8  # Track commanded angular velocity
            lin_vel_z = -2.0  # Penalize vertical oscillation (hopping)
            ang_vel_xy = -0.05  # Penalize roll/pitch angular velocity
            orientation = -1.0  # Penalize deviation from upright orientation
            base_height = -5.0  # Strict penalty for incorrect body height
            feet_air_time = 1.0  # Encourage periodic foot lifting (gait)
            collision = -1.0  # Penalize knee/body collisions
            action_rate = -0.01  # Penalize jerky/high-frequency actions

            # === PIP-Loco Quadratic Barrier Constraints ===
            torque_limits = -0.1  # Soft barrier for motor torque limits
            dof_vel_limits = -1.0  # Soft barrier for joint velocity limits
            feet_slip = -0.05  # Penalize foot slipping during contact

    class domain_rand(GO2Cfg.domain_rand):
        randomize_friction = True  # Randomize ground friction coefficient
        friction_range = [0.5, 1.25]  # Friction coefficient range
        randomize_base_mass = True  # Randomize payload mass
        added_mass_range = [-1.0, 1.0]  # Added mass range [kg]
        push_robots = True  # Apply random external pushes
        push_interval_s = 10  # Push interval [seconds]
        max_push_vel_xy = 1.0  # Maximum push velocity [m/s]


class PIPGO2CfgPPO(GO2CfgPPO):
    """PPO training hyperparameters for PIP-Loco blind locomotion."""

    class runner(GO2CfgPPO.runner):
        run_name = 'pip_loco_blind'  # Experiment run identifier
        experiment_name = 'go2_blind_locomotion'  # Top-level experiment name
        max_iterations = 5000  # Total training iterations
        save_interval = 100  # Checkpoint save frequency [iterations]

    class algorithm(GO2CfgPPO.algorithm):
        entropy_coef = 0.01  # Entropy bonus for exploration
        learning_rate = 1.0e-3  # Adam optimizer learning rate
        num_learning_epochs = 5  # PPO epochs per rollout batch
