"""
PIP-Loco Configuration: Single Source of Truth for all hyperparameters.

This module defines complete configurations for the PIP-Loco blind locomotion system.
All parameters required to instantiate the following classes are centralized here:
    - GenesisWrapper (Environment & Physics)
    - HybridTrainer (Training Hyperparameters)
    - RolloutStorage (Buffer & Rollout Lengths)
    - VelocityEstimator (Network Architecture)
    - NoLatentModel / Dreamer (World Model Architecture)
    - ActorCritic (Policy Architecture)

Author: PIP-Loco Team
"""

from genesis_lr.legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

class PIPGO2Cfg(GO2Cfg):
    """
    Environment configuration for PIP-Loco blind locomotion.
    
    Inherits from GO2Cfg and overrides/adds parameters specific to:
    - Asymmetric actor-critic observations
    - Hardware safety limits (Quadratic Barrier penalties)
    - 50Hz control loop timing
    """

    class env(GO2Cfg.env):
        """Environment dimensions and episode settings."""
        # Used by: GenesisWrapper, RolloutStorage
        num_envs = 2048                     # Number of parallel environments (adjust based on GPU memory)
        num_observations = 45              # Blind proprioceptive obs (actor input)
        num_privileged_obs = None           # Will be calculated by GenesisWrapper based on terrain config
        num_actions = 12                   # Joint position targets
        env_spacing = 2.0                  # Spacing between environments [m]
        episode_length_s = 20              # Episode duration [s]
        
        # PIP-Loco specific
        history_len = 50                   # Used by: GenesisWrapper, RolloutStorage, VelocityEstimator

    class sim(GO2Cfg.sim if hasattr(GO2Cfg, 'sim') else object):
        """
        Simulation parameters for Genesis.
        
        Physics timing: dt=0.005s with decimation=4 yields 50Hz control.
        """
        # Used by: GenesisWrapper (sim_params)
        dt = 0.005                         # Simulation timestep [s] (200Hz physics)
        substeps = 1                       # Physics substeps per sim step
        gravity = [0., 0., -9.81]          # Gravity vector [m/s^2]
        up_axis = 1                        # Z-up (1) vs Y-up (0)
        use_gpu_pipeline = True            # GPU-accelerated simulation
        
        # Genesis-specific
        max_collision_pairs = 100          # GPU memory vs collision accuracy tradeoff
        IK_max_targets = 2                 # IK solver targets

    class control(GO2Cfg.control):
        """
        PD controller and timing parameters.
        
        Control frequency: 1 / (dt * decimation) = 1 / (0.005 * 4) = 50Hz
        """
        # Used by: GenesisWrapper (sim_params)
        control_type = 'P'                 # Position control
        stiffness = {'joint': 20.0}        # PD stiffness [N*m/rad]
        damping = {'joint': 0.5}           # PD damping [N*m*s/rad]
        action_scale = 0.25                # Action to angle scaling
        dt = 0.005                         # Simulation dt [s] (must match sim.dt)
        decimation = 4                     # Actions repeat for decimation sim steps
        # Effective control frequency: 50Hz = 1/(0.005*4)

    class init_state(GO2Cfg.init_state):
        """Initial robot pose and joint configuration."""
        pos = [0.0, 0.0, 0.42]             # Initial position [m]
        rot = [0.0, 0.0, 0.0, 1.0]         # Initial orientation (quat xyzw)
        lin_vel = [0.0, 0.0, 0.0]          # Initial linear velocity [m/s]
        ang_vel = [0.0, 0.0, 0.0]          # Initial angular velocity [rad/s]
        
        # Default standing pose (symmetric trot-ready)
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class asset(GO2Cfg.asset):
        """Robot URDF and collision settings."""
        name = "go2"
        file = '/home/karan/pip_loco/genesis_lr/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        
        # Joint ordering (matches Unitree SDK)
        dof_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        flip_visual_attachments = False

    class terrain(GO2Cfg.terrain):
        """Terrain generation and height measurement."""
        mesh_type = "plane"                # "plane", "heightfield", or "trimesh"
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
        # Height scan grid for privileged observations
        # Used by: GenesisWrapper (num_privileged_obs calculation)
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        ]  # 17 points
        measured_points_y = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5
        ]  # 11 points
        # Total height scan dim: 17 * 11 = 187 points
        measure_heights = False  # Disabled for blind locomotion on flat plane
        
        # Curriculum settings
        curriculum = False
        max_init_terrain_level = 1

    class commands(GO2Cfg.commands):
        """Velocity command sampling and curriculum."""
        # Used by: GenesisWrapper (command generation)
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4                   # vx, vy, yaw_rate, heading
        resampling_time = 10.0             # Command resample interval [s]
        heading_command = True             # Use heading instead of yaw rate
        curriculum_threshold = 0.8         # Reward threshold to increase difficulty
        
        class ranges(GO2Cfg.commands.ranges):
            """Command sampling ranges."""
            lin_vel_x = [-0.5, 0.5]        # Forward/backward [m/s]
            lin_vel_y = [-1.0, 1.0]        # Lateral [m/s]
            ang_vel_yaw = [-1.0, 1.0]      # Yaw rate [rad/s]
            heading = [-3.14, 3.14]        # Heading angle [rad]

    class domain_rand(GO2Cfg.domain_rand):
        """Domain randomization for sim-to-real transfer."""
        # Friction randomization
        randomize_friction = True
        friction_range = [0.25, 1.25]
        
        # Mass randomization
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]     # [kg]
        
        # External perturbations
        push_robots = True
        push_interval_s = 5.0               # Push interval [s]
        max_push_vel_xy = 1.0              # Max push velocity [m/s]
        
        # Center of mass displacement
        randomize_com_displacement = True
        com_pos_x_range = [-0.05, 0.05]    # [m]
        com_pos_y_range = [-0.05, 0.05]    # [m]
        com_pos_z_range = [-0.05, 0.05]    # [m]
        
        # Control delay (disabled by default for stability)
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 2]
        
        # PD gain randomization
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

    class rewards(GO2Cfg.rewards):
        """
        Reward function configuration.
        
        Includes Quadratic Barrier penalties for hardware safety.
        """
        # Used by: GenesisWrapper (reward computation)
        only_positive_rewards = True
        tracking_sigma = 0.25              # Tracking reward exponential scale
        
        # Soft limits for barrier functions
        # Used by: GenesisWrapper (_reward_dof_vel_limits, _reward_torque_limits)
        soft_dof_pos_limit = 0.9           # % of URDF joint limits
        soft_dof_vel_limit = 0.9           # % of max joint velocity
        soft_torque_limit = 0.9            # % of max torque (SOFT_LIMIT_FACTOR)
        
        # Target heights
        base_height_target = 0.36          # Desired base height [m]
        foot_clearance_target = 0.05       # Desired foot clearance [m]
        foot_height_offset = 0.022         # Foot coordinate origin offset [m]
        foot_clearance_tracking_sigma = 0.01
        
        # Termination condition
        max_projected_gravity = -0.1       # Terminate if robot tips over
        
        class scales(GO2Cfg.rewards.scales):
            """Reward component weights."""
            # Tracking rewards
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            
            # Regularization penalties
            lin_vel_z = -0.5               # Penalize vertical velocity
            ang_vel_xy = -0.05             # Penalize roll/pitch rate
            orientation = -1.0             # Penalize non-upright orientation
            base_height = -2.0             # Penalize height deviation
            
            # Smoothness penalties
            dof_vel = -5.e-4               # Penalize joint velocities
            dof_acc = -2.e-7               # Penalize joint accelerations
            action_rate = -0.01            # Penalize action changes
            action_smoothness = -0.01      # Penalize action jerk
            torques = -2.e-4               # Penalize torque magnitude
            
            # Safety penalties (Quadratic Barrier)
            # Used by: GenesisWrapper (_reward_dof_vel_limits, _reward_torque_limits)
            dof_pos_limits = -10.0         # Joint position limit violation
            dof_vel_limits = -1.0          # Joint velocity limit violation
            torque_limits = -1.0           # Torque limit violation
            collision = -1.0               # Body collision penalty
            
            # Gait rewards
            feet_air_time = 1.0            # Encourage foot lift
            foot_clearance = 0.5           # Encourage ground clearance
            feet_slip = -0.1               # Penalize foot slipping

    class hardware_limits:
        """
        Unitree Go2 hardware safety limits.
        
        Used by: GenesisWrapper (Quadratic Barrier penalty computation)
        """
        max_torque = 45.0                  # Maximum joint torque [Nm]
        max_joint_vel = 30.0               # Maximum joint velocity [rad/s]
        soft_limit_factor = 0.9            # Apply penalties at 90% of limit

    class normalization(GO2Cfg.normalization if hasattr(GO2Cfg, 'normalization') else object):
        """Observation and action normalization."""
        clip_observations = 100.0
        clip_actions = 100.0
        
        class obs_scales:
            """Observation scaling factors for network input normalization."""
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(GO2Cfg.noise if hasattr(GO2Cfg, 'noise') else object):
        """Observation noise for sim-to-real robustness."""
        # Used by: GenesisWrapper (_get_noise_scale_vec)
        add_noise = True
        noise_level = 1.0                  # Global noise scaling
        
        class scales:
            """Per-observation noise magnitudes."""
            dof_pos = 0.01                 # Joint position noise [rad]
            dof_vel = 1.5                  # Joint velocity noise [rad/s]
            lin_vel = 0.1                  # Linear velocity noise [m/s]
            ang_vel = 0.2                  # Angular velocity noise [rad/s]
            gravity = 0.05                 # Gravity projection noise
            height_measurements = 0.1      # Height scan noise [m]

    class viewer(GO2Cfg.viewer if hasattr(GO2Cfg, 'viewer') else object):
        """Visualization settings."""
        ref_env = 0
        pos = [2, 2, 2]                    # Camera position [m]
        lookat = [0., 0, 1.]               # Camera target [m]


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class PIPTrainCfg(GO2CfgPPO):
    """
    Training configuration for PIP-Loco hybrid learning system.
    
    Manages hyperparameters for:
    - HybridTrainer (PPO + Supervised + World Model)
    - RolloutStorage (Experience buffer)
    - Network architectures (Actor, Critic, Estimator, Dreamer)
    """
    
    seed = 1                               # Random seed for reproducibility
    runner_class_name = 'PIPRunner'        # Custom runner for PIP-Loco

    class policy(GO2CfgPPO.policy):
        """
        Actor-Critic network architecture.
        
        Used by: ActorCritic
        """
        # Network dimensions
        # Used by: ActorCritic (actor_hidden_dims, critic_hidden_dims)
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'                 # Activation function
        
        # Action distribution
        # Used by: ActorCritic (init_noise_std)
        init_noise_std = 1.0               # Initial policy std (exploration)
        
        # Dreamer horizon for imagined rollouts
        # Used by: ActorCritic (horizon)
        dreamer_horizon = 5                # Steps of imagined future states

    class estimator:
        """
        Velocity Estimator (TCN) architecture.
        
        Used by: VelocityEstimator
        """
        # Used by: VelocityEstimator.__init__
        input_dim = 45                     # Blind observation dimension
        history_length = 50                # Must match env.history_len
        hidden_dims = [128, 64, 32]        # TCN channel progression
        output_dim = 3                     # Velocity [vx, vy, vz]

    class dreamer:
        """
        World Model (NoLatentModel) architecture.
        
        Used by: NoLatentModel
        """
        # Used by: NoLatentModel.__init__
        obs_dim = 45                       # Observation dimension
        action_dim = 12                    # Action dimension
        hidden_dims = [512, 256, 128]      # MLP hidden layers
        activation = 'elu'                 # Activation function

    class algorithm(GO2CfgPPO.algorithm):
        """
        PPO and auxiliary loss hyperparameters.
        
        Used by: HybridTrainer
        """
        # Learning rates (three separate optimizers)
        # Used by: HybridTrainer (optimizer_est, optimizer_dream, optimizer_ppo)
        lr_encoder = 1e-4                  # Estimator learning rate
        lr_actor = 1e-4         # Actor learning rate
        lr_critic = 1e-4        # Critic learning rate
        
        # PPO hyperparameters
        # Used by: HybridTrainer (update method)
        clip_param = 0.2                   # PPO clipping parameter
        entropy_coef = 0.01                # Entropy bonus coefficient
        value_loss_coef = 0.5              # Value function loss weight
        
        # Training schedule
        # Used by: HybridTrainer (num_epochs)
        num_learning_epochs = 5            # PPO epochs per update
        num_mini_batches = 16              # Mini-batches per epoch
        
        # Gradient clipping
        # Used by: HybridTrainer (gradient clipping)
        max_grad_norm = 1.0                # Max gradient norm for stability
        
        # GAE parameters
        # Used by: RolloutStorage (compute_returns)
        gamma = 0.99                       # Discount factor
        lam = 0.95                         # GAE lambda
        
        # Note: desired_kl is NOT included because HybridTrainer
        # does not implement early stopping based on KL divergence.
        # KL is logged for monitoring but does not affect training.
        
        # Velocity estimation indices
        # Used by: HybridTrainer (true velocity extraction from privileged obs)
        velocity_indices = [0, 1, 2]       # [vx, vy, vz] in privileged_obs

    class runner(GO2CfgPPO.runner):
        """
        Training loop and logging configuration.
        
        Used by: Training script (main loop)
        """
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HybridTrainer'
        
        # Rollout settings
        # Used by: RolloutStorage (num_transitions_per_env)
        num_steps_per_env = 24             # Steps collected before update
        
        # Training duration
        max_iterations = 10000              # Total training iterations
        
        # Checkpointing
        save_interval = 200                # Save model every N iterations
        
        # Experiment tracking
        experiment_name = 'pip_go2'
        run_name = 'pip_loco_blind'
        
        # Logging
        sync_wandb = True                  # Enable Weights & Biases logging
        
        # Resume training
        resume = False
        load_run = -1                      # -1 = last run
        checkpoint = -1                    # -1 = last checkpoint

    class storage:
        """
        Rollout buffer configuration.
        
        Used by: RolloutStorage
        """
        # Buffer dimensions are derived from other configs:
        # - num_envs: from PIPGO2Cfg.env.num_envs
        # - num_transitions_per_env: from runner.num_steps_per_env
        # - obs_shape: (PIPGO2Cfg.env.num_observations,)
        # - privileged_obs_shape: (PIPGO2Cfg.env.num_privileged_obs,)
        # - actions_shape: (PIPGO2Cfg.env.num_actions,)
        # - history_len: from PIPGO2Cfg.env.history_len
        
        # Device for tensor allocation
        # Used by: RolloutStorage (device)
        device = "cuda:0"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_env_cfg() -> PIPGO2Cfg:
    """Returns the environment configuration instance."""
    return PIPGO2Cfg()


def get_train_cfg() -> PIPTrainCfg:
    """Returns the training configuration instance."""
    return PIPTrainCfg()


def print_config_summary():
    """Prints a summary of key configuration values."""
    env_cfg = PIPGO2Cfg()
    train_cfg = PIPTrainCfg()
    
    print("=" * 60)
    print("PIP-Loco Configuration Summary")
    print("=" * 60)
    
    print("\n[Environment]")
    print(f"  num_envs:            {env_cfg.env.num_envs}")
    print(f"  num_observations:    {env_cfg.env.num_observations}")
    print(f"  num_privileged_obs:  {env_cfg.env.num_privileged_obs}")
    print(f"  num_actions:         {env_cfg.env.num_actions}")
    print(f"  history_len:         {env_cfg.env.history_len}")
    
    print("\n[Physics Timing]")
    print(f"  sim.dt:              {env_cfg.sim.dt}s")
    print(f"  control.decimation:  {env_cfg.control.decimation}")
    print(f"  control_freq:        {1.0 / (env_cfg.sim.dt * env_cfg.control.decimation)}Hz")
    
    print("\n[Hardware Limits]")
    print(f"  max_torque:          {env_cfg.hardware_limits.max_torque} Nm")
    print(f"  max_joint_vel:       {env_cfg.hardware_limits.max_joint_vel} rad/s")
    print(f"  soft_limit_factor:   {env_cfg.hardware_limits.soft_limit_factor}")
    
    print("\n[Training]")
    print(f"  lr_encoder:          {train_cfg.algorithm.lr_encoder}")
    print(f"  lr_actor:            {train_cfg.algorithm.lr_actor}")
    print(f"  lr_critic:           {train_cfg.algorithm.lr_critic}")
    print(f"  clip_param:          {train_cfg.algorithm.clip_param}")
    print(f"  max_grad_norm:       {train_cfg.algorithm.max_grad_norm}")
    print(f"  num_epochs:          {train_cfg.algorithm.num_learning_epochs}")
    
    print("\n[Networks]")
    print(f"  actor_hidden_dims:   {train_cfg.policy.actor_hidden_dims}")
    print(f"  critic_hidden_dims:  {train_cfg.policy.critic_hidden_dims}")
    print(f"  estimator_dims:      {train_cfg.estimator.hidden_dims}")
    print(f"  dreamer_dims:        {train_cfg.dreamer.hidden_dims}")
    print(f"  dreamer_horizon:     {train_cfg.policy.dreamer_horizon}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
