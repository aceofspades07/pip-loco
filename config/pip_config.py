"""
PIP-Loco Configuration: A single file to manage all parameters for PIP-Loco

This module defines complete configurations for the PIP-Loco RL pipeline.
All parameters required to instantiate the following classes are centralized here:
    GenesisWrapper (Environment & Physics)
    HybridTrainer (Training Hyperparameters)
    RolloutStorage (Buffer & Rollout Lengths)
    VelocityEstimator (Network Architecture)
    NoLatentModel / Dreamer (World Model Architecture)
    ActorCritic (Policy Architecture)

"""

from pathlib import Path
from genesis_lr.legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

HISTORY_LEN = 50 
class PIPGO2Cfg(GO2Cfg): 
    """
    Environment config class 
    Inherits from GO2Cfg and overrides/adds parameters specific to PIP-Loco.
    """

    class env(GO2Cfg.env):
        # Environment dimensions and episode settings
        num_envs = 1024                     # Number of parallel environments
        num_observations = 45              # Blind obs vector dimension
        num_privileged_obs = None           # Will be calculated by GenesisWrapper based on terrain config
        num_actions = 12                   # Joint positions
        env_spacing = 2.0                  # Spacing between environments (m)
        episode_length_s = 20              # Episode duration (s)
        
        history_len = HISTORY_LEN  # Observation history length for velocity estimator                 

    class sim(GO2Cfg.sim if hasattr(GO2Cfg, 'sim') else object):
        # Simulation parameters for Genesis.
        # Used by sim_params in GenesisWrapper 
        dt = 0.005                         # Simulation timestep (s) 
        substeps = 1                       # Physics substeps per sim step
        gravity = [0., 0., -9.81]          # Gravity vector [m/s^2]
        up_axis = 1                        # 1 for Z-up, 0 for Y-up
        use_gpu_pipeline = True            # Enable GPU physics
        
        # Genesis-specific params
        max_collision_pairs = 100         
        IK_max_targets = 2                 

    class control(GO2Cfg.control):
        """
        PD controller and timing parameters.
        Control frequency = 1 / (dt * decimation)
        """
        # Used by: GenesisWrapper (sim_params)
        control_type = 'P'                 # Position control
        stiffness = {'joint': 20.0}        # PD stiffness 
        damping = {'joint': 0.5}           # PD damping
        action_scale = 0.25                # Action to angle scaling
        dt = 0.005                         # Simulation dt (s) (must match sim.dt)
        decimation = 4                     # No. of sim steps actions are held
        # 50 Hz control

    class init_state(GO2Cfg.init_state):
        # Initial robot pose and joint configuration
        pos = [0.0, 0.0, 0.42]             # Initial position (m)
        rot = [0.0, 0.0, 0.0, 1.0]         # Initial orientation (quat xyzw)
        lin_vel = [0.0, 0.0, 0.0]          # Initial linear velocity (m/s)
        ang_vel = [0.0, 0.0, 0.0]          # Initial angular velocity (rad/s)
        
        # Default standing pose 
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
        # Robot URDF and collision settings

        name = "go2"
        file = str(Path(__file__).parent.parent / 'genesis_lr' / 'resources' / 'robots' / 'go2' / 'urdf' / 'go2.urdf')
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
        # Terrain generation and height measurement
        mesh_type = "plane"                # "plane", "heightfield" or "trimesh"
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
        # Height scan grid for privileged observations
        # Used by: GenesisWrapper for num_privileged_obs calculation
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
        # Velocity command sampling and curriculum
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4                   # vx, vy, yaw_rate, heading
        resampling_time = 10.0             # Command resample interval (s)
        heading_command = True             # Use heading instead of yaw rate
        curriculum_threshold = 0.8         # Reward threshold 
        
        class ranges(GO2Cfg.commands.ranges):
            # Command sampling ranges
            lin_vel_x = [-0.5, 0.5]        # Forward/backward (m/s)
            lin_vel_y = [-1.0, 1.0]        # Lateral (m/s)
            ang_vel_yaw = [-1.0, 1.0]      # Yaw rate (rad/s)
            heading = [-3.14, 3.14]        # Heading angle (rad)

    class domain_rand(GO2Cfg.domain_rand):
        # Change these params to make training tougher

        # Friction randomization
        randomize_friction = True
        friction_range = [0.25, 1.25]
        
        # Mass randomization
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]     # kg
        
        # External perturbations
        push_robots = True
        push_interval_s = 5.0               # Push interval (s)
        max_push_vel_xy = 1.0              # Max push velocity (m/s)
        
        # Center of mass displacement
        randomize_com_displacement = True
        com_pos_x_range = [-0.05, 0.05]    # m
        com_pos_y_range = [-0.05, 0.05]    # m
        com_pos_z_range = [-0.05, 0.05]    # m
        
        # Control delay 
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 2]
        
        # PD gain randomization
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

    class rewards(GO2Cfg.rewards):
        # Reward function configuration.
        only_positive_rewards = True
        tracking_sigma = 0.25 # Tracking reward exponential scale
        
        # Soft limits for barrier functions
        soft_dof_pos_limit = 0.9 # fraction of URDF joint limits
        soft_dof_vel_limit = 0.9 # fraction of max joint velocity
        soft_torque_limit = 0.9  # fraction of max torque 
        
        # Target heights
        base_height_target = 0.36 # Desired base height (m)
        foot_clearance_target = 0.05 # Desired foot clearance (m)
        foot_height_offset = 0.022 # Foot coordinate origin offset (m)
        foot_clearance_tracking_sigma = 0.01
        
        max_projected_gravity = -0.1 # Terminate if robot tips over
        
        class scales(GO2Cfg.rewards.scales):
            # Reward component weights

            # Tracking rewards
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            
            # Regularization penalties
            lin_vel_z = -0.5 # Penalize vertical velocity
            ang_vel_xy = -0.05 # Penalize roll and pitch rate
            orientation = -1.0 # Penalize non-upright orientation
            base_height = -2.0 # Penalize height deviation
            
            # Smoothness penalties
            dof_vel = -5.e-4    # Penalize joint velocities
            dof_acc = -2.e-7    # Penalize joint accelerations
            action_rate = -0.01 # Penalize action changes
            action_smoothness = -0.01 # Penalize action jerk
            torques = -2.e-4    # Penalize torque magnitude
            
            # Quadratic barrier function penalties
            dof_pos_limits = -10.0   # Joint position limit violation
            dof_vel_limits = -1.0   # Joint velocity limit violation
            torque_limits = -1.0    # Torque limit violation
            collision = -1.0    # Body collision penalty
            
            # Gait rewards
            feet_air_time = 1.0            # Encourage foot lift
            foot_clearance = 0.5           # Encourage ground clearance
            feet_slip = -0.1               # Penalize foot slipping

    class hardware_limits:
        """
        Unitree Go2 hardware safety limits.
        Used for monitoring and quadratic barrier penalty computation
        """
        max_torque = 45.0   # Maximum joint torque (Nm)
        max_joint_vel = 30.0    # Maximum joint velocity (rad/s)
        soft_limit_factor = 0.9 # Apply penalties at this much fraction of the limit

    class normalization(GO2Cfg.normalization if hasattr(GO2Cfg, 'normalization') else object):
        # Observation and action normalization
        clip_observations = 100.0
        clip_actions = 100.0
        
        class obs_scales:
            # Observation scaling factors for network input normalization
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(GO2Cfg.noise if hasattr(GO2Cfg, 'noise') else object):
        # Observation noise
        add_noise = True
        noise_level = 1.0                  # Global noise scaling
        
        class scales:
            # Per-observation noise magnitudes
            dof_pos = 0.01                 # Joint position noise (rad)
            dof_vel = 1.5                  # Joint velocity noise (rad/s)
            lin_vel = 0.1                  # Linear velocity noise (m/s)
            ang_vel = 0.2                  # Angular velocity noise (rad/s)
            gravity = 0.05                 # Gravity projection noise
            height_measurements = 0.1      # Height scan noise (m)

    class viewer(GO2Cfg.viewer if hasattr(GO2Cfg, 'viewer') else object):
        # Visualization settings
        ref_env = 0
        pos = [2, 2, 2]                    # Camera position (m)
        lookat = [0., 0, 1.]               # Camera target (m)



class PIPTrainCfg(GO2CfgPPO):
    """
    Training configuration for PIP-Loco hybrid learning system.
    Manages hyperparameters for training the ActorCritic policy, VelocityEstimator, and NoLatentModel world model.
    """
    
    seed = 1                               
    runner_class_name = 'PIPRunner' # Custom runner for PIP-Loco

    class policy(GO2CfgPPO.policy):
        # Actor-Critic network architecture.

        # Network dimensions
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # Activation function
        
        init_noise_std = 1.0 # Initial policy std deviation for exploration
        
        # Dreamer horizon for imagined rollouts
        dreamer_horizon = 5 

    class estimator:
        # Velocity estimator architecture
        input_dim = 45 # Blind observation dimension
        history_length = HISTORY_LEN
        hidden_dims = [128, 64, 32] # TCN channel progression
        output_dim = 3 # Velocity (vx, vy, vz)

    class dreamer:
        # Dreamer architecture (NoLatentModel)
        obs_dim = 45 # Observation dimension
        action_dim = 12 # Action dimension
        hidden_dims = [512, 256, 128] # MLP hidden layers
        activation = 'elu' # Activation function

    class algorithm(GO2CfgPPO.algorithm):
        # PPO and auxiliary loss hyperparameters

        # Learning rates for 3 separate optimizers
        lr_encoder = 1e-4   # Estimator and Dreamer
        lr_actor = 1e-4     # Actor
        lr_critic = 1e-4    # Critic
        
        # PPO hyperparameters
        clip_param = 0.2        # PPO clipping parameter
        entropy_coef = 0.005    # Entropy bonus coefficient
        value_loss_coef = 0.5   # Value function loss weight
        
        # Training schedule
        num_learning_epochs = 5 # PPO epochs per update
        num_mini_batches = 4    # Mini-batches per epoch
        
        # Gradient clipping
        max_grad_norm = 1.0     # Max gradient norm for stability
        
        # GAE parameters
        gamma = 0.99    # Discount factor
        lam = 0.95      # GAE lambda

        # KL divergence is logged for monitoring but does not affect training at all
        
        # True velocity indices in privileged_obs
        velocity_indices = [0, 1, 2]

    class runner(GO2CfgPPO.runner):
        # Training loop and logging configuration.

        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HybridTrainer'
        
        num_steps_per_env = 24 # Steps collected before update
        
        max_iterations = 5000 # Total training iterations
        
        save_interval = 200 # Save model every N iterations
        
        # Experiment tracking
        experiment_name = 'pip_go2'
        run_name = 'pip-loco_blind'
        
        # Resume training
        resume = False
        load_run = -1                      # -1 = last run
        checkpoint = -1                    # -1 = last checkpoint

    class storage:
        # Device for tensor allocation
        device = "cuda:0"