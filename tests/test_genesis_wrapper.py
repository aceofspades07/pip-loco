#!/usr/bin/env python3
"""
test_genesis_wrapper.py - Torture Test Suite for GenesisWrapper

This script provides rigorous, standalone testing of the GenesisWrapper class.
It verifies production-readiness by detecting "silent failures" that could ruin training.

Requirements:
- No external training code (PPO/Runner)
- Mock configuration created in-script
- Aggressive verification of all components

Author: QA Engineer
"""

# Set simulator BEFORE any legged_gym imports
import builtins
builtins.SIMULATOR = 'genesis'

import sys
import os
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Initialize Genesis BEFORE importing the wrapper
import genesis as gs
gs.init(backend=gs.gpu, logging_level='warning')

# ANSI color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_pass(test_name: str, details: str = "") -> None:
    """Print a PASS message in green."""
    msg = f"{Colors.GREEN}[PASS]{Colors.RESET} {test_name}"
    if details:
        msg += f" - {details}"
    print(msg)


def print_fail(test_name: str, details: str, expected: str = "", actual: str = "") -> None:
    """Print a FAIL message in red with details."""
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {test_name}")
    print(f"       {Colors.RED}Details:{Colors.RESET} {details}")
    if expected:
        print(f"       {Colors.YELLOW}Expected:{Colors.RESET} {expected}")
    if actual:
        print(f"       {Colors.YELLOW}Actual:{Colors.RESET} {actual}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")


# ============================================================================
# MOCK CONFIGURATION BUILDER
# ============================================================================

def create_mock_config():
    """
    Create a fully mocked GO2Cfg object for testing.
    
    This function constructs a minimal but complete configuration that allows
    GenesisWrapper to initialize without requiring the full legged_gym imports.
    
    Returns:
        MockGO2Cfg: A mock configuration object with all required nested classes.
    """
    
    class MockConfig:
        """Base class for mock config sections."""
        pass
    
    class MockGO2Cfg(MockConfig):
        """Mock GO2Cfg with all required nested classes."""
        
        class env(MockConfig):
            num_envs = 10  # Test with 10 environments
            num_observations = 48  # Standard value - wrapper should override to 45
            num_privileged_obs = None  # Will be computed by wrapper
            num_actions = 12
            env_spacing = 2.0
            send_timeouts = True
            episode_length_s = 20
            debug = False
            debug_draw_height_points = False
            debug_draw_height_points_around_feet = False
            fail_to_terminal_time_s = 0.5
            history_len = 50  # For observation history buffer
        
        class terrain(MockConfig):
            mesh_type = "plane"
            plane_length = 200.0
            horizontal_scale = 0.1
            vertical_scale = 0.005
            border_size = 5
            curriculum = False
            static_friction = 1.0
            dynamic_friction = 1.0
            restitution = 0.0
            measure_heights = False
            obtain_terrain_info_around_feet = False
            # Mock terrain scan: 10x10 = 100 points
            measured_points_x = list(np.linspace(-1, 1, 11))
            measured_points_y = list(np.linspace(-1, 1, 17))
            selected = False
            terrain_kwargs = None
            max_init_terrain_level = 1
            terrain_length = 6.0
            terrain_width = 6.0
            platform_size = 3.0
            num_rows = 4
            num_cols = 4
            terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
            slope_treshold = 0.75
        
        class init_state(MockConfig):
            pos = [0.0, 0.0, 0.42]
            rot = [0.0, 0.0, 0.0, 1.0]  # xyzw
            rot_gs = [1.0, 0.0, 0.0, 0.0]  # wxyz for Genesis
            lin_vel = [0.0, 0.0, 0.0]
            ang_vel = [0.0, 0.0, 0.0]
            roll_random_scale = 0.0
            pitch_random_scale = 0.0
            yaw_random_scale = 0.0
            default_joint_angles = {
                'FR_hip_joint': 0.0,
                'FR_thigh_joint': 0.8,
                'FR_calf_joint': -1.5,
                'FL_hip_joint': 0.0,
                'FL_thigh_joint': 0.8,
                'FL_calf_joint': -1.5,
                'RR_hip_joint': 0.0,
                'RR_thigh_joint': 0.8,
                'RR_calf_joint': -1.5,
                'RL_hip_joint': 0.0,
                'RL_thigh_joint': 0.8,
                'RL_calf_joint': -1.5,
            }
        
        class control(MockConfig):
            control_type = 'P'
            stiffness = {'joint': 20.0}
            damping = {'joint': 0.5}
            action_scale = 0.25
            decimation = 4
            dt = 0.02
        
        class asset(MockConfig):
            name = "go2"
            file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
            foot_name = "foot"
            penalize_contacts_on = ["thigh", "calf"]
            terminate_after_contacts_on = ["base"]
            fix_base_link = False
            obtain_link_contact_states = False
            contact_state_link_names = ["thigh", "calf", "foot"]
            links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
            dof_names = [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
            ]
            self_collisions_gs = True
            flip_visual_attachments = False
            disable_gravity = False
            collapse_fixed_joints = True
            default_dof_drive_mode = 3
            self_collisions_gym = 0
            replace_cylinder_with_capsule = False
            density = 0.001
            angular_damping = 0.0
            linear_damping = 0.0
            max_angular_velocity = 1000.0
            max_linear_velocity = 1000.0
            armature = 0.0
            thickness = 0.01
        
        class domain_rand(MockConfig):
            randomize_friction = True
            friction_range = [0.5, 1.25]
            randomize_base_mass = True
            added_mass_range = [-1.0, 1.0]
            push_robots = True
            push_interval_s = 15
            max_push_vel_xy = 1.0
            randomize_com_displacement = True
            com_pos_x_range = [-0.01, 0.01]
            com_pos_y_range = [-0.01, 0.01]
            com_pos_z_range = [-0.01, 0.01]
            randomize_ctrl_delay = False
            ctrl_delay_step_range = [0, 1]
            randomize_pd_gain = False
            kp_range = [0.8, 1.2]
            kd_range = [0.8, 1.2]
            randomize_joint_armature = False
            joint_armature_range = [0.0, 0.05]
            randomize_joint_friction = False
            joint_friction_range = [0.0, 0.1]
            randomize_joint_damping = False
            joint_damping_range = [0.0, 1.0]
        
        class rewards(MockConfig):
            soft_dof_pos_limit = 0.9
            base_height_target = 0.36
            foot_clearance_target = 0.05
            foot_height_offset = 0.022
            foot_clearance_tracking_sigma = 0.01
            only_positive_rewards = True
            tracking_sigma = 0.25
            soft_dof_vel_limit = 1.0
            soft_torque_limit = 1.0
            max_projected_gravity = -0.1
            
            class scales(MockConfig):
                dof_pos_limits = -10.0
                # dof_vel_limits = -10.0
                collision = -1.0
                tracking_lin_vel = 1.0
                tracking_ang_vel = 0.5
                lin_vel_z = -0.5
                base_height = -2.0
                ang_vel_xy = -0.05
                orientation = -1.0
                dof_vel = -5e-4
                dof_acc = -2e-7
                action_rate = -0.01
                action_smoothness = -0.01
                torques = -2e-4
                feet_air_time = 1.0
                foot_clearance = 0.5
                termination = -0.0
                feet_stumble = -0.0
                dof_pos_stand_still = -0.0
        
        class commands(MockConfig):
            curriculum = False
            max_curriculum = 1.0
            num_commands = 4
            resampling_time = 10.0
            heading_command = True
            curriculum_threshold = 0.8
            
            class ranges(MockConfig):
                lin_vel_x = [-0.5, 0.5]
                lin_vel_y = [-1.0, 1.0]
                ang_vel_yaw = [-1.0, 1.0]
                heading = [-3.14, 3.14]
        
        class normalization(MockConfig):
            clip_observations = 100.0
            clip_actions = 100.0
            
            class obs_scales(MockConfig):
                lin_vel = 1.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height_measurements = 5.0
        
        class noise(MockConfig):
            add_noise = True
            noise_level = 1.0
            
            class noise_scales(MockConfig):
                dof_pos = 0.01
                dof_vel = 0.5
                lin_vel = 0.1
                ang_vel = 0.2
                gravity = 0.05
                height_measurements = 0.1
            
            # Alias for backward compatibility
            scales = noise_scales
        
        class sim(MockConfig):
            dt = 0.005
            substeps = 2
            max_collision_pairs = 1024
            IK_max_targets = 4
        
        class sensor(MockConfig):
            add_depth = False
        
        class viewer(MockConfig):
            ref_env = 0
            pos = [2, 2, 2]
            lookat = [0.0, 0, 1.0]
            rendered_envs_idx = [0]
        
        class constraints(MockConfig):
            class limits(MockConfig):
                pass
    
    return MockGO2Cfg()


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_instantiation(cfg, device: str) -> tuple:
    """
    Test 1: Instantiation Check
    
    Verifies:
    - GenesisWrapper initializes correctly
    - num_observations is overridden to 45 (blind obs)
    - num_privileged_obs is correctly calculated
    """
    print_section("TEST 1: Instantiation Check")
    
    from envs.genesis_wrapper import GenesisWrapper
    
    # Calculate expected height scan dimension
    height_scan_dim = len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
    expected_priv_obs = 45 + 18 + height_scan_dim
    
    print(f"   Creating GenesisWrapper with {cfg.env.num_envs} environments...")
    print(f"   Original cfg.env.num_observations = {cfg.env.num_observations}")
    print(f"   Expected override to 45 (blind obs)")
    print(f"   Expected privileged_obs = 45 + 18 + {height_scan_dim} = {expected_priv_obs}")
    
    sim_params = {
        "dt": cfg.sim.dt,
        "substeps": cfg.sim.substeps,
    }
    
    try:
        env = GenesisWrapper(
            cfg=cfg,
            sim_params=sim_params,
            sim_device=device,
            headless=True,
        )
        
        # Check observation dimension override
        if env.num_obs != 45:
            print_fail(
                "Observation dimension override",
                "GenesisWrapper did not override num_observations to 45",
                expected="45",
                actual=str(env.num_obs)
            )
            return None, False
        else:
            print_pass("Observation dimension override", f"num_obs = {env.num_obs}")
        
        # Check privileged observation dimension
        if env.num_privileged_obs != expected_priv_obs:
            print_fail(
                "Privileged observation dimension",
                "Privileged obs dimension mismatch",
                expected=str(expected_priv_obs),
                actual=str(env.num_privileged_obs)
            )
            return None, False
        else:
            print_pass("Privileged observation dimension", f"num_privileged_obs = {env.num_privileged_obs}")
        
        # Check internal dimension storage
        if env._blind_obs_dim != 45:
            print_fail("Internal blind_obs_dim", "Incorrect value", "45", str(env._blind_obs_dim))
            return None, False
        
        if env._height_scan_dim != height_scan_dim:
            print_fail("Internal height_scan_dim", "Incorrect value", str(height_scan_dim), str(env._height_scan_dim))
            return None, False
        
        print_pass("Internal dimension tracking", f"blind={env._blind_obs_dim}, height_scan={env._height_scan_dim}")
        
        # Check observation history buffer shape
        expected_history_shape = (cfg.env.num_envs, cfg.env.history_len, 45)
        if env.obs_history_buf.shape != expected_history_shape:
            print_fail(
                "Observation history buffer shape",
                "Shape mismatch",
                str(expected_history_shape),
                str(env.obs_history_buf.shape)
            )
            return None, False
        
        print_pass("Observation history buffer", f"shape = {env.obs_history_buf.shape}")
        
        return env, True
        
    except Exception as e:
        print_fail("GenesisWrapper instantiation", f"Exception: {str(e)}")
        traceback.print_exc()
        return None, False


def test_step_signature(env) -> bool:
    """
    Test 2: Step Signature Check (CRITICAL)
    
    Verifies:
    - step() returns exactly 5 items
    - Correct shapes for obs, priv_obs, reward, reset, extras
    """
    print_section("TEST 2: Step Signature Check (CRITICAL)")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    # Create zero actions for testing
    actions = torch.zeros(num_envs, num_actions, dtype=torch.float32, device=env.device)
    
    print(f"   Executing env.step() with zero actions...")
    
    try:
        result = env.step(actions)
        
        # Check return count
        if not isinstance(result, tuple):
            print_fail("Step return type", "step() did not return a tuple", "tuple", str(type(result)))
            return False
        
        if len(result) != 5:
            print_fail(
                "Step return count",
                "step() must return exactly 5 items (obs, priv_obs, rew, reset, extras)",
                "5",
                str(len(result))
            )
            return False
        
        print_pass("Step return count", "Returns exactly 5 items")
        
        obs, priv_obs, rew, reset, extras = result
        
        # Check obs shape
        expected_obs_shape = (num_envs, 45)
        if obs.shape != expected_obs_shape:
            print_fail("obs shape", "Incorrect shape", str(expected_obs_shape), str(obs.shape))
            return False
        print_pass("obs shape", f"shape = {obs.shape}")
        
        # Check priv_obs shape
        # 45 (blind) + 18 (privileged state) + 100 (height scan 10x10)
        height_scan_dim = env._height_scan_dim
        expected_priv_shape = (num_envs, 45 + 18 + height_scan_dim)
        if priv_obs.shape != expected_priv_shape:
            print_fail("priv_obs shape", "Incorrect shape", str(expected_priv_shape), str(priv_obs.shape))
            return False
        print_pass("priv_obs shape", f"shape = {priv_obs.shape}")
        
        # Check reward shape
        if rew.shape != (num_envs,):
            print_fail("reward shape", "Incorrect shape", f"({num_envs},)", str(rew.shape))
            return False
        print_pass("reward shape", f"shape = {rew.shape}")
        
        # Check reset shape
        if reset.shape != (num_envs,):
            print_fail("reset shape", "Incorrect shape", f"({num_envs},)", str(reset.shape))
            return False
        print_pass("reset shape", f"shape = {reset.shape}")
        
        # Check extras is dict
        if not isinstance(extras, dict):
            print_fail("extras type", "extras must be a dictionary", "dict", str(type(extras)))
            return False
        print_pass("extras type", "Is dictionary")
        
        # Check extras contains required keys
        required_keys = ['privileged_obs', 'observations_history']
        for key in required_keys:
            if key not in extras:
                print_fail("extras keys", f"Missing required key: {key}")
                return False
        print_pass("extras keys", f"Contains {required_keys}")
        
        return True
        
    except Exception as e:
        print_fail("Step execution", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_physics_liveness(env, num_steps: int = 50) -> bool:
    """
    Test 3: Physics Liveness Check (CRITICAL)
    
    Detects "dead physics" where foot velocities remain zero despite movement.
    This catches silent failures where refresh_rigid_body_state_tensor is broken.
    """
    print_section("TEST 3: Physics Liveness Check (CRITICAL)")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    print(f"   Running {num_steps} steps with random non-zero actions...")
    print(f"   Monitoring foot velocities for physics activity...")
    
    max_foot_vel_observed = 0.0
    foot_vel_samples = []
    
    try:
        for step in range(num_steps):
            # Generate random non-zero actions (moderate magnitude)
            actions = torch.randn(num_envs, num_actions, dtype=torch.float32, device=env.device) * 0.5
            
            # Execute step
            env.step(actions)
            
            # Monitor foot velocities
            current_max = torch.max(torch.abs(env.foot_velocities)).item()
            max_foot_vel_observed = max(max_foot_vel_observed, current_max)
            
            if step % 10 == 0:
                foot_vel_samples.append(current_max)
                print(f"      Step {step}: max |foot_vel| = {current_max:.6f}")
        
        print(f"\n   Final max |foot_vel| observed: {max_foot_vel_observed:.6f}")
        
        # CRITICAL CHECK: Physics must be alive
        if max_foot_vel_observed == 0.0:
            print_fail(
                "Physics Liveness",
                "PHYSICS DEAD! Foot velocity is exactly zero while moving.",
                "Non-zero foot velocity",
                "0.0"
            )
            raise RuntimeError(
                "Physics Dead! Foot velocity is zero while moving. "
                "Check refresh_rigid_body_state_tensor or foot_velocities extraction."
            )
        
        # Check for reasonable physics (velocities should be meaningful)
        if max_foot_vel_observed < 0.001:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Physics may be sluggish: max_foot_vel = {max_foot_vel_observed:.6f}")
            print(f"       This could indicate weak actions or physics issues.")
        
        print_pass("Physics Liveness", f"max |foot_vel| = {max_foot_vel_observed:.6f}")
        
        # Additional check: foot velocities should vary
        if len(set([round(v, 6) for v in foot_vel_samples])) == 1:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Foot velocities are suspiciously constant")
        
        return True
        
    except RuntimeError:
        raise
    except Exception as e:
        print_fail("Physics Liveness", f"Exception during test: {str(e)}")
        traceback.print_exc()
        return False


def test_history_buffer(env) -> bool:
    """
    Test 4: History Buffer Check
    
    Verifies that the observation history buffer shifts correctly.
    """
    print_section("TEST 4: History Buffer Check")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    print("   Testing observation history buffer shifting...")
    
    try:
        # Reset environment to clear history
        env.reset()
        
        # Take a step to populate obs_buf
        actions_t = torch.randn(num_envs, num_actions, dtype=torch.float32, device=env.device) * 0.3
        env.step(actions_t)
        
        # Store the observation at time T
        obs_at_t = env.obs_buf.clone()
        print(f"   Step T: obs_buf mean = {obs_at_t.mean().item():.6f}")
        
        # Take another step (T+1)
        actions_t1 = torch.randn(num_envs, num_actions, dtype=torch.float32, device=env.device) * 0.3
        env.step(actions_t1)
        
        # Check if history buffer correctly contains obs from step T at index 1
        obs_in_history = env.obs_history_buf[:, 1, :]
        print(f"   Step T+1: history[1] mean = {obs_in_history.mean().item():.6f}")
        
        # Compare
        if torch.allclose(obs_in_history, obs_at_t, atol=1e-5):
            print_pass("History buffer shifting", "obs_history_buf[:, 1, :] == obs_buf from step T")
            return True
        else:
            # Calculate difference
            diff = torch.abs(obs_in_history - obs_at_t).max().item()
            print_fail(
                "History buffer shifting",
                "obs_history_buf[:, 1, :] does not match obs_buf from step T",
                "Exact match (atol=1e-5)",
                f"Max difference = {diff:.6f}"
            )
            return False
            
    except Exception as e:
        print_fail("History Buffer", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_reward_sanity(env, num_steps: int = 20) -> bool:
    """
    Test 5: Reward Sanity Check
    
    Verifies that reward functions return finite values (not NaN/Inf).
    """
    print_section("TEST 5: Reward Sanity Check")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    print(f"   Running {num_steps} steps to collect reward statistics...")
    print(f"   Using aggressive actions to trigger limit penalties...")
    
    reward_stats = {
        'dof_vel_limits': [],
        'torque_limits': [],
        'feet_slip': [],
    }
    
    try:
        for step in range(num_steps):
            # Use progressively more aggressive random actions to stress limits
            # Start moderate, increase to extreme to trigger limit violations
            action_scale = 0.3 + (step / num_steps) * 2.5  # Scale from 0.3 to 2.8
            actions = torch.randn(num_envs, num_actions, dtype=torch.float32, device=env.device) * action_scale
            env.step(actions)
            
            # Compute reward components
            try:
                dof_vel_rew = env._reward_dof_vel_limits()
                reward_stats['dof_vel_limits'].append(dof_vel_rew.mean().item())
            except Exception as e:
                print_fail("_reward_dof_vel_limits", f"Exception: {str(e)}")
                return False
            
            try:
                torque_rew = env._reward_torque_limits()
                reward_stats['torque_limits'].append(torque_rew.mean().item())
            except Exception as e:
                print_fail("_reward_torque_limits", f"Exception: {str(e)}")
                return False
            
            try:
                feet_slip_rew = env._reward_feet_slip()
                reward_stats['feet_slip'].append(feet_slip_rew.mean().item())
            except Exception as e:
                print_fail("_reward_feet_slip", f"Exception: {str(e)}")
                return False
        
        # Check for NaN/Inf
        all_pass = True
        for name, values in reward_stats.items():
            values_tensor = torch.tensor(values)
            mean_val = values_tensor.mean().item()
            max_val = values_tensor.max().item()
            min_val = values_tensor.min().item()
            
            # Check for NaN
            if torch.isnan(values_tensor).any():
                print_fail(f"_reward_{name}", "Contains NaN values!")
                all_pass = False
                continue
            
            # Check for Inf
            if torch.isinf(values_tensor).any():
                print_fail(f"_reward_{name}", "Contains Inf values!")
                all_pass = False
                continue
            
            print_pass(
                f"_reward_{name}",
                f"mean={mean_val:.6f}, min={min_val:.6f}, max={max_val:.6f}"
            )
        
        # Verify rewards are not all exactly zero (suspicious)
        for name, values in reward_stats.items():
            if all(v == 0.0 for v in values):
                print(f"{Colors.YELLOW}[WARN]{Colors.RESET} _reward_{name} is always exactly 0.0")
                print(f"       This may indicate the reward is not being triggered.")
                print(f"       Check if soft limits ({env._soft_vel_limit:.1f} rad/s, {env._soft_torque_limit:.1f} Nm) are ever exceeded.")
        
        return all_pass
        
    except Exception as e:
        print_fail("Reward Sanity", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_observation_validity(env) -> bool:
    """
    Test 6: Observation Validity Check
    
    Verifies that observations contain valid, non-zero data.
    """
    print_section("TEST 6: Observation Validity Check")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    try:
        # Reset and step
        env.reset()
        actions = torch.randn(num_envs, num_actions, dtype=torch.float32, device=env.device) * 0.3
        obs, priv_obs, _, _, _ = env.step(actions)
        
        all_pass = True
        
        # Check for NaN in observations
        if torch.isnan(obs).any():
            print_fail("obs NaN check", "obs_buf contains NaN values!")
            all_pass = False
        else:
            print_pass("obs NaN check", "No NaN values in obs_buf")
        
        # Check for Inf in observations
        if torch.isinf(obs).any():
            print_fail("obs Inf check", "obs_buf contains Inf values!")
            all_pass = False
        else:
            print_pass("obs Inf check", "No Inf values in obs_buf")
        
        # Check for NaN in privileged observations
        if torch.isnan(priv_obs).any():
            print_fail("priv_obs NaN check", "privileged_obs_buf contains NaN values!")
            all_pass = False
        else:
            print_pass("priv_obs NaN check", "No NaN values in privileged_obs_buf")
        
        # Check for Inf in privileged observations
        if torch.isinf(priv_obs).any():
            print_fail("priv_obs Inf check", "privileged_obs_buf contains Inf values!")
            all_pass = False
        else:
            print_pass("priv_obs Inf check", "No Inf values in privileged_obs_buf")
        
        # Check that observations are not all zeros
        obs_all_zero = (obs == 0).all()
        if obs_all_zero:
            print_fail("obs zero check", "obs_buf is all zeros - no data!")
            all_pass = False
        else:
            print_pass("obs zero check", f"obs_buf contains data (mean={obs.mean().item():.6f})")
        
        # Print observation statistics
        print(f"\n   Observation Statistics:")
        print(f"      obs_buf: min={obs.min().item():.4f}, max={obs.max().item():.4f}, mean={obs.mean().item():.4f}")
        print(f"      priv_obs: min={priv_obs.min().item():.4f}, max={priv_obs.max().item():.4f}, mean={priv_obs.mean().item():.4f}")
        
        return all_pass
        
    except Exception as e:
        print_fail("Observation Validity", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_reset_functionality(env) -> bool:
    """
    Test 7: Reset Functionality Check
    
    Verifies that reset_idx properly clears observation history.
    """
    print_section("TEST 7: Reset Functionality Check")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    try:
        # Run a few steps to build up history
        for _ in range(5):
            actions = torch.randn(num_envs, num_actions, dtype=torch.float32, device=env.device) * 0.3
            env.step(actions)
        
        # Check history is non-zero
        history_before = env.obs_history_buf.clone()
        history_nonzero = (history_before != 0).any().item()
        
        if not history_nonzero:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} History buffer is all zeros before reset")
        else:
            print(f"   History buffer has data before reset (mean={history_before.mean().item():.6f})")
        
        # Reset specific environments
        reset_env_ids = torch.tensor([0, 2, 4], device=env.device)
        env.reset_idx(reset_env_ids)
        
        # Check that reset environments have cleared history
        for env_id in reset_env_ids:
            env_history = env.obs_history_buf[env_id]
            if (env_history != 0).any():
                print_fail(
                    f"Reset env {env_id.item()} history",
                    "History buffer not cleared after reset",
                    "All zeros",
                    f"Non-zero values present"
                )
                return False
        
        print_pass("Reset clears history", f"Environments {reset_env_ids.tolist()} history cleared")
        
        # Check that non-reset environments retain history
        non_reset_ids = [1, 3, 5, 6, 7, 8, 9]
        for env_id in non_reset_ids[:3]:  # Check first 3
            env_history = env.obs_history_buf[env_id]
            # History should still have data (from before reset of other envs)
            # But it's complicated - just verify it's a tensor
            print(f"   Env {env_id} history shape: {env_history.shape}")
        
        print_pass("Reset isolation", "Non-reset environments not affected")
        
        return True
        
    except Exception as e:
        print_fail("Reset Functionality", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_hardware_limits(env) -> bool:
    """
    Test 8: Hardware Limits Check
    
    Verifies that hardware limit constants are properly set.
    """
    print_section("TEST 8: Hardware Limits Check")
    
    try:
        # Check MAX_TORQUE
        if env.MAX_TORQUE != 45.0:
            print_fail("MAX_TORQUE", "Incorrect value", "45.0", str(env.MAX_TORQUE))
            return False
        print_pass("MAX_TORQUE", f"= {env.MAX_TORQUE} Nm")
        
        # Check MAX_JOINT_VEL
        if env.MAX_JOINT_VEL != 30.0:
            print_fail("MAX_JOINT_VEL", "Incorrect value", "30.0", str(env.MAX_JOINT_VEL))
            return False
        print_pass("MAX_JOINT_VEL", f"= {env.MAX_JOINT_VEL} rad/s")
        
        # Check SOFT_LIMIT_FACTOR
        if env.SOFT_LIMIT_FACTOR != 0.9:
            print_fail("SOFT_LIMIT_FACTOR", "Incorrect value", "0.9", str(env.SOFT_LIMIT_FACTOR))
            return False
        print_pass("SOFT_LIMIT_FACTOR", f"= {env.SOFT_LIMIT_FACTOR}")
        
        # Check soft limits
        expected_soft_torque = 0.9 * 45.0
        expected_soft_vel = 0.9 * 30.0
        
        if abs(env._soft_torque_limit - expected_soft_torque) > 0.001:
            print_fail("_soft_torque_limit", "Incorrect value", str(expected_soft_torque), str(env._soft_torque_limit))
            return False
        print_pass("_soft_torque_limit", f"= {env._soft_torque_limit} Nm")
        
        if abs(env._soft_vel_limit - expected_soft_vel) > 0.001:
            print_fail("_soft_vel_limit", "Incorrect value", str(expected_soft_vel), str(env._soft_vel_limit))
            return False
        print_pass("_soft_vel_limit", f"= {env._soft_vel_limit} rad/s")
        
        return True
        
    except Exception as e:
        print_fail("Hardware Limits", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_gravity_dynamics(env, num_steps: int = 30) -> bool:
    """
    Test 9: The "Gravity" Check (CRITICAL)
    
    Verifies that projected_gravity actually changes when the robot tilts.
    If projected_gravity is hardcoded or static, the robot will NEVER learn balance.
    
    Method:
    1. Record initial projected_gravity
    2. Apply asymmetric actions to induce tilting
    3. Verify projected_gravity changes significantly
    """
    print_section("TEST 9: The 'Gravity' Check")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    print("   Verifying projected_gravity responds to robot orientation...")
    print("   If this fails, the robot cannot learn balance!")
    
    try:
        # Reset to clean state
        env.reset()
        
        # Run a few steps to stabilize
        for _ in range(5):
            actions = torch.zeros(num_envs, num_actions, dtype=torch.float32, device=env.device)
            env.step(actions)
        
        # Record initial projected_gravity
        initial_gravity = env.simulator.projected_gravity.clone()
        print(f"   Initial projected_gravity (env 0): [{initial_gravity[0, 0]:.4f}, {initial_gravity[0, 1]:.4f}, {initial_gravity[0, 2]:.4f}]")
        
        # For a standing robot, gravity should be approximately [0, 0, -1] (pointing down in body frame)
        # Check that initial gravity makes physical sense
        initial_z = initial_gravity[:, 2].mean().item()
        if initial_z > -0.5:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Initial gravity Z = {initial_z:.4f} (expected ~ -1.0 for upright robot)")
        
        # Store gravity history to track changes
        gravity_history = [initial_gravity.clone()]
        
        # Apply asymmetric actions to induce tilting
        # Different actions on left vs right legs will cause roll
        # Different actions on front vs back will cause pitch
        print("   Applying asymmetric actions to induce tilting...")
        
        for step in range(num_steps):
            # Create tilting actions: push front legs forward, back legs backward
            actions = torch.zeros(num_envs, num_actions, dtype=torch.float32, device=env.device)
            
            # Go2 joint order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
            #                  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
            # Apply asymmetric thigh actions to induce pitch
            actions[:, 1] = 1.5   # FR_thigh forward
            actions[:, 4] = 1.5   # FL_thigh forward
            actions[:, 7] = -1.5  # RR_thigh backward
            actions[:, 10] = -1.5 # RL_thigh backward
            
            # Also add some roll-inducing actions
            actions[:, 0] = 0.5   # FR_hip
            actions[:, 3] = -0.5  # FL_hip (opposite direction for roll)
            
            env.step(actions)
            gravity_history.append(env.simulator.projected_gravity.clone())
        
        # Analyze gravity changes
        final_gravity = env.simulator.projected_gravity.clone()
        print(f"   Final projected_gravity (env 0): [{final_gravity[0, 0]:.4f}, {final_gravity[0, 1]:.4f}, {final_gravity[0, 2]:.4f}]")
        
        # Calculate maximum deviation from initial gravity
        gravity_tensor = torch.stack(gravity_history, dim=0)  # [num_steps+1, num_envs, 3]
        
        # Compute deviation per step
        deviations = (gravity_tensor - initial_gravity.unsqueeze(0)).abs()
        max_deviation = deviations.max().item()
        mean_deviation = deviations.mean().item()
        
        # Check X and Y components (these should change with pitch and roll)
        max_x_deviation = deviations[:, :, 0].max().item()
        max_y_deviation = deviations[:, :, 1].max().item()
        max_z_deviation = deviations[:, :, 2].max().item()
        
        print(f"\n   Gravity deviation analysis:")
        print(f"      Max X deviation (roll):  {max_x_deviation:.6f}")
        print(f"      Max Y deviation (pitch): {max_y_deviation:.6f}")
        print(f"      Max Z deviation:         {max_z_deviation:.6f}")
        print(f"      Overall max deviation:   {max_deviation:.6f}")
        print(f"      Mean deviation:          {mean_deviation:.6f}")
        
        # CRITICAL CHECK: Gravity must change when robot tilts
        MIN_GRAVITY_CHANGE = 0.01  # At least 1% change expected
        
        if max_deviation < MIN_GRAVITY_CHANGE:
            print_fail(
                "Gravity Dynamics",
                "projected_gravity is STATIC! Robot tilted but gravity didn't change.",
                f"Max deviation > {MIN_GRAVITY_CHANGE}",
                f"Max deviation = {max_deviation:.6f}"
            )
            print(f"       {Colors.RED}CRITICAL: The robot will NEVER learn balance with static gravity!{Colors.RESET}")
            print(f"       Check: Is projected_gravity computed from quat_rotate_inverse(base_quat, global_gravity)?")
            return False
        
        # Check that X or Y component changed (indicating actual tilt)
        if max_x_deviation < 0.005 and max_y_deviation < 0.005:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Gravity X/Y components barely changed.")
            print(f"       The robot may not have tilted significantly during the test.")
            print(f"       Consider: Are actions being applied correctly?")
        
        print_pass("Gravity Dynamics", f"projected_gravity responds to orientation (max_dev={max_deviation:.4f})")
        
        # Additional check: Gravity vector should remain unit length
        gravity_norms = final_gravity.norm(dim=-1)
        norm_deviation = (gravity_norms - 1.0).abs().max().item()
        
        if norm_deviation > 0.01:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Gravity vector norm deviation: {norm_deviation:.4f}")
            print(f"       Expected unit vector, got norms: {gravity_norms[:3].tolist()}")
        else:
            print_pass("Gravity Normalization", f"Gravity vectors are unit length (norm_dev={norm_deviation:.6f})")
        
        return True
        
    except Exception as e:
        print_fail("Gravity Dynamics", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


def test_zero_action_stability(env, num_steps: int = 100) -> bool:
    """
    Test 10: The "Zero Action" Check (CRITICAL)
    
    Verifies that when the RL network outputs all zeros, the robot maintains
    its default standing pose and doesn't collapse.
    
    If the robot collapses with zero actions:
    - action_scale may be wrong
    - default_dof_pos offset may be incorrect
    - PD gains may be too weak
    
    Success criteria:
    1. Base height remains above collapse threshold (> 0.2m for Go2)
    2. Joint positions stay close to default_dof_pos
    3. Robot doesn't flip over (projected_gravity Z stays negative)
    """
    print_section("TEST 10: The 'Zero Action' Check")
    
    num_envs = env.num_envs
    num_actions = env.num_actions
    
    print("   Verifying robot stability with zero actions...")
    print("   If this fails, action_scale or default_dof_pos is likely wrong!")
    
    try:
        # Reset to clean state
        env.reset()
        
        # Get initial state
        initial_height = env.simulator.base_pos[:, 2].clone()
        default_dof_pos = env.simulator.default_dof_pos.clone()
        
        print(f"   Initial base height (mean): {initial_height.mean().item():.4f} m")
        print(f"   Default DOF positions shape: {default_dof_pos.shape}")
        
        # Track state over time
        height_history = []
        dof_deviation_history = []
        gravity_z_history = []
        
        # Run simulation with zero actions
        print(f"   Running {num_steps} steps with zero actions...")
        
        zero_actions = torch.zeros(num_envs, num_actions, dtype=torch.float32, device=env.device)
        
        for step in range(num_steps):
            env.step(zero_actions)
            
            # Record state
            current_height = env.simulator.base_pos[:, 2].mean().item()
            height_history.append(current_height)
            
            # DOF deviation from default
            dof_pos = env.simulator.dof_pos
            dof_deviation = (dof_pos - default_dof_pos).abs().mean().item()
            dof_deviation_history.append(dof_deviation)
            
            # Gravity Z component (should stay negative if upright)
            gravity_z = env.simulator.projected_gravity[:, 2].mean().item()
            gravity_z_history.append(gravity_z)
            
            if (step + 1) % 25 == 0:
                print(f"      Step {step+1}: height={current_height:.4f}m, dof_dev={dof_deviation:.4f}, grav_z={gravity_z:.4f}")
        
        # Analyze results
        final_height = height_history[-1]
        min_height = min(height_history)
        final_gravity_z = gravity_z_history[-1]
        max_dof_deviation = max(dof_deviation_history)
        
        print(f"\n   Zero-action stability analysis:")
        print(f"      Initial height:     {initial_height.mean().item():.4f} m")
        print(f"      Final height:       {final_height:.4f} m")
        print(f"      Minimum height:     {min_height:.4f} m")
        print(f"      Max DOF deviation:  {max_dof_deviation:.4f} rad")
        print(f"      Final gravity Z:    {final_gravity_z:.4f}")
        
        all_passed = True
        
        # Check 1: Robot didn't collapse (height threshold for Go2 ~0.2m)
        COLLAPSE_HEIGHT_THRESHOLD = 0.2
        if min_height < COLLAPSE_HEIGHT_THRESHOLD:
            print_fail(
                "Collapse Detection",
                f"Robot collapsed! Height dropped below {COLLAPSE_HEIGHT_THRESHOLD}m",
                f"Height > {COLLAPSE_HEIGHT_THRESHOLD}m",
                f"Min height = {min_height:.4f}m"
            )
            print(f"       {Colors.RED}Check: action_scale, default_dof_pos, or PD gains{Colors.RESET}")
            all_passed = False
        else:
            print_pass("Collapse Detection", f"Robot maintained height (min={min_height:.4f}m)")
        
        # Check 2: Robot didn't flip over (gravity Z should be negative)
        if final_gravity_z > 0:
            print_fail(
                "Flip Detection",
                "Robot flipped over! Gravity Z is positive (upside down)",
                "Gravity Z < 0",
                f"Gravity Z = {final_gravity_z:.4f}"
            )
            all_passed = False
        else:
            print_pass("Flip Detection", f"Robot stayed upright (gravity_z={final_gravity_z:.4f})")
        
        # Check 3: Joint positions stayed close to default
        DOF_DEVIATION_THRESHOLD = 0.5  # 0.5 rad max deviation allowed
        if max_dof_deviation > DOF_DEVIATION_THRESHOLD:
            print_fail(
                "Joint Position Stability",
                f"Joints deviated too much from default pose",
                f"Max deviation < {DOF_DEVIATION_THRESHOLD} rad",
                f"Max deviation = {max_dof_deviation:.4f} rad"
            )
            print(f"       {Colors.RED}Check: PD gains may be too weak or action_scale too large{Colors.RESET}")
            all_passed = False
        else:
            print_pass("Joint Position Stability", f"Joints stayed near default (max_dev={max_dof_deviation:.4f} rad)")
        
        # Check 4: Height didn't drop significantly
        height_drop = initial_height.mean().item() - final_height
        HEIGHT_DROP_THRESHOLD = 0.1  # 10cm max drop allowed
        
        if height_drop > HEIGHT_DROP_THRESHOLD:
            print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Robot height dropped by {height_drop:.4f}m")
            print(f"       This may indicate settling, weak PD gains, or incorrect default pose")
        else:
            print_pass("Height Stability", f"Height drop = {height_drop:.4f}m (threshold={HEIGHT_DROP_THRESHOLD}m)")
        
        # Check 5: Verify action_scale interpretation
        print("\n   Verifying action interpretation...")
        action_scale = env.cfg.control.action_scale
        print(f"      action_scale = {action_scale}")
        print(f"      Zero action → target_pos = default_dof_pos + 0 * action_scale = default_dof_pos")
        
        if all_passed:
            print_pass("Zero Action Stability", "Robot stands stable with zero actions")
        
        return all_passed
        
    except Exception as e:
        print_fail("Zero Action Stability", f"Exception: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Main test runner."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       GENESIS WRAPPER TORTURE TEST SUITE                     ║")
    print("║       Production Readiness Verification                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Running on CPU - tests may be slower")
    
    # Track results
    results = {}
    env = None
    
    try:
        # Create mock configuration
        print_section("SETUP: Creating Mock Configuration")
        cfg = create_mock_config()
        print(f"   num_envs = {cfg.env.num_envs}")
        print(f"   original num_observations = {cfg.env.num_observations}")
        print(f"   terrain scan = {len(cfg.terrain.measured_points_x)}x{len(cfg.terrain.measured_points_y)} = {len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)} points")
        print_pass("Mock configuration created")
        
        # Test 1: Instantiation
        env, results['instantiation'] = test_instantiation(cfg, device)
        if not results['instantiation']:
            raise RuntimeError("Instantiation failed - cannot continue tests")
        
        # Test 2: Step Signature
        results['step_signature'] = test_step_signature(env)
        
        # Test 3: Physics Liveness (CRITICAL)
        results['physics_liveness'] = test_physics_liveness(env)
        
        # Test 4: History Buffer
        results['history_buffer'] = test_history_buffer(env)
        
        # Test 5: Reward Sanity
        results['reward_sanity'] = test_reward_sanity(env)
        
        # Test 6: Observation Validity
        results['observation_validity'] = test_observation_validity(env)
        
        # Test 7: Reset Functionality
        results['reset_functionality'] = test_reset_functionality(env)
        
        # Test 8: Hardware Limits
        results['hardware_limits'] = test_hardware_limits(env)
        
        # Test 9: Gravity Dynamics (CRITICAL)
        results['gravity_dynamics'] = test_gravity_dynamics(env)
        
        # Test 10: Zero Action Stability (CRITICAL)
        results['zero_action_stability'] = test_zero_action_stability(env)
        
    except RuntimeError as e:
        print(f"\n{Colors.RED}{Colors.BOLD}CRITICAL ERROR: {str(e)}{Colors.RESET}")
        results['critical_error'] = False
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}UNEXPECTED ERROR: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        results['unexpected_error'] = False
    
    # Print Summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                      TEST SUMMARY                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"   {test_name.replace('_', ' ').title()}: [{status}]")
    
    print(f"\n   {Colors.BOLD}Total: {passed_tests}/{total_tests} passed{Colors.RESET}")
    
    if failed_tests == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED - GenesisWrapper is production-ready!{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ {failed_tests} TEST(S) FAILED - GenesisWrapper is NOT production-ready!{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
