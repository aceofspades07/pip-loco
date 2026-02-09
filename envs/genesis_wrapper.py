"""
GenesisWrapper: Bridge between Genesis simulation and PIP-Loco hybrid-PPO training.

This wrapper inherits from GO2 and provides:
1. 45-dimensional "blind" proprioceptive observations for the actor
2. Privileged observations with ground truth physics for the critic
3. Quadratic-barrier safety constraints for Unitree Go2 hardware limits
4. Observation history buffer for velocity estimation

Author: PIP-Loco Team
"""

import sys
import os
from typing import Dict, Tuple

import torch
from torch import Tensor

# Add parent directory to path to allow importing from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis_lr.legged_gym.envs.go2.go2 import GO2
from genesis_lr.legged_gym.envs.go2.go2_config import GO2Cfg


class GenesisWrapper(GO2):
    """
    Genesis environment wrapper for PIP-Loco blind locomotion training.
    
    Provides asymmetric actor-critic observations:
    - Actor (obs_buf): 45-dim blind proprioception (no velocity/terrain/friction)
    - Critic (privileged_obs_buf): Full physics state + terrain heights
    
    Hardware limits enforced via Quadratic Barrier rewards:
    - Max Torque: 45.0 Nm (Unitree Go2)
    - Max Joint Velocity: 30.0 rad/s (Unitree Go2)
    """

    # Unitree Go2 Hardware Limits
    MAX_TORQUE: float = 45.0  # Nm
    MAX_JOINT_VEL: float = 30.0  # rad/s
    SOFT_LIMIT_FACTOR: float = 0.9  # Apply penalties at 90% of limit

    def __init__(
        self,
        cfg: GO2Cfg,
        sim_params: dict,
        sim_device: str,
        headless: bool,
    ) -> None:
        """
        Initialize GenesisWrapper with PIP-Loco observation structure.
        """
        # Override observation dimensions before calling super().__init__
        cfg.env.num_observations = 45  # Blind obs
        
        # Calculate height scan dimension for privileged observations
        # Note: measured_points is a list of coordinates
        height_scan_dim = len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
        
        # Privileged obs: blind_obs(45) + privileged_state(18) + terrain_scan(height_scan_dim)
        cfg.env.num_privileged_obs = 45 + 18 + height_scan_dim
        
        # Store dimensions for later use
        self._blind_obs_dim = 45
        self._privileged_state_dim = 18
        self._height_scan_dim = height_scan_dim
        
        # Retrieve history length from config (default to 50 if not specified)
        self._history_len = getattr(cfg.env, 'history_len', 50)
        
        # Initialize parent class
        super().__init__(cfg, sim_params, sim_device, headless)
        
        # Initialize observation history buffer: [num_envs, history_len, obs_dim]
        self.obs_history_buf = torch.zeros(
            self.num_envs,
            self._history_len,
            self._blind_obs_dim,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        
        # Initialize foot velocities buffer
        # We must initialize this manually because we cannot rely on the parent to update it
        # Note: feet_indices is on the simulator object
        self.foot_velocities = torch.zeros(
            self.num_envs,
            len(self.simulator.feet_indices),
            3,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        
        # Precompute soft limits for barrier functions
        self._soft_torque_limit = self.SOFT_LIMIT_FACTOR * self.MAX_TORQUE
        self._soft_vel_limit = self.SOFT_LIMIT_FACTOR * self.MAX_JOINT_VEL

    def _get_noise_scale_vec(self) -> Tensor:
        """
        Construct noise scale vector for 45-dimensional blind observations.
        """
        noise_vec = torch.zeros(self._blind_obs_dim, dtype=torch.float32, device=self.device)
        
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales 
        noise_level = self.cfg.noise.noise_level
        
        # Angular velocity noise (indices 0-3)
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        
        # Projected gravity noise (indices 3-6)
        noise_vec[3:6] = noise_scales.gravity * noise_level
        
        # Commands - no noise (indices 6-9)
        noise_vec[6:9] = 0.0
        
        # Joint position noise (indices 9-21)
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        
        # Joint velocity noise (indices 21-33)
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        
        # Last actions - no noise (indices 33-45)
        noise_vec[33:45] = 0.0
        
        return noise_vec

    def compute_observations(self) -> None:
        """
        Compute blind and privileged observations for asymmetric actor-critic.
        """
        # 1. MANUALLY REFRESH PHYSICS STATE
        # Genesis updates physics state in simulator.post_physics_step() which is called by parent step()
        # For compute_observations called within post_physics_step, the state is already updated
        
        # 2. MANUALLY EXTRACT FOOT VELOCITIES from simulator
        # Genesis provides feet_vel directly from the simulator
        self.foot_velocities = self.simulator.feet_vel.clone()
        
        # 3. Construct blind observations (45 dims)
        self.obs_buf = torch.cat([
            # Angular velocity (3 dims)
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            # Projected gravity (3 dims)
            self.simulator.projected_gravity,
            # Commands (3 dims)
            self.commands[:, :3] * self.commands_scale,
            # Joint positions relative to default (12 dims)
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            # Joint velocities (12 dims)
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            # Last actions (12 dims)
            self.actions,
        ], dim=-1)
        
        # Apply observation noise
        if self.add_noise:
            noise = (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec
            self.obs_buf += noise
        
        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        
        # Update observation history buffer
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=1, dims=1)
        self.obs_history_buf[:, 0, :] = self.obs_buf.clone()
        
        # 4. Construct privileged observations
        if self.num_privileged_obs is not None:
            # Get measured terrain heights
            if self.cfg.terrain.measure_heights:
                # Use measured_heights from simulator
                heights = torch.clip(
                    self.simulator.base_pos[:, 2:3] - 0.5 - self.simulator.measured_heights,
                    -1.0, 1.0
                ) * self.obs_scales.height_measurements
            else:
                heights = torch.zeros(
                    self.num_envs, self._height_scan_dim,
                    dtype=torch.float32, device=self.device
                )
            
            # Extract privileged physics state (18 dims)
            privileged_state = torch.cat([
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                self.simulator.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.simulator._friction_values.view(-1, 1),
                self.simulator._added_base_mass.view(-1, 1),
                self.simulator._base_com_bias.view(-1, 3),
                self.simulator._rand_push_vels[:, 0:1], 
            ], dim=-1)
            
            # Concatenate: [blind_obs, privileged_state, terrain_heights]
            self.privileged_obs_buf = torch.cat([
                self.obs_buf,
                privileged_state,
                heights,
            ], dim=-1)
            
            # Clip privileged observations
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict]:
        """
        Execute one environment step with action clipping and extras injection.
        """
        # Clip actions to configured limits
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # Execute parent step 
        # Captures 5 values: obs, privileged_obs, rew, reset, extras
        obs, privileged_obs, rew, reset, extras = super().step(self.actions)
        
        # Inject PIP-Loco specific data into extras
        extras['privileged_obs'] = self.privileged_obs_buf
        extras['observations_history'] = self.obs_history_buf
        # Return exactly 5 values to match parent signature
        return obs, privileged_obs, rew, reset, extras
    
    def reset_idx(self, env_ids: Tensor) -> None:
        """
        Reset specified environments and clear their observation history.
        """
        super().reset_idx(env_ids)
        
        # Clear observation history for reset environments
        if len(env_ids) > 0:
            self.obs_history_buf[env_ids] = 0.0

    def _reward_dof_vel_limits(self) -> Tensor:
        """
        Quadratic Barrier penalty for joint velocity limits.
        Ref: Nilaksh et al., ICRA 2024
        """
        # Calculate violation: max(0, |vel| - soft_limit)
        vel_magnitude = torch.abs(self.simulator.dof_vel)
        violation = torch.clamp(vel_magnitude - self._soft_vel_limit, min=0.0)
        
        # Return squared violation summed over joints
        return torch.sum(torch.square(violation), dim=-1)

    def _reward_torque_limits(self) -> Tensor:
        """
        Quadratic Barrier penalty for torque limits.
        """
        # Calculate violation: max(0, |torque| - soft_limit)
        torque_magnitude = torch.abs(self.simulator.torques)
        violation = torch.clamp(torque_magnitude - self._soft_torque_limit, min=0.0)
        
        # Return squared violation summed over joints
        return torch.sum(torch.square(violation), dim=-1)

    def _reward_feet_slip(self) -> Tensor:
        """
        Penalize foot slipping while in contact with ground.
        """
        # Detect foot contact using vertical contact force threshold
        # Using link_contact_forces from simulator (Genesis API)
        forces = self.simulator.link_contact_forces
        
        contact_mask = (forces[:, self.simulator.feet_indices, 2] > 1.0).float()
        
        # Calculate squared XY speed of each foot
        # Using foot_velocities (from simulator.feet_vel)
        foot_xy_vel = self.foot_velocities[:, :, :2] 
        foot_xy_speed_sq = torch.sum(torch.square(foot_xy_vel), dim=-1)
        
        # Penalize slip only when in contact
        slip_penalty = torch.sum(foot_xy_speed_sq * contact_mask, dim=-1)
        
        return slip_penalty