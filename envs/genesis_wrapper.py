"""
Genesis environment wrapper for PIP-Loco training.
Provides 45-dim blind proprioceptive observations for actor and privileged observations for critic.
Enforces Unitree Go2 hardware limits via quadratic barrier rewards and maintains observation history for velocity estimation.
"""

import sys
import os
from typing import Dict, Tuple

import torch
from torch import Tensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis_lr.legged_gym.envs.go2.go2 import GO2
from genesis_lr.legged_gym.envs.go2.go2_config import GO2Cfg


class GenesisWrapper(GO2):

    MAX_TORQUE: float = 45.0
    MAX_JOINT_VEL: float = 30.0
    SOFT_LIMIT_FACTOR: float = 0.9

    def __init__(
        self,
        cfg: GO2Cfg,
        sim_params: dict,
        sim_device: str,
        headless: bool,
    ) -> None:
        cfg.env.num_observations = 53
        
        height_scan_dim = len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
        
        cfg.env.num_privileged_obs = 53 + 18 + height_scan_dim
        
        self._blind_obs_dim = 53
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
        # We must initialize this manually because the parent class does not do it
        self.foot_velocities = torch.zeros(
            self.num_envs,
            len(self.simulator.feet_indices),
            3,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        
        # Soft limits for barrier functions
        self._soft_torque_limit = self.SOFT_LIMIT_FACTOR * self.MAX_TORQUE
        self._soft_vel_limit = self.SOFT_LIMIT_FACTOR * self.MAX_JOINT_VEL

        # Initialize Phase Clock for 4-beat gait
        self.gait_frequency = 1.0 # Hz
        self.gait_phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False
        )
        
        # Phase offsets for Unitree joint order: [FR, FL, RR, RL]
        # Target sequence: FL(0.0) -> RR(0.25) -> FR(0.5) -> RL(0.75)
        self.phase_offsets = torch.tensor(
            [0.5, 0.0, 0.75, 0.25], dtype=torch.float32, device=self.device, requires_grad=False
        )

        # Initialize motor strength domain randomization buffer
        self.motor_strength = torch.ones(
            self.num_envs, self.cfg.env.num_actions, 
            dtype=torch.float32, device=self.device, requires_grad=False
        )


    def _get_noise_scale_vec(self) -> Tensor:
        # Construct noise scale vector for 53-dim blind observations
        noise_vec = torch.zeros(self._blind_obs_dim, dtype=torch.float32, device=self.device)
        
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales 
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.0
        noise_vec[45:53] = 0.0
        
        return noise_vec

    def compute_observations(self) -> None:
        self.foot_velocities = self.simulator.feet_vel.clone()
        
        # Construct blind observations (53 dims)
        leg_phases = (self.gait_phase + self.phase_offsets) % 1.0

        sin_phases = torch.sin(2 * torch.pi * leg_phases)
        cos_phases = torch.cos(2 * torch.pi * leg_phases)

    
        self.obs_buf = torch.cat([
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phases,
            cos_phases,
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
        
        # Construct privileged observations
        if self.num_privileged_obs is not None:
            if self.cfg.terrain.measure_heights:
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
            
            self.privileged_obs_buf = torch.cat([
                self.obs_buf,
                privileged_state,
                heights,
            ], dim=-1)
            
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict]:
        clip_actions = self.cfg.normalization.clip_actions

        scaled_actions = actions * self.motor_strength
        self.actions = torch.clip(scaled_actions, -clip_actions, clip_actions).to(self.device)
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        obs, privileged_obs, rew, reset, extras = super().step(self.actions)

        # Check if the environment is receiving a movement command
        command_norm = torch.norm(self.commands[:, :3], dim=1, keepdim=True)
        is_moving = (command_norm > 0.1).float()
        
        # Only advance clock for robots that are moving actively
        dt_control = self.cfg.control.dt * self.cfg.control.decimation
        self.gait_phase = (self.gait_phase + dt_control * self.gait_frequency * is_moving) % 1.0
        
        extras['privileged_obs'] = self.privileged_obs_buf
        extras['observations_history'] = self.obs_history_buf
        return obs, privileged_obs, rew, reset, extras
    
    def reset_idx(self, env_ids: Tensor) -> None:
        super().reset_idx(env_ids)
        
        if len(env_ids) > 0:
            # Reset custom buffers
            self.obs_history_buf[env_ids] = 0.0
            self.gait_phase[env_ids] = 0.0
            
            # Motor strength 
            if getattr(self.cfg.domain_rand, 'randomize_motor_strength', False):
                min_str, max_str = self.cfg.domain_rand.motor_strength_range
                self.motor_strength[env_ids] = torch.rand(
                    len(env_ids), self.cfg.env.num_actions, 
                    dtype=torch.float32, device=self.device
                ) * (max_str - min_str) + min_str

            # Link mass
            if getattr(self.cfg.domain_rand, 'randomize_link_mass', False):
                min_mass, max_mass = self.cfg.domain_rand.added_link_mass_range
                # Generate a single mass multiplier per resetting environment
                mass_multiplier = torch.rand(len(env_ids), 1, dtype=torch.float32, device=self.device)
                mass_multiplier = mass_multiplier * (max_mass - min_mass) + min_mass
                
                # Apply the multiplier to the non-base links (thighs, calves)
                if hasattr(self.simulator, 'link_names') and hasattr(self.simulator, 'default_link_masses'):
                    link_indices = [i for i, name in enumerate(self.simulator.link_names) if "base" not in name]
                    if link_indices:
                        new_masses = self.simulator.default_link_masses[env_ids][:, link_indices] * mass_multiplier
                        self.simulator.set_link_masses(env_ids, link_indices, new_masses)

    def _reward_dof_vel_limits(self) -> Tensor:
        # Quadratic barrier reward for joint velocity limits
        vel_magnitude = torch.abs(self.simulator.dof_vel)
        violation = torch.clamp(vel_magnitude - self._soft_vel_limit, min=0.0)
        return torch.sum(torch.square(violation), dim=-1)

    def _reward_torque_limits(self) -> Tensor:
        torque_magnitude = torch.abs(self.simulator.torques)
        violation = torch.clamp(torque_magnitude - self._soft_torque_limit, min=0.0)
        return torch.sum(torch.square(violation), dim=-1)

    def _reward_feet_slip(self) -> Tensor:

        # Detect foot contact using vertical contact force threshold
        forces = self.simulator.link_contact_forces
        contact_mask = (forces[:, self.simulator.feet_indices, 2] > 1.0).float()
        
        foot_xy_vel = self.foot_velocities[:, :, :2]
        foot_xy_speed_sq = torch.sum(torch.square(foot_xy_vel), dim=-1)
        
        # Penalize slip only when in contact
        slip_penalty = torch.sum(foot_xy_speed_sq * contact_mask, dim=-1)
        return slip_penalty

    def _reward_gait_timing(self) -> Tensor:
        # Enforce 4-beat sequence
        leg_phases = (self.gait_phase + self.phase_offsets) % 1.0
        target_contact = (leg_phases >= 0.25).float()
        forces = self.simulator.link_contact_forces
        actual_contact = (forces[:, self.simulator.feet_indices, 2] > 1.0).float()
        
        alpha = 5.0
        error = torch.abs(actual_contact - target_contact)
        reward = torch.exp(-alpha * torch.mean(error, dim=-1)) 

        command_norm = torch.norm(self.commands[:, :3], dim=1)
        moving_mask = (command_norm > 0.1).float()
        return reward * moving_mask 

    def _reward_contact_number(self) -> Tensor:
        # Encourage more feet in contact
        forces = self.simulator.link_contact_forces
        contact_mask = (forces[:, self.simulator.feet_indices, 2] > 1.0).float()
        num_contacts = torch.sum(contact_mask, dim=-1)
        violation = torch.abs(num_contacts - 3.0)
        reward = torch.exp(-2.0 * violation)

        command_norm = torch.norm(self.commands[:, :3], dim=1)
        moving_mask = (command_norm > 0.1).float()

        return reward * moving_mask

    def _reward_stand_still(self) -> Tensor:
        # Penalize joint velocities only when commanded to stop
        command_norm = torch.norm(self.commands[:, :3], dim=1)
        idle_mask = (command_norm < 0.1).float()
        
        return torch.sum(torch.abs(self.simulator.dof_vel), dim=-1) * idle_mask



