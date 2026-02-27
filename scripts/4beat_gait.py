"""
Deployment script for 4 beat gait with PIP-Loco framework.
Logs physical states for post-run analysis and visualization.
"""

import os
from pathlib import Path
import sys
import time
import torch
import numpy as np
import pygame
import genesis as gs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.pip_config import PIPGO2Cfg, PIPTrainCfg
from envs.genesis_wrapper import GenesisWrapper
from modules.velocity_estimator import VelocityEstimator
from modules.dreamer import NoLatentModel
from modules.pip_actor_critic import ActorCritic

MODEL_PATH = str(Path(__file__).parent.parent / 'logs' / 'pip_go2_20260224_230452' / 'model_final.pt')

VEL_X_MAX = 0.8
VEL_X_MIN = 0.0  
VEL_Y_MAX = 0.1
YAW_RATE_MAX = 0.2
VEL_INCREMENT = 0.05
VEL_DECAY = 0.90

class KeyboardController:
    def __init__(self, device: str = "cuda:0"):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 250))
        pygame.display.set_caption("PIP-Loco 4-Beat Controller")
        self.font = pygame.font.Font(None, 32)
        
        self.vx, self.vy, self.yaw = 0.0, 0.0, 0.0
        self.device = device
        self.running = True
        self.reset_requested = False

    def update(self):
        self.reset_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self.running = False
                if event.key == pygame.K_r: self.reset_requested = True

        keys = pygame.key.get_pressed()
        # X-Velocity (W/S)
        if keys[pygame.K_w]:
            self.vx = np.clip(self.vx + VEL_INCREMENT, 0.0, VEL_X_MAX)
        elif keys[pygame.K_s]:
            self.vx = np.clip(self.vx - VEL_INCREMENT, -VEL_X_MAX, 0.0)
        else:
            self.vx *= VEL_DECAY

        # Y-Velocity and Yaw
        if keys[pygame.K_a]: self.vy = min(self.vy + VEL_INCREMENT, VEL_Y_MAX)
        elif keys[pygame.K_d]: self.vy = max(self.vy - VEL_INCREMENT, -VEL_Y_MAX)
        else: self.vy *= VEL_DECAY

        if keys[pygame.K_LEFT]: self.yaw = min(self.yaw + VEL_INCREMENT, YAW_RATE_MAX)
        elif keys[pygame.K_RIGHT]: self.yaw = max(self.yaw - VEL_INCREMENT, -YAW_RATE_MAX)
        else: self.yaw *= VEL_DECAY

        self._render()
        return self.running

    def _render(self):
        self.screen.fill((30, 30, 30))
        text = [
            f"Commanded Vx: {self.vx:.2f}",
            f"Commanded Vy: {self.vy:.2f}",
            f"Commanded Yaw: {self.yaw:.2f}",
            "",
            "W/S: Fwd/Back (Target 0.2-0.8)",
            "Arrows: Turn | R: Reset | ESC: Quit"
        ]
        for i, line in enumerate(text):
            img = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(img, (20, 20 + i*30))
        pygame.display.flip()

def play():
    # 1. Load Configs
    env_cfg = PIPGO2Cfg()
    train_cfg = PIPTrainCfg()
    device = "cuda:0"
    
    # Force single env
    env_cfg.env.num_envs = 1 
    env_cfg.noise.add_noise = False
    
    # Fix the visualizer crash by restricting render index to 0
    if not hasattr(env_cfg, 'viewer'):
        env_cfg.viewer = type('viewer', (), {})()
    env_cfg.viewer.rendered_envs_idx = [0]
    
    # Initialize Env
    gs.init(logging_level="warning")
    env = GenesisWrapper(cfg=env_cfg, sim_params={"dt": env_cfg.sim.dt, "substeps": 1}, sim_device=device, headless=False)
    
    # Load Policy
    estimator = VelocityEstimator(input_dim=53, history_length=50)
    dreamer = NoLatentModel(obs_dim=53, action_dim=12)
    actor_critic = ActorCritic(
        num_actor_obs=53, 
        num_critic_obs=53 + 18 + (17*11), 
        num_actions=12,
        estimator=estimator, dreamer=dreamer
    ).to(device)
    
    print(f"Loading checkpoint: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()

    controller = KeyboardController(device=device)
    obs, _ = env.reset()
    
    # Data logging buffers
    logged_data = {
        'contact_forces': [],      # [T, 4] vertical forces per foot
        'base_lin_vel': [],        # [T, 3] XYZ velocity
        'commanded_vel': [],       # [T, 3] X, Y, Yaw commands
        'joint_torques': [],       # [T, 12] torques
        'joint_velocities': [],    # [T, 12] dof velocities
        'projected_gravity': [],   # [T, 3] for roll/pitch
    }
    
    control_dt = env_cfg.sim.dt * env_cfg.control.decimation

    print("Running... Use Keyboard window to control robot.")
    
    try:
        with torch.no_grad():
            while controller.update():
                start_time = time.perf_counter()
                
                if controller.reset_requested:
                    obs, _ = env.reset()
                    print("Environment Reset.")

                # Set commands from keyboard
                env.commands[0, 0] = controller.vx
                env.commands[0, 1] = controller.vy
                env.commands[0, 2] = controller.yaw

                # Step simulation
                actions = actor_critic.act_inference(obs, env.obs_history_buf)
                obs, _, _, _, extras = env.step(actions)

                # Log state data
                forces = env.simulator.link_contact_forces[:, env.simulator.feet_indices, 2]
                logged_data['contact_forces'].append(forces[0].cpu().numpy())
                logged_data['base_lin_vel'].append(env.simulator.base_lin_vel[0].cpu().numpy())
                logged_data['commanded_vel'].append(env.commands[0, :3].cpu().numpy())
                logged_data['joint_torques'].append(env.simulator.torques[0].cpu().numpy())
                logged_data['joint_velocities'].append(env.simulator.dof_vel[0].cpu().numpy())
                logged_data['projected_gravity'].append(env.simulator.projected_gravity[0].cpu().numpy())

                elapsed = time.perf_counter() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stack and save all logged data
        for key in logged_data:
            logged_data[key] = np.stack(logged_data[key], axis=0)
        np.save("gait_data.npy", logged_data)
        print("Gait data saved to gait_data.npy")
        pygame.quit()

if __name__ == "__main__":
    play()