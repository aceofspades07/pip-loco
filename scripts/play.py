"""
PIP-Loco Play Script: Real-time inference with keyboard control.

Run a trained PIP-Loco model with interactive velocity commands via PyGame.
Controls:
    W/S: Forward/Backward velocity (x)
    A/D: Left/Right velocity (y)
    Left/Right Arrows: Yaw rate (turning)
    R: Reset environment
    ESC/Q: Quit

Author: PIP-Loco Team
"""

import os
import sys
import time
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of launch directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PyGame for keyboard input
import pygame

# Genesis simulator
import genesis as gs

# Project imports
from config.pip_config import PIPGO2Cfg, PIPTrainCfg
from envs.genesis_wrapper import GenesisWrapper
from modules.velocity_estimator import VelocityEstimator
from modules.dreamer import NoLatentModel
from modules.pip_actor_critic import ActorCritic


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model checkpoint path (update this to your trained model)
MODEL_PATH = "/home/karan/pip_loco/logs/pip_go2_20260211_223927/model_final.pt"

# Velocity command limits
VEL_X_MAX = 1.0      # m/s forward/backward
VEL_Y_MAX = 0.5      # m/s left/right  
YAW_RATE_MAX = 1.0   # rad/s turning

# Velocity increment per key press (for smooth ramping)
VEL_X_INCREMENT = 0.1
VEL_Y_INCREMENT = 0.1
YAW_INCREMENT = 0.15

# Velocity decay factor when no key pressed (smooth stopping)
VEL_DECAY = 0.95


# =============================================================================
# KEYBOARD CONTROLLER
# =============================================================================

class KeyboardController:
    """
    PyGame-based keyboard controller for velocity commands.
    
    Provides smooth velocity ramping and decay for natural control feel.
    """
    
    def __init__(self, device: str = "cuda:0"):
        """Initialize PyGame and velocity state."""
        pygame.init()
        
        # Create a small window to capture keyboard events
        # PyGame needs a display surface to receive events
        self.screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("PIP-Loco Controller - Press ESC to quit")
        
        # Font for displaying commands
        self.font = pygame.font.Font(None, 36)
        
        # Current velocity commands
        self.vel_x = 0.0    # Forward/backward
        self.vel_y = 0.0    # Left/right
        self.yaw_rate = 0.0 # Turning
        
        self.device = device
        self.running = True
        self.reset_requested = False
        
    def update(self) -> bool:
        """
        Process keyboard events and update velocity commands.
        
        Returns:
            bool: False if quit requested, True otherwise
        """
        self.reset_requested = False
        
        # Process all pending events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                    return False
                elif event.key == pygame.K_r:
                    self.reset_requested = True
        
        # Get currently pressed keys for continuous control
        keys = pygame.key.get_pressed()
        
        # Track if any movement key is pressed
        x_pressed = False
        y_pressed = False
        yaw_pressed = False
        
        # Forward/Backward (W/S)
        if keys[pygame.K_w]:
            self.vel_x = min(self.vel_x + VEL_X_INCREMENT, VEL_X_MAX)
            x_pressed = True
        if keys[pygame.K_s]:
            self.vel_x = max(self.vel_x - VEL_X_INCREMENT, -VEL_X_MAX)
            x_pressed = True
            
        # Left/Right strafe (A/D)
        if keys[pygame.K_a]:
            self.vel_y = min(self.vel_y + VEL_Y_INCREMENT, VEL_Y_MAX)
            y_pressed = True
        if keys[pygame.K_d]:
            self.vel_y = max(self.vel_y - VEL_Y_INCREMENT, -VEL_Y_MAX)
            y_pressed = True
            
        # Turning (Arrow keys)
        if keys[pygame.K_LEFT]:
            self.yaw_rate = min(self.yaw_rate + YAW_INCREMENT, YAW_RATE_MAX)
            yaw_pressed = True
        if keys[pygame.K_RIGHT]:
            self.yaw_rate = max(self.yaw_rate - YAW_INCREMENT, -YAW_RATE_MAX)
            yaw_pressed = True
        
        # Apply decay when keys not pressed (smooth stopping)
        if not x_pressed:
            self.vel_x *= VEL_DECAY
            if abs(self.vel_x) < 0.01:
                self.vel_x = 0.0
        if not y_pressed:
            self.vel_y *= VEL_DECAY
            if abs(self.vel_y) < 0.01:
                self.vel_y = 0.0
        if not yaw_pressed:
            self.yaw_rate *= VEL_DECAY
            if abs(self.yaw_rate) < 0.01:
                self.yaw_rate = 0.0
        
        # Update display
        self._render()
        
        return True
    
    def _render(self):
        """Render current velocity commands to PyGame window."""
        self.screen.fill((30, 30, 30))  # Dark background
        
        # Render velocity text
        lines = [
            f"Vx: {self.vel_x:+.2f} m/s",
            f"Vy: {self.vel_y:+.2f} m/s", 
            f"Yaw: {self.yaw_rate:+.2f} rad/s",
            "",
            "W/S: Fwd/Back | A/D: Strafe",
            "Arrows: Turn | R: Reset | ESC: Quit"
        ]
        
        y_offset = 20
        for line in lines:
            text = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(text, (20, y_offset))
            y_offset += 30
            
        pygame.display.flip()
    
    def get_commands_tensor(self) -> torch.Tensor:
        """
        Get velocity commands as a CUDA tensor.
        
        Returns:
            torch.Tensor: Shape (1, 3) with [vel_x, vel_y, yaw_rate]
        """
        return torch.tensor(
            [[self.vel_x, self.vel_y, self.yaw_rate]],
            dtype=torch.float32,
            device=self.device
        )
    
    def close(self):
        """Clean up PyGame resources."""
        pygame.quit()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: str, env_cfg: PIPGO2Cfg, train_cfg: PIPTrainCfg, device: str):
    """
    Load trained PIP-Loco model with all sub-modules.
    
    Args:
        model_path: Path to saved checkpoint (.pt file)
        env_cfg: Environment configuration
        train_cfg: Training configuration  
        device: CUDA device string
        
    Returns:
        ActorCritic: Loaded model in eval mode
    """
    print(f"[PIP-Loco] Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct sub-modules
    # 1. Velocity Estimator (TCN)
    estimator = VelocityEstimator(
        input_dim=train_cfg.estimator.input_dim,
        history_length=train_cfg.estimator.history_length,
        hidden_dims=train_cfg.estimator.hidden_dims,
        output_dim=train_cfg.estimator.output_dim,
    )
    
    # 2. Dreamer (NoLatentModel world model)
    activation_map = {'elu': torch.nn.ELU, 'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh}
    dreamer_activation = activation_map.get(train_cfg.dreamer.activation, torch.nn.ELU)
    
    dreamer = NoLatentModel(
        obs_dim=train_cfg.dreamer.obs_dim,
        action_dim=train_cfg.dreamer.action_dim,
        hidden_dims=train_cfg.dreamer.hidden_dims,
        activation=dreamer_activation,
    )
    
    # Calculate privileged obs dimension (same as in wrapper)
    height_scan_dim = len(env_cfg.terrain.measured_points_x) * len(env_cfg.terrain.measured_points_y)
    num_privileged_obs = env_cfg.env.num_observations + 18 + height_scan_dim
    
    # 3. ActorCritic (main policy)
    actor_activation = activation_map.get(train_cfg.policy.activation, torch.nn.ELU)
    
    actor_critic = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=num_privileged_obs,
        num_actions=env_cfg.env.num_actions,
        estimator=estimator,
        dreamer=dreamer,
        horizon=train_cfg.policy.dreamer_horizon,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=actor_activation,
        init_noise_std=train_cfg.policy.init_noise_std,
    ).to(device)
    
    # Load weights
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    
    # CRITICAL: Set to eval mode (disables dropout, uses running stats for batchnorm)
    actor_critic.eval()
    
    print(f"[PIP-Loco] Model loaded successfully!")
    print(f"  - Actor dims: {train_cfg.policy.actor_hidden_dims}")
    print(f"  - Estimator dims: {train_cfg.estimator.hidden_dims}")
    print(f"  - Dreamer horizon: {train_cfg.policy.dreamer_horizon}")
    
    return actor_critic


# =============================================================================
# MAIN PLAY LOOP
# =============================================================================

def play():
    """
    Main entry point for interactive inference.
    
    Sets up environment, loads model, and runs the control loop with
    keyboard-based velocity commands.
    """
    
    # ==================================================================
    # 1. CONFIGURATION
    # ==================================================================
    env_cfg = PIPGO2Cfg()
    train_cfg = PIPTrainCfg()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[PIP-Loco] Using device: {device}")
    
    # Override for visualization
    # NOTE: Genesis has a bug with num_envs=1 visualization, so we use 2 but only control the first
    env_cfg.env.num_envs = 2
    env_cfg.noise.add_noise = False  # Disable observation noise for cleaner inference
    
    # Only render the first robot (index 0) - hides the unused second robot
    env_cfg.viewer.rendered_envs_idx = [0]
    
    # Camera position: place camera behind and above the robot, looking at spawn point
    # Robot spawns at init_state.pos = [0, 0, 0.42]
    env_cfg.viewer.pos = [2.0, 0.0, 1.0]      # 2m behind, 1m high
    env_cfg.viewer.lookat = [0.0, 0.0, 0.3]   # Look at robot's body height
    
    # ==================================================================
    # 2. INITIALIZE GENESIS
    # ==================================================================
    print("[PIP-Loco] Initializing Genesis simulator...")
    gs.init(logging_level="warning")
    
    # Simulation parameters
    sim_params = {
        "dt": env_cfg.sim.dt,
        "substeps": env_cfg.sim.substeps,
    }
    
    # Create environment with visualization enabled
    env = GenesisWrapper(
        cfg=env_cfg,
        sim_params=sim_params,
        sim_device=device,
        headless=False,  # Enable visualization
    )
    
    # Calculate control dt for real-time pacing
    control_dt = env_cfg.sim.dt * env_cfg.control.decimation  # 0.005 * 4 = 0.02s (50Hz)
    print(f"[PIP-Loco] Control frequency: {1.0/control_dt:.1f} Hz (dt={control_dt:.4f}s)")
    
    # ==================================================================
    # 3. LOAD MODEL
    # ==================================================================
    actor_critic = load_model(MODEL_PATH, env_cfg, train_cfg, device)
    
    # ==================================================================
    # 4. INITIALIZE KEYBOARD CONTROLLER
    # ==================================================================
    controller = KeyboardController(device=device)
    print("[PIP-Loco] Keyboard controller initialized")
    print("  Controls: W/S=Fwd/Back, A/D=Strafe, Arrows=Turn, R=Reset, ESC=Quit")
    
    # ==================================================================
    # 5. RESET ENVIRONMENT
    # ==================================================================
    # Base reset() returns (obs, privileged_obs) - need to do an extra step for history
    obs, privileged_obs = env.reset()
    
    # Do one step with zero actions to populate obs_history
    zero_actions = torch.zeros(env_cfg.env.num_envs, env_cfg.env.num_actions, device=device)
    obs, privileged_obs, _, _, extras = env.step(zero_actions)
    obs_history = extras['observations_history']
    
    print("\n[PIP-Loco] Starting simulation... Focus on the PyGame window for control!")
    print("=" * 60)
    
    # ==================================================================
    # 6. MAIN LOOP
    # ==================================================================
    step_count = 0
    
    try:
        with torch.no_grad():
            while controller.running:
                loop_start = time.perf_counter()
                
                # Update keyboard input
                if not controller.update():
                    break
                
                # Handle reset request
                if controller.reset_requested:
                    print("[PIP-Loco] Resetting environment...")
                    obs, privileged_obs = env.reset()
                    # Do one step with zero actions to populate obs_history
                    zero_actions = torch.zeros(env_cfg.env.num_envs, env_cfg.env.num_actions, device=device)
                    obs, privileged_obs, _, _, extras = env.step(zero_actions)
                    obs_history = extras['observations_history']
                    step_count = 0
                    continue
                
                # Get velocity commands from keyboard
                commands = controller.get_commands_tensor()
                
                # Inject commands into environment
                # Only control the first robot (index 0), second robot stands still
                # Commands are at indices 6:9 in the 45-dim blind observation
                # Format: [ang_vel(3), proj_gravity(3), commands(3), dof_pos(12), dof_vel(12), actions(12)]
                env.commands[0, 0] = commands[0, 0]  # vel_x
                env.commands[0, 1] = commands[0, 1]  # vel_y  
                env.commands[0, 2] = commands[0, 2]  # yaw_rate
                # Second robot stays stationary
                env.commands[1, :3] = 0.0
                
                # Get action from policy (deterministic inference)
                actions = actor_critic.act_inference(obs, obs_history)
                
                # Step environment
                obs, privileged_obs, rewards, dones, extras = env.step(actions)
                obs_history = extras['observations_history']
                
                step_count += 1
                
                # Print status periodically
                if step_count % 100 == 0:
                    print(f"[Step {step_count:6d}] "
                          f"Vx={commands[0,0]:+.2f} Vy={commands[0,1]:+.2f} Yaw={commands[0,2]:+.2f}")
                
                # Pace the loop to match real-time (50Hz)
                loop_elapsed = time.perf_counter() - loop_start
                sleep_time = control_dt - loop_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
    except KeyboardInterrupt:
        print("\n[PIP-Loco] Interrupted by user")
    except Exception as e:
        # Handle Genesis viewer closed gracefully
        if "Viewer closed" in str(e):
            print("\n[PIP-Loco] Viewer window closed")
        else:
            raise
    finally:
        # Clean up
        controller.close()
        print("[PIP-Loco] Simulation ended")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    play()