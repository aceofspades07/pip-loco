import genesis as gs
# 1. Initialize Genesis First
gs.init(backend=gs.gpu)

import torch
import time
import numpy as np

# 2. Import necessary components
from genesis_lr.legged_gym.envs.base.legged_robot import LeggedRobot
from genesis_lr.legged_gym.utils.task_registry import task_registry
from genesis_lr.legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO

# --- HELPER CLASS TO FORCE HEADLESS MODE ---
class HeadlessArgs:
    headless = True          
    task = "go1_verification"
    resume = False
    run_name = "test"
    num_envs = 1 
    
    # Hardware/Pipeline settings
    cpu = False
    use_gpu_pipeline = True
    device = "cuda:0"

def verify_environment():
    print("=== PIP-Loco Environment Verification (Final) ===")

    # 3. MANUAL REGISTRATION
    task_registry.register(
        name="go1_verification",
        task_class=LeggedRobot,
        env_cfg=Go1RoughCfg,
        train_cfg=Go1RoughCfgPPO
    )

    # 4. Load Config
    env_cfg, train_cfg = task_registry.get_cfgs(name="go1_verification")
    
    # 5. OVERRIDE SETTINGS
    print("Overriding num_envs to 1...")
    env_cfg.env.num_envs = 1
    env_cfg.viewer.rendered_envs_idx = [] 
    
    # 6. Create Environment
    print("Creating environment...")
    env, _ = task_registry.make_env(name="go1_verification", args=HeadlessArgs(), env_cfg=env_cfg)
    
    # 7. Check Observation Shapes
    print("\n--- 1. Sensor Check ---")
    obs_pack = env.reset()
    
    if isinstance(obs_pack, tuple):
        obs = obs_pack[0]
        priv_obs = obs_pack[1]
    else:
        obs = obs_pack
        priv_obs = None

    print(f"Proprioceptive Obs Shape: {obs.shape}") 
    if obs.shape[-1] == 48: 
        print("✅ Proprioceptive matches Paper (48 dimensions)")
    else:
        print(f"⚠️ Shape mismatch! Expected last dim 48, got {obs.shape[-1]}")

    # Check Privileged Obs
    if priv_obs is not None:
        print(f"Privileged Obs Shape:     {priv_obs.shape}")
        if priv_obs.shape[-1] == 67:
             print("✅ Privileged Obs matches Config (67 dimensions)")
        else:
             print(f"⚠️ Privileged mismatch! Expected 67, got {priv_obs.shape[-1]}")
    else:
        print("❌ Privileged observations are MISSING.")

    # 8. Motor Check (Action -> Movement)
    print("\n--- 2. Motor Check ---")
    print("Test: Commanding Front-Right Hip (Joint 0) to +0.5 (Raw Action)...")
    
    raw_target = 0.5
    action_scale = env.cfg.control.action_scale # Fetch scale from config (0.25)
    real_target = raw_target * action_scale
    
    print(f"Action Scale: {action_scale}")
    print(f"Expected Physical Movement: {real_target:.4f} rad")

    action = torch.zeros((1, env.num_actions), device=env.device)
    action[0, 0] = raw_target 

    final_error = 100.0
    for i in range(50):
        env.step(action)
        
        # Access simulator object
        current_pos = env.simulator.dof_pos[0, 0].item()
        default_pos = env.simulator.default_dof_pos[0, 0].item()
        measured_offset = current_pos - default_pos
        
        # Compare against REAL target (scaled)
        final_error = abs(real_target - measured_offset)
        
        if i % 10 == 0:
            print(f"Step {i:02d} | Target: {real_target:.3f} | Actual: {measured_offset:.3f} | Error: {final_error:.3f}")
        
        time.sleep(0.02)

    print(f"\nFinal Error: {final_error:.4f}")
    if final_error < 0.1:
        print("✅ Motors are responding correctly.")
    else:
        print("❌ Motors are NOT moving.")

if __name__ == '__main__':
    verify_environment()