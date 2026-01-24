import builtins
import torch

# 1. Set the Simulator backend global
builtins.SIMULATOR = 'genesis'

import genesis as gs

# 2. Import Config (Now safely finds the package, not the folder)
from genesis_lr.legged_gym.envs.go1.go1_config import Go1RoughCfg
from genesis_lr.legged_gym.envs.go2.go2 import GO2Env 

def test_robot():
    print("--- Initializing Go1 Environment ---")
    
    cfg = Go1RoughCfg()
    cfg.terrain.mesh_type = 'plane' 
    
    print(f"Loading Asset: {cfg.asset.file}")
    env = GO2Env(num_envs=1, cfg=cfg, sim_device='cuda:0', graphics_device_id=0, headless=False)
    
    obs, _ = env.reset()
    start_angle = env.dof_pos[0, 0].item()
    print(f"\n[Start] Joint 0: {start_angle:.4f} rad")

    print(">>> Commanding Joint 0 to move +0.25 rad...")
    action = torch.zeros(1, 12, device='cuda:0')
    action[0, 0] = 1.0 
    
    for i in range(50):
        obs, _, _, _, _ = env.step(action)
        
    end_angle = env.dof_pos[0, 0].item()
    print(f"[End]   Joint 0: {end_angle:.4f} rad")
    
    if end_angle > start_angle + 0.1:
        print("\n✅ SUCCESS: The robot physically moved!")
    else:
        print("\n❌ FAILURE: The robot did not move.")

if __name__ == '__main__':
    test_robot()