import genesis as gs
# Initialize Genesis First (Only once!)
gs.init(backend=gs.gpu)

import torch
import time
import numpy as np

# Import necessary components
from genesis_lr.legged_gym.envs.base.legged_robot import LeggedRobot
from genesis_lr.legged_gym.utils.task_registry import task_registry
from genesis_lr.legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO

# Helper class to force headless mode
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

def setup_genesis_env():
    print("=== Initializing Genesis Environment (Run Once) ===")

    # Register the task manually
    task_registry.register(
        name="go1_verification",
        task_class=LeggedRobot,
        env_cfg=Go1RoughCfg,
        train_cfg=Go1RoughCfgPPO
    )

    # Load Config
    env_cfg, train_cfg = task_registry.get_cfgs(name="go1_verification")
    
    # Override settings for testing
    print("Overriding num_envs to 1...")
    env_cfg.env.num_envs = 1
    # Disable rendering to prevent crash
    env_cfg.viewer.rendered_envs_idx = [] 
    
    # Create Environment
    print("Creating environment...")
    env, _ = task_registry.make_env(name="go1_verification", args=HeadlessArgs(), env_cfg=env_cfg)
    return env

def verify_sensors_and_motors(env):
    """
    Checks Proprioception (48), Gravity, Velocity, and Motor Control.
    """
    print("\n\n=== TESTING SENSORS & MOTORS ===")
    
    # Reset to get fresh observations
    obs_pack = env.reset()
    
    if isinstance(obs_pack, tuple):
        obs = obs_pack[0]
        priv_obs = obs_pack[1]
    else:
        obs = obs_pack
        priv_obs = None

    print("\n--- Sensor Integrity Check ---")
    
    # Verify Proprioceptive Observation Shape
    if obs is not None:
        print(f"Proprioceptive Shape: {obs.shape}")
        if obs.shape[-1] == 48:
            print("SUCCESS: Proprioceptive dimensions match paper (48)")
        else:
            print(f"FAILURE: Expected 48, got {obs.shape[-1]}")

    # Verify Privileged Observation Shape
    if priv_obs is not None:
        print(f"Privileged Shape:     {priv_obs.shape}")
        if priv_obs.shape[-1] == 67:
            print("SUCCESS: Privileged dimensions match config (67)")
        else:
            print(f"FAILURE: Expected 67, got {priv_obs.shape[-1]}")
    else:
        print("FAILURE: Privileged observations returned None")


    print("\n--- Physics Reality Check ---")
    
    # Check Gravity Vector
    gravity_vec = obs[0, 6:9].cpu().numpy()
    print(f"Projected Gravity: {gravity_vec}")
    if np.allclose(gravity_vec, [0, 0, -1.0], atol=0.05):
        print("SUCCESS: Gravity vector is pointing down")
    else:
        print("WARNING: Gravity vector looks wrong")

    # Check Base Velocity
    lin_vel = obs[0, 0:3].cpu().numpy()
    print(f"Base Linear Vel:   {lin_vel}")
    if np.allclose(lin_vel, [0, 0, 0], atol=0.1):
        print("SUCCESS: Robot is stationary at start")
    else:
        print("WARNING: Robot is drifting")


    print("\n--- Motor Control Check ---")
    
    raw_target = 0.5
    action_scale = env.cfg.control.action_scale
    target_offset_rad = raw_target * action_scale
    
    print(f"Commanding Joint 0 (FL_hip) with raw action: {raw_target}")
    print(f"Action Scale: {action_scale}")
    print(f"Target Offset: {target_offset_rad:.4f} rad")

    # Construct Action Tensor
    action = torch.zeros((1, env.num_actions), device=env.device)
    action[0, 0] = raw_target 

    final_error = 100.0
    
    # Run Simulation Loop
    for i in range(50):
        step_result = env.step(action)
        obs_new = step_result[0]
        
        # Access internal simulator state
        current_pos = env.simulator.dof_pos[0, 0].item()
        default_pos = env.simulator.default_dof_pos[0, 0].item()
        
        measured_offset = current_pos - default_pos
        final_error = abs(target_offset_rad - measured_offset)
        
        if i % 10 == 0:
            joint_0_obs = obs_new[0, 12].item()
            joint_1_obs = obs_new[0, 13].item()
            
            print(f"Step {i:02d} | Target Offset: {target_offset_rad:.3f} | Actual Offset: {measured_offset:.3f} | Error: {final_error:.3f}")
            print(f"         | Obs Joint 0: {joint_0_obs:.3f} | Obs Joint 1: {joint_1_obs:.3f}")
        
        time.sleep(0.02)

    print(f"\nFinal Error: {final_error:.4f}")
    if final_error < 0.1:
        print("SUCCESS: Motors tracked the command correctly")
    else:
        print("FAILURE: Motors did not reach target")


def verify_privileged_content(env):
    """
    Checks specific privileged data fields (Last Action, Mass, Friction).
    """
    print("\n\n=== TESTING PRIVILEGED OBSERVATIONS ===")
    
    # CRITICAL: Reset environment to clear previous motor tests
    print("Resetting environment for fresh test...")
    env.reset()

    print("\n--- Last Actions Memory Check ---")
    
    # Send Action A (All +1.0)
    action_A = torch.ones((1, 12), device=env.device) * 1.0
    print("Step 1: Sending Action A (All 1.0)...")
    _, priv_obs_1, _, _, _ = env.step(action_A)
    
    # Send Action B (All -1.0)
    action_B = torch.ones((1, 12), device=env.device) * -1.0
    print("Step 2: Sending Action B (All -1.0)...")
    _, priv_obs_2, _, _, _ = env.step(action_B)
    
    # CHECK: Privileged Obs should contain Action A (from Step 1)
    # Indices 48 to 59 correspond to 'last_actions'
    stored_last_action = priv_obs_2[0, 48:60].cpu().numpy()
    
    print(f"Stored 'Last Action' in Priv Buff: {stored_last_action[0]:.2f}")
    
    if np.allclose(stored_last_action, 1.0, atol=0.01):
        print("SUCCESS: System correctly remembered Action A (1.0) as the previous action.")
    else:
        print(f"FAILURE: Expected 1.0, got {stored_last_action[0]}")

   
    print("\n--- Physics Parameters Check ---")
    
    # Extract the tail of the privileged vector
    priv_tail = priv_obs_2[0, 60:].cpu().numpy()
    
    friction = priv_tail[0]
    mass = priv_tail[1]
    com_bias = priv_tail[2:5]
    push_vel = priv_tail[5:7]
    
    print(f"Index 60 [Friction]:  {friction:.4f} (Should be ~0.5 to 1.5)")
    print(f"Index 61 [Add Mass]:  {mass:.4f} (Should be >= 0)")
    print(f"Index 62-64 [CoM]:    {com_bias}")
    print(f"Index 65-66 [Push]:   {push_vel}")
    
    if friction > 0.01:
        print("SUCCESS: Friction is non-zero and physically valid.")
    else:
        print("WARNING: Friction seems to be zero.")
        
    if abs(mass) < 10.0:
        print("SUCCESS: Added mass is within reasonable limits.")
    else:
        print("WARNING: Added mass is suspiciously high.")


if __name__ == '__main__':
    # Setup Environment
    env = setup_genesis_env()
    
    # Run Test 1 (Sensors/Motors)
    verify_sensors_and_motors(env)
    
    # Run Test 2 (Privileged Observations)
    verify_privileged_content(env)