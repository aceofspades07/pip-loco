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
    headless = False          
    task = "go1_verification"
    resume = False
    run_name = "test"
    num_envs = 1 
    
    # Hardware/Pipeline settings
    cpu = False
    use_gpu_pipeline = True
    device = "cuda:0"

def setup_genesis_env():
    
    print("=== Initializing Genesis Environment ===")

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


if __name__ == '__main__':
    env = setup_genesis_env()



    