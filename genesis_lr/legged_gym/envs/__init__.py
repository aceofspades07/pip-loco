from legged_gym.utils.task_registry import task_registry
from .go1.go1 import Go1Env
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO

# CORRECT ORDER:
# 1. Name ("go1")
# 2. Environment Class (Go1Env)  <-- THIS MUST BE FIRST
# 3. Robot Config (Go1RoughCfg)
# 4. Training Config (Go1RoughCfgPPO)

task_registry.register(
    "go1", 
    Go1Env, 
    Go1RoughCfg, 
    Go1RoughCfgPPO
)