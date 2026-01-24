from legged_gym.envs.base.legged_robot import LeggedRobot
from .go1_config import Go1RoughCfg

class Go1Env(LeggedRobot):
    """
    The Go1 Environment class for Genesis.
    """
    # FIX: Use *args and **kwargs to accept whatever the registry passes
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)