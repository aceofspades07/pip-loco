import os
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

class Go1RoughCfg(GO2Cfg):
    class env(GO2Cfg.env):
        num_envs = 128
        num_observations = 48
        num_privileged_obs = 67
        num_actions = 12

    class init_state(GO2Cfg.init_state):
        pos = [0.0, 0.0, 0.42] 
        default_joint_angles = { 
            'FL_hip_joint': 0.1,   'FR_hip_joint': -0.1,   'RL_hip_joint': 0.1,   'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8,  'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5,  'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(GO2Cfg.control):
        stiffness = {'joint': 20.0}
        damping = {'joint': 0.5}
        action_scale = 0.25 

    class asset(GO2Cfg.asset):
        file = os.path.expanduser("~/pip_loco/robot_assets/resources/robots/go1/urdf/go1.urdf")
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 

    # --- ADD THIS TO FIX THE CAMERA VIEW ---
    class viewer(GO2Cfg.viewer):
        ref_env = 0             # Center camera on Robot 0
        pos = [3.0, -3.0, 2.0]  # x, y, z position (3 meters away)
        lookat = [0., 0, 0.]    # Look at the robot center
  
class Go1RoughCfgPPO(GO2CfgPPO):
    class runner(GO2CfgPPO.runner):
        run_name = 'go1_experiment'
        experiment_name = 'go1_rough'