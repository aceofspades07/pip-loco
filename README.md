# PIP-Loco: Genesis Implementation for Unitree Go2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)
[![Genesis 0.3](https://img.shields.io/badge/Genesis-0.3-green.svg)](https://genesis-embodied-ai.github.io/)

![Unitree Go2 Walking](./assets/hero_walk.gif)

A fast, robust, blind RL-based locomotion policy for the Unitree Go2 quadruped. Trains on a **single consumer laptop GPU** (RTX 3050 Ti) in **~4 hours (~160M steps)** using 1024 parallel environments in the Genesis physics simulator. Produces sim-to-real ready policies via aggressive domain randomization and hardware-safe Quadratic Barrier constraints.

> **Evaluators & Researchers:** For the deep dive into the math, architecture, and gradient isolation strategy, see the [Technical Blog](<!-- PLACEHOLDER: INSERT GITHUB.IO BLOG LINK HERE -->).

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/aceofspades07/pip-loco.git
cd pip-loco

# Create and activate conda environment
conda create -n pip_genesis python=3.10 -y
conda activate pip_genesis

# Install PyTorch with CUDA support
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install Genesis physics simulator
pip install genesis-world==0.3.10

# Install remaining dependencies
pip install numpy==2.1.2 pandas scipy matplotlib tensorboard wandb tqdm libigl==2.5.1

# Install local packages in editable mode
pip install -e genesis_lr/
```

### Inference (Play)

```bash
# Run pre-trained policy with keyboard control
# W/S: Forward/Backward | A/D: Strafe | Arrows: Turn | R: Reset | ESC: Quit
python scripts/play.py
```

### Training

```bash
# Train from scratch (~4 hours on RTX 3050 Ti)
python scripts/train.py
```

Checkpoints are saved to `logs/pip_go2_<timestamp>/`.

---

## High-Level System Architecture

| Component | Description |
|-----------|-------------|
| **Asymmetric Actor-Critic** | Actor receives 45-dim blind proprioception (joint pos/vel, IMU, commands). Critic receives privileged simulator data (true velocity, friction, terrain heights). |
| **TCN Velocity Estimator** | 3-layer Temporal Convolutional Network that regresses body velocity from 50-step observation history — no kinematic assumptions. |
| **Dreamer (No-Latent Model)** | 4 independent MLPs (dynamics, reward, policy, value) that predict future states. Generates 5-step 'dreamed' rollouts fed to the actor. |
| **HybridTrainer** | Coordinates three separate optimizers with strict gradient isolation: Estimator (supervised learning), Dreamer (model-based learning), PPO (policy learning). |

---

## Hardware Safety & Sim-to-Real Hardening

### Domain Randomization

- **Friction:** Uniform `[0.25, 1.25]`
- **Payload Mass:** Uniform `[-2.0, +3.0] kg`
- **External Pushes:** `1.0 m/s` lateral impulse every `5s`
- **Control Latency:** `0–2` policy steps (`0–10 ms` at 50 Hz)
- **PD Gains:** Multiplier `[0.8, 1.2]`
- **CoM Displacement:** `±5 cm` in each axis

### Quadratic Barrier Safety Constraints

The policy is penalized when approaching hardware limits—not just at violation:

| Limit | Unitree Go2 Spec | Soft Threshold (90%) |
|-------|------------------|----------------------|
| Max Torque | 45 Nm | 40.5 Nm |
| Max Joint Velocity | 30 rad/s | 27 rad/s |

Penalties scale quadratically as the robot approaches these limits, promoting smooth, hardware-safe motions.

---

## Configuration Guide

All hyperparameters and domain randomization ranges are centralized in:

```
config/pip_config.py
```

Key sections:
- `PIPGO2Cfg.env` — Number of parallel envs, observation dims, episode length
- `PIPGO2Cfg.domain_rand` — Friction, mass, push, and latency ranges
- `PIPGO2Cfg.rewards.scales` — Reward component weights
- `PIPTrainCfg.algorithm` — Learning rates, PPO clip, gradient norm
- `PIPTrainCfg.policy` — Actor/critic network dimensions, dreamer horizon

---

## Roadmap

- [x] **Phase 1:** Blind flat-ground locomotion (Asymmetric Actor-Critic + TCN)
- [x] **Phase 2:** Internal Model training (NLM Dreamer) and Sim-to-Real hardening
- [ ] **Phase 3:** Model Predictive Path Integral (MPPI) planner integration
- [ ] **Phase 4:** Rough terrain curriculum and visual exteroception

---

## Acknowledgments & References

This implementation is based on **PIP-Loco** by Shirwatkar et al. (ICRA 2025).

**Project Website:** [https://www.stochlab.com/PIP-Loco/](https://www.stochlab.com/PIP-Loco/)

**Citation:**
```bibtex
@misc{shirwatkar2024piplocoproprioceptiveinfinitehorizon,
    title={PIP-Loco: A Proprioceptive Infinite Horizon Planning Framework for Quadrupedal Robot Locomotion}, 
    author={Aditya Shirwatkar and Naman Saxena and Kishore Chandra and Shishir Kolathaya},
    year={2024},
    eprint={2409.09441},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2409.09441}, 
}
```

**Additional Credits:**
- [Genesis Physics Simulator](https://genesis-embodied-ai.github.io/) — GPU-parallel rigid body simulation environment.
- [LeggedGym-Ex](https://github.com/lupinjia/LeggedGym-Ex) — Foundational environment wrapper (Formerly known as Genesis-LR).
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — A library of simple implementations of learning algorithms for robotics. 

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
