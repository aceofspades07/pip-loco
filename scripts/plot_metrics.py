"""
Visualization and metrics script for PIP-Loco gait analysis.
Generates phase diagrams, contact force plots, and computes quantitative metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Physical constants
MASS = 15.0
GRAVITY = 9.81
CONTACT_THRESHOLD = 1.0
MIN_VEL_COT = 0.1

LEG_NAMES = ['FL', 'FR', 'RL', 'RR']
LEG_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_data(filepath='gait_data.npy'):
    return np.load(filepath, allow_pickle=True).item()


def plot_phase_diagram(contact_forces, dt=0.02, save_path='phase_diagram.png'):
    """Broken bar plot showing contact intervals for each leg."""
    T = contact_forces.shape[0]
    time = np.arange(T) * dt
    contacts = contact_forces > CONTACT_THRESHOLD

    fig, ax = plt.subplots(figsize=(12, 3))
    
    for leg_idx, (name, color) in enumerate(zip(LEG_NAMES, LEG_COLORS)):
        contact_seq = contacts[:, leg_idx]
        intervals = []
        start = None
        
        for t in range(T):
            if contact_seq[t] and start is None:
                start = time[t]
            elif not contact_seq[t] and start is not None:
                intervals.append((start, time[t] - start))
                start = None
        if start is not None:
            intervals.append((start, time[-1] - start))
        
        ax.broken_barh(intervals, (leg_idx - 0.3, 0.6), facecolors=color, edgecolors='black', linewidth=0.5)
    
    ax.set_yticks(range(len(LEG_NAMES)))
    ax.set_yticklabels(LEG_NAMES)
    ax.set_xlabel('Time (s)')
    ax.set_title('Gait Phase Diagram')
    ax.set_xlim(0, time[-1])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_contact_forces(contact_forces, dt=0.02, save_path='contact_forces.png'):
    """4-subplot figure of continuous vertical contact force per leg."""
    T = contact_forces.shape[0]
    time = np.arange(T) * dt

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    for leg_idx, (ax, name, color) in enumerate(zip(axes, LEG_NAMES, LEG_COLORS)):
        ax.plot(time, contact_forces[:, leg_idx], color=color, linewidth=0.8)
        ax.axhline(CONTACT_THRESHOLD, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel(f'{name} (N)')
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Vertical Contact Forces')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def compute_metrics(data):
    """Compute and print quantitative metrics."""
    contact_forces = data['contact_forces']
    base_lin_vel = data['base_lin_vel']
    commanded_vel = data['commanded_vel']
    joint_torques = data['joint_torques']
    joint_velocities = data['joint_velocities']
    projected_gravity = data['projected_gravity']

    # Success rate: percentage of timesteps with exactly 3 feet in contact
    contacts = contact_forces > CONTACT_THRESHOLD
    num_contacts = contacts.sum(axis=1)
    success_rate = (num_contacts == 3).mean() * 100.0

    # Velocity tracking RMSE (X and Y)
    vel_error_x = commanded_vel[:, 0] - base_lin_vel[:, 0]
    vel_error_y = commanded_vel[:, 1] - base_lin_vel[:, 1]
    rmse_vx = np.sqrt((vel_error_x ** 2).mean())
    rmse_vy = np.sqrt((vel_error_y ** 2).mean())
    rmse_total = np.sqrt(((vel_error_x ** 2) + (vel_error_y ** 2)).mean())

    # Cost of Transport: Power / (m * g * v)
    power = np.abs(joint_torques * joint_velocities).sum(axis=1)
    speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)
    speed_clipped = np.clip(speed, MIN_VEL_COT, None)
    cot = power / (MASS * GRAVITY * speed_clipped)
    cot_mean = cot.mean()

    # Base stability: std of roll/pitch from projected gravity
    roll_proxy = projected_gravity[:, 1]
    pitch_proxy = projected_gravity[:, 0]
    roll_std = np.std(roll_proxy)
    pitch_std = np.std(pitch_proxy)

    print("\n===== Quantitative Metrics =====")
    print(f"Success Rate (3-foot contact): {success_rate:.2f}%")
    print(f"Velocity Tracking RMSE (X):    {rmse_vx:.4f} m/s")
    print(f"Velocity Tracking RMSE (Y):    {rmse_vy:.4f} m/s")
    print(f"Velocity Tracking RMSE (XY):   {rmse_total:.4f} m/s")
    print(f"Cost of Transport (mean):      {cot_mean:.4f}")
    print(f"Roll Stability (std):          {roll_std:.4f}")
    print(f"Pitch Stability (std):         {pitch_std:.4f}")
    print("================================\n")


def main():
    data = load_data()
    
    plot_phase_diagram(data['contact_forces'])
    plot_contact_forces(data['contact_forces'])
    compute_metrics(data)


if __name__ == "__main__":
    main()
