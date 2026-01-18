"""
Demo: Visualizing Config-Based Keypoints

This script demonstrates how to visualize keypoints defined in robot config.
Run with: python visualize/visualize_keypoints_demo.py
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.robot_config import load_robot_config
from utils.keypoints import compute_keypoints_from_qpos, get_foot_keypoint_positions, get_hand_keypoint_positions


def demo_keypoint_computation():
    """Demonstrate keypoint computation from qpos."""
    print("=" * 80)
    print("Config-Based Keypoint Computation Demo")
    print("=" * 80)

    # Load G1 robot config
    config = load_robot_config("g1")
    print(f"\nRobot: {config.name}")
    print(f"Number of joints: {config.num_joints}")
    print(f"Number of keypoints: {len(config.keypoints)}")

    # List all keypoints
    print("\nDefined keypoints:")
    for kp_name, kp_config in config.keypoints.items():
        print(f"  - {kp_name:15s} (parent: {kp_config.parent_joint:25s}, "
              f"offset: {kp_config.offset}, type: {kp_config.type})")

    # Create sample motion data
    print("\nGenerating sample motion...")
    B, T = 1, 100  # 1 batch, 100 frames
    qpos = torch.zeros(B, T, 36)

    # Set root position (slight forward motion)
    qpos[:, :, 0] = torch.linspace(0, 1, T)  # Move forward 1 meter
    qpos[:, :, 2] = 0.75  # Height

    # Set root quaternion (identity)
    qpos[:, :, 3] = 1.0  # w = 1
    qpos[:, :, 4:7] = 0.0  # xyz = 0

    # Add some joint motion (simple walking pattern)
    t = torch.linspace(0, 4 * np.pi, T)
    qpos[:, :, 7] = 0.3 * torch.sin(t)  # Left hip pitch
    qpos[:, :, 10] = 0.5 * torch.sin(t)  # Left knee
    qpos[:, :, 13] = 0.3 * torch.sin(t + np.pi)  # Right hip pitch
    qpos[:, :, 16] = 0.5 * torch.sin(t + np.pi)  # Right knee

    # Compute all keypoints
    print(f"\nComputing keypoints for qpos shape: {qpos.shape}")
    keypoints = compute_keypoints_from_qpos(qpos)

    print(f"\nComputed keypoints:")
    for kp_name, kp_pos in keypoints.items():
        print(f"  {kp_name:15s}: {kp_pos.shape}")

    # Get specific keypoint groups
    foot_kps = get_foot_keypoint_positions(qpos)
    hand_kps = get_hand_keypoint_positions(qpos)

    print(f"\nKeypoint groups:")
    print(f"  Foot keypoints: {foot_kps.shape}  # [left_heel, left_toe, right_heel, right_toe]")
    print(f"  Hand keypoints: {hand_kps.shape}  # [left_hand, right_hand]")

    # Analyze foot motion
    print(f"\nFoot height analysis (frame 0):")
    foot_names = ["left_heel", "left_toe", "right_heel", "right_toe"]
    for i, name in enumerate(foot_names):
        height = foot_kps[0, 0, i, 2].item()
        print(f"  {name:15s}: z = {height:.4f} m")

    # Compute foot velocities
    foot_vel = (foot_kps[:, 1:] - foot_kps[:, :-1]) * 30  # Assuming 30 fps
    foot_speed = torch.norm(foot_vel, dim=-1)  # [B, T-1, 4]

    print(f"\nFoot speed statistics (m/s):")
    for i, name in enumerate(foot_names):
        speed = foot_speed[0, :, i]
        print(f"  {name:15s}: mean={speed.mean().item():.4f}, "
              f"max={speed.max().item():.4f}, min={speed.min().item():.4f}")

    print("\n✓ Demo completed successfully!")
    print("\nTo integrate these keypoints into visualization:")
    print("  1. Load config: config = load_robot_config('g1')")
    print("  2. Compute keypoints: keypoints = compute_keypoints_from_qpos(qpos)")
    print("  3. Render as spheres with colors from config.viz_keypoint_colors")


def show_integration_example():
    """Show example code for integrating into visualizer."""
    print("\n" + "=" * 80)
    print("Integration Example for visualize_fk_contact.py")
    print("=" * 80)

    example_code = '''
# Add to visualizer __init__:
from utils.robot_config import load_robot_config
from utils.keypoints import compute_keypoints_from_qpos

self.robot_config = load_robot_config("g1")
self.show_keypoints = True  # Toggle for keypoint visualization

# In compute_fk_and_contact method, add:
self.keypoint_positions = {}
for frame_idx in range(num_frames):
    qpos = torch.from_numpy(motion_data.qpos[frame_idx]).float().unsqueeze(0)
    frame_keypoints = compute_keypoints_from_qpos(qpos)
    for kp_name, kp_pos in frame_keypoints.items():
        if kp_name not in self.keypoint_positions:
            self.keypoint_positions[kp_name] = []
        self.keypoint_positions[kp_name].append(kp_pos.squeeze(0).numpy())

# Convert to arrays
for kp_name in self.keypoint_positions:
    self.keypoint_positions[kp_name] = np.array(self.keypoint_positions[kp_name])

# Add new render method:
def render_keypoints(self, scene):
    """Render config-defined keypoints."""
    if not self.show_keypoints:
        return

    for kp_name, kp_config in self.robot_config.keypoints.items():
        if kp_name not in self.keypoint_positions:
            continue

        pos = self.keypoint_positions[kp_name][self.current_frame]
        color = self.robot_config.viz_keypoint_colors.get(kp_name, [1, 1, 1, 0.8])

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.035, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color
        )
        scene.ngeom += 1

# Add to render_scene:
self.render_keypoints(scene)

# Add keyboard binding:
elif keycode == ord('p') or keycode == ord('P'):
    visualizer.toggle_keypoints()
'''

    print(example_code)


if __name__ == "__main__":
    demo_keypoint_computation()
    show_integration_example()
