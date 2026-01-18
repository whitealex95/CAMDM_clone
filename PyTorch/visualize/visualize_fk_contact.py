"""
Visualize Forward Kinematics and Foot Contact Detection in MuJoCo

This script loads motion data and visualizes:
1. Ground truth motion from dataset
2. Forward kinematics computed joint positions
3. Foot contact masks (colored markers on feet)

Usage:
    python visualize/visualize_fk_contact.py --dataset lafan1_g1_motion27
"""

import os
import sys
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import draw_trajectory
import utils.g1_kinematics as g1_kin


class FKContactVisualizer:
    """Visualize FK results and foot contact detection."""

    def __init__(self, model, data, dataset):
        self.model = model
        self.data = data
        self.dataset = dataset

        # Playback state
        self.current_motion_idx = 0
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()

        # Display options
        self.show_fk_joints = True
        self.show_fk_skeleton = False  # Show skeleton connections
        self.show_foot_contact = True
        self.show_trajectory = False

        # Frame rate
        self.fps = 30
        self.frame_dt = 1.0 / self.fps

        # Load first motion
        self.load_motion(0)

        # Precompute FK and contact for current motion
        self.compute_fk_and_contact()

    def load_motion(self, motion_idx):
        """Load a specific motion clip."""
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion_data = self.dataset[self.current_motion_idx]
        self.current_frame = 0

        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Style: {self.current_motion_data.style}")
        print(f"Frames: {self.current_motion_data.num_frames}")
        print(f"Duration: {self.current_motion_data.num_frames / self.fps:.2f}s")
        print(f"{'='*60}\n")

        self.update_pose()

    def compute_fk_and_contact(self):
        """Compute FK and foot contact for entire current motion."""
        print("Computing FK and foot contact...")

        # Get all qpos for current motion
        all_qpos = []
        for frame_idx in range(self.current_motion_data.num_frames):
            qpos = self.current_motion_data.get_qpos(frame_idx)
            all_qpos.append(qpos)

        # Stack to [T, 36]
        qpos_sequence = np.stack(all_qpos, axis=0)
        qpos_tensor = torch.from_numpy(qpos_sequence).float().unsqueeze(0)  # [1, T, 36]

        # Compute FK
        with torch.no_grad():
            joint_positions = g1_kin.forward_kinematics_g1(qpos_tensor)  # [1, T, 30, 3]
            foot_positions = g1_kin.extract_foot_positions(joint_positions)  # [1, T, 2, 3]
            contact_mask = g1_kin.compute_foot_contact_mask(foot_positions, threshold=0.02)  # [1, T-1, 2]

        # Store results
        self.fk_joint_positions = joint_positions.squeeze(0).numpy()  # [T, 30, 3]
        self.fk_foot_positions = foot_positions.squeeze(0).numpy()    # [T, 2, 3]
        self.foot_contact_mask = contact_mask.squeeze(0).numpy()      # [T-1, 2]

        print(f"FK computed: {self.fk_joint_positions.shape}")
        print(f"Foot positions: {self.fk_foot_positions.shape}")
        print(f"Contact mask: {self.foot_contact_mask.shape}")

        # Print contact statistics
        left_contact_ratio = self.foot_contact_mask[:, 0].mean()
        right_contact_ratio = self.foot_contact_mask[:, 1].mean()
        print(f"Left foot contact: {left_contact_ratio:.1%}")
        print(f"Right foot contact: {right_contact_ratio:.1%}")
        print()

    def update_pose(self):
        """Update MuJoCo model with current frame pose."""
        qpos = self.current_motion_data.get_qpos(self.current_frame)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """Step the animation forward."""
        if not self.playing:
            return

        current_time = time.time()
        dt = current_time - self.last_update_time

        # Check if enough time has passed for next frame
        if dt >= self.frame_dt / self.playback_speed:
            self.current_frame += 1

            # Loop back to start when reaching end
            if self.current_frame >= self.current_motion_data.num_frames:
                self.current_frame = 0

            self.update_pose()
            self.last_update_time = current_time

    def render_fk_joints(self, scene):
        """Render FK joint positions as spheres."""
        if not self.show_fk_joints:
            return

        # Get FK positions for current frame
        joint_pos = self.fk_joint_positions[self.current_frame]  # [30, 3]

        # Render each joint as a sphere
        for i, pos in enumerate(joint_pos):
            # Color: different colors for different body parts (more transparent)
            if i <= 12:  # Root and legs
                color = [0.2, 0.7, 1.0, 0.4]  # Light blue, semi-transparent
                size = 0.03
            elif i <= 15:  # Waist
                color = [0.7, 0.2, 1.0, 0.4]  # Purple, semi-transparent
                size = 0.03
            else:  # Arms
                color = [1.0, 0.7, 0.2, 0.4]  # Orange, semi-transparent
                size = 0.025

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[size, 0, 0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=color
            )
            scene.ngeom += 1

            # Don't exceed max geoms
            if scene.ngeom >= scene.maxgeom:
                break

    def render_fk_skeleton(self, scene):
        """Render FK skeleton as lines connecting joints."""
        if not self.show_fk_skeleton:
            return

        # Get FK positions for current frame
        joint_pos = self.fk_joint_positions[self.current_frame]  # [30, 3]

        # Define parent indices (from g1_kinematics.py)
        parents = [
            -1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 15, 23, 24, 25, 26, 27, 28
        ]

        # Draw lines from each joint to its parent
        for i in range(1, len(joint_pos)):
            parent_idx = parents[i]
            if parent_idx < 0:
                continue

            # Get positions
            pos_child = joint_pos[i]
            pos_parent = joint_pos[parent_idx]

            # Line color based on body part
            if i <= 6:  # Left leg
                color = [0.3, 0.8, 1.0, 0.8]
            elif i <= 12:  # Right leg
                color = [1.0, 0.3, 0.8, 0.8]
            elif i <= 15:  # Torso
                color = [0.8, 0.8, 0.3, 0.8]
            elif i <= 22:  # Left arm
                color = [0.3, 1.0, 0.5, 0.8]
            else:  # Right arm
                color = [1.0, 0.5, 0.3, 0.8]

            # Compute line direction and length
            direction = pos_child - pos_parent
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            # Midpoint
            midpoint = (pos_parent + pos_child) / 2

            # Create rotation matrix to align cylinder with direction
            z_axis = direction / length
            # Find perpendicular vector
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            rot_mat = np.column_stack([x_axis, y_axis, z_axis])

            # Render as capsule (cylinder with rounded ends)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[0.008, length/2, 0],  # radius, half-length
                pos=midpoint,
                mat=rot_mat.flatten(),
                rgba=color
            )
            scene.ngeom += 1

            if scene.ngeom >= scene.maxgeom:
                break

    def render_foot_contact(self, scene):
        """Render foot contact markers."""
        if not self.show_foot_contact or self.current_frame == 0:
            return

        # Get contact mask for current frame (frame i uses contact between i-1 and i)
        frame_idx = min(self.current_frame - 1, len(self.foot_contact_mask) - 1)
        contact = self.foot_contact_mask[frame_idx]  # [2] (left, right)

        # Get foot positions
        left_foot_pos = self.fk_foot_positions[self.current_frame, 0]   # [3]
        right_foot_pos = self.fk_foot_positions[self.current_frame, 1]  # [3]

        # Render contact markers
        for foot_idx, (pos, is_contact) in enumerate([
            (left_foot_pos, contact[0]),
            (right_foot_pos, contact[1])
        ]):
            # Color: red if in contact, transparent otherwise
            if is_contact > 0.5:
                color = [1.0, 0.0, 0.0, 0.8]  # Bright red (contact)
                size = 0.04
            else:
                color = [0.2, 1.0, 0.2, 0.3]  # Faint green (no contact)
                size = 0.03

            # Marker slightly below foot
            marker_pos = pos.copy()
            marker_pos[2] -= 0.05

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[size, 0, 0],
                pos=marker_pos,
                mat=np.eye(3).flatten(),
                rgba=color
            )
            scene.ngeom += 1

            # Add contact flag above foot
            if is_contact > 0.5:
                flag_pos = pos.copy()
                flag_pos[2] += 0.1
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.01, 0.05, 0],
                    pos=flag_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.0, 0.0, 0.9]
                )
                scene.ngeom += 1

            if scene.ngeom >= scene.maxgeom:
                break

    def render_hand_positions(self, scene):
        """Render hand positions as highlighted spheres."""
        if not self.show_fk_joints:
            return

        # Get FK positions for current frame
        joint_pos = self.fk_joint_positions[self.current_frame]  # [30, 3]

        # Hand joint indices (from g1_kinematics.py)
        # Left hand: joint 22 (left_wrist_yaw)
        # Right hand: joint 29 (right_wrist_yaw)
        hand_indices = [22, 29]
        hand_colors = [
            [0.0, 1.0, 0.0, 0.9],  # Left hand: bright green
            [1.0, 0.5, 0.0, 0.9]   # Right hand: bright orange
        ]

        for hand_idx, color in zip(hand_indices, hand_colors):
            pos = joint_pos[hand_idx]

            # Render hand as larger, brighter sphere
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.04, 0, 0],  # Larger than joint spheres
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=color
            )
            scene.ngeom += 1

            if scene.ngeom >= scene.maxgeom:
                break

    def render_scene(self, scene):
        """Render all visualization elements."""
        # Clear previous custom geoms
        scene.ngeom = 0

        # Render FK skeleton (lines between joints)
        self.render_fk_skeleton(scene)

        # Render FK joints (spheres at joint positions)
        self.render_fk_joints(scene)

        # Render hand positions (highlighted)
        self.render_hand_positions(scene)

        # Render foot contact
        self.render_foot_contact(scene)

    def next_motion(self):
        """Load next motion clip."""
        self.load_motion(self.current_motion_idx + 1)
        self.compute_fk_and_contact()

    def prev_motion(self):
        """Load previous motion clip."""
        self.load_motion(self.current_motion_idx - 1)
        self.compute_fk_and_contact()

    def toggle_pause(self):
        """Toggle play/pause."""
        self.playing = not self.playing
        print(f"{'Playing' if self.playing else 'Paused'}")

    def toggle_fk_joints(self):
        """Toggle FK joint visualization."""
        self.show_fk_joints = not self.show_fk_joints
        print(f"FK joints: {'ON' if self.show_fk_joints else 'OFF'}")

    def toggle_fk_skeleton(self):
        """Toggle FK skeleton visualization."""
        self.show_fk_skeleton = not self.show_fk_skeleton
        print(f"FK skeleton: {'ON' if self.show_fk_skeleton else 'OFF'}")

    def toggle_foot_contact(self):
        """Toggle foot contact visualization."""
        self.show_foot_contact = not self.show_foot_contact
        print(f"Foot contact: {'ON' if self.show_foot_contact else 'OFF'}")

    def set_speed(self, speed):
        """Set playback speed multiplier."""
        self.playback_speed = speed
        print(f"Playback speed: {speed}x")

    def print_status(self):
        """Print current status."""
        frame_idx = min(self.current_frame - 1, len(self.foot_contact_mask) - 1)
        if frame_idx >= 0:
            left_contact = self.foot_contact_mask[frame_idx, 0]
            right_contact = self.foot_contact_mask[frame_idx, 1]
            contact_str = f"L:{left_contact:.0f} R:{right_contact:.0f}"
        else:
            contact_str = "N/A"

        status = (
            f"Motion: {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"Frame: {self.current_frame}/{self.current_motion_data.num_frames} | "
            f"Contact: {contact_str} | "
            f"{'Playing' if self.playing else 'Paused'} ({self.playback_speed}x)"
        )
        print(status)


def key_callback(visualizer: FKContactVisualizer, keycode):
    """Handle keyboard input."""
    if keycode == 32:  # SPACE
        visualizer.toggle_pause()
    elif keycode == 265:  # UP
        visualizer.next_motion()
    elif keycode == 264:  # DOWN
        visualizer.prev_motion()
    elif keycode == ord('f') or keycode == ord('F'):
        visualizer.toggle_fk_joints()
    elif keycode == ord('k') or keycode == ord('K'):
        visualizer.toggle_fk_skeleton()
    elif keycode == ord('c') or keycode == ord('C'):
        visualizer.toggle_foot_contact()
    elif keycode == ord('s') or keycode == ord('S'):
        visualizer.print_status()
    elif ord('1') <= keycode <= ord('9'):
        speed_map = {
            ord('1'): 0.25,
            ord('2'): 0.5,
            ord('3'): 0.75,
            ord('4'): 0.9,
            ord('5'): 1.0,
            ord('6'): 1.25,
            ord('7'): 1.5,
            ord('8'): 1.75,
            ord('9'): 2.0,
        }
        visualizer.set_speed(speed_map[keycode])


def print_instruction():
    print("\n" + "=" * 60)
    print("Controls:")
    print("-" * 60)
    print("  SPACE       : Pause/Resume")
    print("  UP/DOWN     : Previous/Next motion clip")
    print("  F           : Toggle FK joint visualization (spheres)")
    print("  K           : Toggle FK skeleton visualization (wireframe)")
    print("  C           : Toggle foot contact visualization")
    print("  1-9         : Set playback speed (1=0.25x, 5=1x, 9=2x)")
    print("  S           : Print status")
    print("  ESC         : Exit")
    print("=" * 60)
    print("\nVisualization Legend:")
    print("  FK Joint Spheres (toggle with F):")
    print("    - Blue spheres       : FK computed joint positions (legs)")
    print("    - Purple spheres     : FK computed joint positions (waist)")
    print("    - Orange spheres     : FK computed joint positions (arms)")
    print("    - Bright green sphere: Left hand position (highlighted)")
    print("    - Bright orange sphere: Right hand position (highlighted)")
    print("  FK Skeleton Wireframe (toggle with K):")
    print("    - Cyan lines      : Left leg connections")
    print("    - Pink lines      : Right leg connections")
    print("    - Yellow lines    : Torso connections")
    print("    - Green lines     : Left arm connections")
    print("    - Orange lines    : Right arm connections")
    print("  Foot Contact (toggle with C):")
    print("    - Red marker      : Foot in contact with ground")
    print("    - Green marker    : Foot not in contact")
    print("    - Red cylinder    : Contact flag")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize FK and foot contact")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lafan1_g1_motion27",
        help="Dataset name (pkl file in data/pkls/)"
    )
    parser.add_argument(
        "--motion",
        type=int,
        default=0,
        help="Starting motion index"
    )
    args = parser.parse_args()

    # Paths
    scene_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "scene.xml"
    )
    dataset_path = f"data/pkls/{args.dataset}.pkl"

    print("=" * 60)
    print("FK and Foot Contact Visualization")
    print("=" * 60)

    # Load MuJoCo model
    print(f"\nLoading MuJoCo scene: {scene_path}")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)

    # Load motion dataset
    print(f"Loading motion dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found: {dataset_path}")
        return

    dataset = MotionDataset(dataset_path)
    dataset.print_summary()

    # Create visualizer
    visualizer = FKContactVisualizer(mj_model, mj_data, dataset)

    # Start from specified motion
    if args.motion > 0:
        visualizer.load_motion(args.motion)
        visualizer.compute_fk_and_contact()

    print_instruction()

    # Launch viewer
    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=lambda keycode: key_callback(visualizer, keycode)
    ) as viewer:
        viewer.sync()
        try:
            while viewer.is_running():
                visualizer.step()
                visualizer.render_scene(viewer.user_scn)
                viewer.sync()
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
