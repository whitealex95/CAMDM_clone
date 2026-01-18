"""
Generic Batched Forward Kinematics for Humanoid Robots

This module provides config-based FK computation that works with any humanoid robot
defined via a YAML configuration file.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from utils.robot_config import RobotConfig, load_robot_config


def quat_to_rotmat(quat):
    """
    Convert quaternions (wxyz) to rotation matrices.

    Args:
        quat: [..., 4] quaternions in wxyz format

    Returns:
        [..., 3, 3] rotation matrices
    """
    # Normalize quaternion
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Compute rotation matrix elements
    rot_matrix = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)

    return rot_matrix


def axis_angle_to_rotmat(axis, angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    Args:
        axis: [3] or [..., 3] normalized rotation axis
        angle: [...] rotation angle in radians

    Returns:
        [..., 3, 3] rotation matrices
    """
    # Expand axis to match angle's batch dimensions if needed
    if axis.dim() == 1:  # axis is [3]
        axis = axis.expand(*angle.shape, 3)  # [..., 3]

    # Normalize axis
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)

    # Extract axis components
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    # Compute sin and cos
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    # Build rotation matrix using Rodrigues' formula
    r00 = cos_angle + x*x*one_minus_cos
    r01 = x*y*one_minus_cos - z*sin_angle
    r02 = x*z*one_minus_cos + y*sin_angle
    r10 = y*x*one_minus_cos + z*sin_angle
    r11 = cos_angle + y*y*one_minus_cos
    r12 = y*z*one_minus_cos - x*sin_angle
    r20 = z*x*one_minus_cos - y*sin_angle
    r21 = z*y*one_minus_cos + x*sin_angle
    r22 = cos_angle + z*z*one_minus_cos

    # Stack into rotation matrix
    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).reshape(*angle.shape, 3, 3)

    return R


class HumanoidFK:
    """
    Generic forward kinematics for humanoid robots.

    This class uses robot configuration to compute FK for any humanoid robot.
    """

    def __init__(self, robot_config: RobotConfig, device: str = "cpu"):
        """
        Args:
            robot_config: Robot configuration object
            device: Device for computation ("cpu" or "cuda")
        """
        self.config = robot_config
        self.device = device

        # Convert skeleton data to torch tensors
        self.parents = torch.from_numpy(robot_config.joint_parents).to(device)
        self.offsets = torch.from_numpy(robot_config.joint_offsets).to(device).float()
        self.axes = torch.from_numpy(robot_config.joint_axes).to(device).float()
        self.init_quats = torch.from_numpy(robot_config.body_init_quats).to(device).float()

        self.num_joints = robot_config.num_joints

    def forward_kinematics(
        self,
        qpos: torch.Tensor,
        return_rotations: bool = False
    ) -> torch.Tensor:
        """
        Batched Forward Kinematics.

        Converts joint configuration (root pose + joint angles) to 3D positions of all joints.

        Args:
            qpos: [B, T, qpos_dim] or [B, qpos_dim] or [T, qpos_dim] tensor
                - qpos[..., 0:3]: root position (XYZ)
                - qpos[..., 3:7]: root quaternion (WXYZ)
                - qpos[..., 7:]: N joint angles (1D revolute joints)
            return_rotations: If True, also return rotation matrices

        Returns:
            positions: [..., num_joints, 3] global 3D positions of all joints
            rotations (optional): [..., num_joints, 3, 3] global rotation matrices
        """
        # Handle different input shapes
        original_shape = qpos.shape
        if len(qpos.shape) == 2:  # [B, qpos_dim] or [T, qpos_dim]
            qpos = qpos.unsqueeze(1)  # [B, 1, qpos_dim]
            squeeze_time = True
        else:
            squeeze_time = False

        B, T = qpos.shape[0], qpos.shape[1]
        device = qpos.device

        # Parse qpos
        root_pos = qpos[..., :3]  # [B, T, 3]
        root_quat = qpos[..., 3:7]  # [B, T, 4] (wxyz)
        joint_angles = qpos[..., 7:]  # [B, T, num_actuated_joints]

        # Prepare output
        positions = torch.zeros(B, T, self.num_joints, 3, device=device)
        rotations = torch.zeros(B, T, self.num_joints, 3, 3, device=device)

        # Set root
        positions[:, :, 0] = root_pos
        rotations[:, :, 0] = quat_to_rotmat(root_quat)

        # Forward pass through kinematic chain
        for i in range(1, self.num_joints):
            parent_idx = self.parents[i].item()

            # Get parent's global rotation and position
            R_parent = rotations[:, :, parent_idx].clone()  # [B, T, 3, 3]
            p_parent = positions[:, :, parent_idx].clone()  # [B, T, 3]

            # Get initial body orientation (from XML quat field)
            init_quat = self.init_quats[i]  # [4] wxyz
            R_init = quat_to_rotmat(init_quat)  # [3, 3]
            R_init = R_init.unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3)  # [B, T, 3, 3]

            # Local rotation for this joint (1D revolute joint around axis)
            angle = joint_angles[:, :, i-1]  # [B, T]
            axis = self.axes[i]  # [3]
            R_joint = axis_angle_to_rotmat(axis, angle)  # [B, T, 3, 3]

            # Total local rotation: R_local = R_init @ R_joint
            R_local = torch.matmul(R_init, R_joint)  # [B, T, 3, 3]

            # Global rotation: R_global = R_parent @ R_local
            R_global = torch.matmul(R_parent, R_local)  # [B, T, 3, 3]
            rotations[:, :, i] = R_global

            # Local offset in parent's frame
            offset = self.offsets[i]  # [3]

            # Transform offset to global frame: offset_global = R_parent @ offset
            offset_global = torch.matmul(R_parent, offset.unsqueeze(-1)).squeeze(-1)  # [B, T, 3]

            # Global position: p_global = p_parent + offset_global
            p_global = p_parent + offset_global
            positions[:, :, i] = p_global

        # Restore original shape
        if squeeze_time:
            positions = positions.squeeze(1)
            rotations = rotations.squeeze(1)

        if return_rotations:
            return positions, rotations
        return positions

    def compute_keypoints(
        self,
        joint_positions: torch.Tensor,
        joint_rotations: torch.Tensor,
        keypoint_names: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute virtual keypoint positions from joint positions.

        Args:
            joint_positions: [..., num_joints, 3] joint positions
            joint_rotations: [..., num_joints, 3, 3] joint rotation matrices
            keypoint_names: List of keypoint names to compute (None = all)

        Returns:
            Dictionary mapping keypoint name to position tensor [..., 3]
        """
        if keypoint_names is None:
            keypoint_names = list(self.config.keypoints.keys())

        keypoints = {}

        for kp_name in keypoint_names:
            if kp_name not in self.config.keypoints:
                raise ValueError(f"Keypoint '{kp_name}' not defined in robot config")

            kp_config = self.config.keypoints[kp_name]
            parent_idx = kp_config.parent_joint_idx

            # Get parent joint position and rotation
            p_parent = joint_positions[..., parent_idx, :]  # [..., 3]
            R_parent = joint_rotations[..., parent_idx, :, :]  # [..., 3, 3]

            # Transform offset to global frame
            offset = torch.from_numpy(kp_config.offset).to(joint_positions.device).float()
            offset_global = torch.matmul(R_parent, offset.unsqueeze(-1)).squeeze(-1)  # [..., 3]

            # Keypoint position
            kp_pos = p_parent + offset_global
            keypoints[kp_name] = kp_pos

        return keypoints

    def qpos_to_keypoints(
        self,
        qpos: torch.Tensor,
        keypoint_names: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience function: qpos -> keypoints.

        Args:
            qpos: [..., qpos_dim] joint configuration
            keypoint_names: List of keypoint names (None = all)

        Returns:
            Dictionary mapping keypoint name to position tensor
        """
        positions, rotations = self.forward_kinematics(qpos, return_rotations=True)
        return self.compute_keypoints(positions, rotations, keypoint_names)


def compute_foot_contact_mask(
    foot_positions: torch.Tensor,
    threshold: float = 0.02,
    fps: int = 30
) -> torch.Tensor:
    """
    Compute binary contact mask for feet based on velocity.

    Args:
        foot_positions: [B, T, num_feet, 3] foot positions over time
        threshold: Velocity threshold in m/s (below = contact)
        fps: Frames per second for velocity computation

    Returns:
        contact_mask: [B, T-1, num_feet] binary mask (1 = contact, 0 = no contact)
    """
    # Compute velocity: (pos[t+1] - pos[t]) * fps
    velocity = (foot_positions[:, 1:] - foot_positions[:, :-1]) * fps  # [B, T-1, num_feet, 3]

    # Velocity magnitude
    vel_magnitude = torch.norm(velocity, dim=-1)  # [B, T-1, num_feet]

    # Contact if velocity below threshold
    contact_mask = (vel_magnitude < threshold).float()

    return contact_mask


# Backward compatibility: create a default G1 FK instance
_default_g1_fk = None


def get_default_g1_fk(device: str = "cpu") -> HumanoidFK:
    """Get or create default G1 FK instance."""
    global _default_g1_fk
    if _default_g1_fk is None or _default_g1_fk.device != device:
        config = load_robot_config("g1")
        _default_g1_fk = HumanoidFK(config, device=device)
    return _default_g1_fk


def forward_kinematics_g1(qpos, **kwargs):
    """Backward compatibility wrapper for G1 FK."""
    device = qpos.device if torch.is_tensor(qpos) else "cpu"
    fk = get_default_g1_fk(str(device))
    return fk.forward_kinematics(qpos, **kwargs)


# Test
if __name__ == "__main__":
    print("=" * 80)
    print("Generic Humanoid FK Test")
    print("=" * 80)

    # Load G1 configuration
    config = load_robot_config("g1")
    fk = HumanoidFK(config, device="cpu")

    print(f"\nLoaded robot: {config.name}")
    print(f"Number of joints: {config.num_joints}")
    print(f"Number of keypoints: {len(config.keypoints)}")

    # Test FK
    print("\nTesting Forward Kinematics...")
    B, T = 2, 10
    qpos_dim = 7 + (config.num_joints - 1)  # root (7) + actuated joints
    qpos = torch.randn(B, T, qpos_dim)
    qpos[..., 3] = 1.0  # Set quaternion w=1
    qpos[..., 4:7] = 0.0  # Set quaternion xyz=0

    positions, rotations = fk.forward_kinematics(qpos, return_rotations=True)
    print(f"  ✓ Joint positions: {positions.shape}")
    print(f"  ✓ Joint rotations: {rotations.shape}")

    # Test keypoint computation
    print("\nTesting Keypoint Computation...")
    keypoints = fk.compute_keypoints(positions, rotations)
    for kp_name, kp_pos in keypoints.items():
        print(f"  ✓ {kp_name}: {kp_pos.shape}")

    # Test foot contact
    print("\nTesting Foot Contact Detection...")
    foot_kp_names = ["left_heel", "left_toe", "right_heel", "right_toe"]
    foot_positions = torch.stack([keypoints[name] for name in foot_kp_names], dim=2)
    print(f"  Foot positions: {foot_positions.shape}")

    contact_mask = compute_foot_contact_mask(foot_positions, threshold=0.02, fps=30)
    print(f"  ✓ Contact mask: {contact_mask.shape}")

    print("\nAll tests passed!")
