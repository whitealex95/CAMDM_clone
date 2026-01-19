"""
G1 Humanoid Forward Kinematics and Foot Contact Detection

This module provides batched PyTorch implementations for:
1. Forward Kinematics (FK) - Convert joint angles to 3D positions
2. Foot Contact Detection - Binary masks based on foot velocity and height
3. Keypoint computation - Virtual points (heel, toe, hands, head)

Joint Structure (30 total):
- Joint 0: Root (quaternion rotation)
- Joints 1-29: 1D revolute joints

Important: The robot uses +X forward, +Y left, +Z up coordinate system.
All skeleton parameters are loaded from robot config (auto-extracted from MuJoCo XML).
"""

import torch
import numpy as np


class G1Kinematics:
    """
    G1 Robot kinematics with skeleton and keypoint information.

    Loads all parameters from robot config (auto-extracted from MuJoCo XML).
    """

    def __init__(self):
        """Initialize by loading robot configuration."""
        try:
            from utils.robot_config import load_robot_config
        except ModuleNotFoundError:
            # For standalone script execution
            from robot_config import load_robot_config
        self.config = load_robot_config("g1")

        # Skeleton data (numpy arrays from config)
        self.parents = self.config.joint_parents
        self.offsets = self.config.joint_offsets
        self.axes = self.config.joint_axes
        self.init_quats = self.config.body_init_quats
        self.num_joints = self.config.num_joints

        # Keypoints
        self.keypoints = self.config.keypoints

    def forward_kinematics(self, qpos, return_rotations=False):
        """
        Batched Forward Kinematics for G1 humanoid robot.

        Args:
            qpos: [B, T, 36] or [B, 36] or [T, 36] tensor
                - qpos[..., 0:3]: root position (XYZ)
                - qpos[..., 3:7]: root quaternion (WXYZ)
                - qpos[..., 7:36]: 29 joint angles
            return_rotations: If True, return (positions, rotations) tuple

        Returns:
            positions: [..., 30, 3] global 3D positions of all joints
            rotations (optional): [..., 30, 3, 3] global rotation matrices
        """
        # Handle different input shapes
        original_shape = qpos.shape
        if len(qpos.shape) == 2:
            qpos = qpos.unsqueeze(1)
            squeeze_time = True
        elif len(qpos.shape) == 3:
            squeeze_time = False
        else:
            raise ValueError(f"qpos must be 2D or 3D, got shape {qpos.shape}")

        B, T, _ = qpos.shape
        device = qpos.device

        # Convert skeleton data to torch tensors
        parents = torch.from_numpy(self.parents).to(device)
        offsets = torch.from_numpy(self.offsets).to(device).float()
        axes = torch.from_numpy(self.axes).to(device).float()
        init_quats = torch.from_numpy(self.init_quats).to(device).float()

        # Parse qpos
        root_pos = qpos[..., :3]  # [B, T, 3]
        root_quat = qpos[..., 3:7]  # [B, T, 4] (wxyz)
        joint_angles = qpos[..., 7:]  # [B, T, 29]

        # Prepare output
        positions = torch.zeros(B, T, self.num_joints, 3, device=device, dtype=qpos.dtype)
        rotations = torch.zeros(B, T, self.num_joints, 3, 3, device=device, dtype=qpos.dtype)

        # Initialize root
        positions[:, :, 0] = root_pos
        rotations[:, :, 0] = quat_to_rotmat(root_quat)

        # Forward pass through kinematic chain
        for i in range(1, self.num_joints):
            parent_idx = parents[i].item()

            # Get parent's global rotation and position
            R_parent = rotations[:, :, parent_idx].clone()
            p_parent = positions[:, :, parent_idx].clone()

            # Get initial body orientation
            init_quat = init_quats[i]
            R_init = quat_to_rotmat(init_quat)
            R_init = R_init.unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3)

            # Local rotation for this joint
            angle = joint_angles[:, :, i-1]
            axis = axes[i]
            R_joint = axis_angle_to_rotmat(axis, angle)

            # Total local rotation
            R_local = torch.matmul(R_init, R_joint)

            # Global rotation
            R_global = torch.matmul(R_parent, R_local)
            rotations[:, :, i] = R_global

            # Transform offset to global frame
            offset = offsets[i]
            offset_global = torch.matmul(R_parent, offset.unsqueeze(-1)).squeeze(-1)

            # Global position
            positions[:, :, i] = p_parent + offset_global

        # Remove time dimension if it was added
        if squeeze_time:
            positions = positions.squeeze(1)
            if return_rotations:
                rotations = rotations.squeeze(1)

        if return_rotations:
            return positions, rotations
        return positions

    def compute_keypoints(self, joint_positions, joint_rotations, keypoint_names=None):
        """
        Compute virtual keypoint positions from joint FK results.

        Args:
            joint_positions: [..., num_joints, 3] joint positions from FK
            joint_rotations: [..., num_joints, 3, 3] joint rotation matrices
            keypoint_names: List of keypoint names (None = all)

        Returns:
            Dictionary mapping keypoint name to position tensor [..., 3]
        """
        if keypoint_names is None:
            keypoint_names = list(self.keypoints.keys())

        keypoints_out = {}
        device = joint_positions.device

        for kp_name in keypoint_names:
            if kp_name not in self.keypoints:
                continue

            kp_config = self.keypoints[kp_name]
            parent_idx = kp_config.parent_joint_idx

            # Get parent joint position and rotation
            p_parent = joint_positions[..., parent_idx, :]
            R_parent = joint_rotations[..., parent_idx, :, :]

            # Transform offset to global frame
            offset = torch.from_numpy(kp_config.offset).to(device).float()
            offset_global = torch.matmul(R_parent, offset.unsqueeze(-1)).squeeze(-1)

            # Keypoint position
            kp_pos = p_parent + offset_global
            keypoints_out[kp_name] = kp_pos

        return keypoints_out

    def forward_kinematics_with_keypoints(self, qpos, keypoint_names=None):
        """
        Convenience function: FK + keypoints in one call.

        Args:
            qpos: [..., qpos_dim] joint configuration
            keypoint_names: List of keypoint names (None = all)

        Returns:
            joint_positions: [..., num_joints, 3]
            keypoints: Dict[str, Tensor] keypoint positions
        """
        positions, rotations = self.forward_kinematics(qpos, return_rotations=True)
        keypoints = self.compute_keypoints(positions, rotations, keypoint_names)
        return positions, keypoints

    def compute_foot_contact(self, qpos, vel_threshold=None, height_threshold=None, fps=30):
        """
        Compute foot contact mask from qpos using velocity and/or height thresholds.

        Args:
            qpos: [B, T, 36] or [T, 36] joint configuration over time
            vel_threshold: Velocity threshold in m/s (None = use config default)
            height_threshold: Height threshold in m (None = use config default)
            fps: Frames per second for velocity computation

        Returns:
            contact_mask: [..., T-1, 2] binary mask (1 = contact) for [left_foot, right_foot]
        """
        # Use config defaults if not specified
        if vel_threshold is None:
            vel_threshold = self.config.contact_velocity_threshold
        if height_threshold is None:
            height_threshold = self.config.contact_height_threshold

        # Handle different input shapes
        original_ndim = qpos.ndim
        if qpos.ndim == 2:
            qpos = qpos.unsqueeze(0)  # [T, 36] -> [1, T, 36]

        # Compute FK to get joint positions
        joint_positions = self.forward_kinematics(qpos)  # [B, T, 30, 3]

        # Extract foot positions (left_ankle_roll=6, right_ankle_roll=12)
        left_foot = joint_positions[..., 6, :]   # [B, T, 3]
        right_foot = joint_positions[..., 12, :]  # [B, T, 3]
        foot_positions = torch.stack([left_foot, right_foot], dim=-2)  # [B, T, 2, 3]

        # Initialize contact mask
        B, T = foot_positions.shape[0], foot_positions.shape[1]
        contact_mask = torch.ones(B, T-1, 2, device=qpos.device, dtype=torch.float32)

        # Apply velocity threshold if specified
        if vel_threshold is not None and vel_threshold > 0:
            velocity = (foot_positions[:, 1:] - foot_positions[:, :-1]) * fps  # [B, T-1, 2, 3]
            vel_magnitude = torch.norm(velocity, dim=-1)  # [B, T-1, 2]
            vel_contact = (vel_magnitude < vel_threshold).float()
            contact_mask = contact_mask * vel_contact

        # Apply height threshold if specified
        if height_threshold is not None and height_threshold > 0:
            foot_heights = foot_positions[:, 1:, :, 2]  # [B, T-1, 2] (z coordinate)
            height_contact = (foot_heights < height_threshold).float()
            contact_mask = contact_mask * height_contact

        # Remove batch dimension if input was 2D
        if original_ndim == 2:
            contact_mask = contact_mask.squeeze(0)

        return contact_mask


# Helper functions (same as before)
def quat_to_rotmat(quat):
    """Convert quaternions (wxyz) to rotation matrices."""
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    rot_matrix = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)

    return rot_matrix


def axis_angle_to_rotmat(axis, angle):
    """Convert axis-angle to rotation matrix using Rodrigues' formula."""
    if axis.dim() == 1:
        axis = axis.expand(*angle.shape, 3)

    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    r00 = cos_angle + x*x*one_minus_cos
    r01 = x*y*one_minus_cos - z*sin_angle
    r02 = x*z*one_minus_cos + y*sin_angle
    r10 = y*x*one_minus_cos + z*sin_angle
    r11 = cos_angle + y*y*one_minus_cos
    r12 = y*z*one_minus_cos - x*sin_angle
    r20 = z*x*one_minus_cos - y*sin_angle
    r21 = z*y*one_minus_cos + x*sin_angle
    r22 = cos_angle + z*z*one_minus_cos

    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).reshape(*angle.shape, 3, 3)

    return R




# Test
if __name__ == "__main__":
    print("=" * 80)
    print("G1 Kinematics Test")
    print("=" * 80)

    g1_kin = G1Kinematics()
    print(f"\nRobot: {g1_kin.config.name}")
    print(f"Joints: {g1_kin.num_joints}")
    print(f"Keypoints: {list(g1_kin.keypoints.keys())}")

    # Test FK
    B, T = 2, 10
    qpos = torch.randn(B, T, 36)
    qpos[..., 3] = 1.0  # w=1
    qpos[..., 4:7] = 0.0
    qpos[..., 2] = 0.75  # height

    positions, rotations = g1_kin.forward_kinematics(qpos, return_rotations=True)
    print(f"\n✓ FK: positions {positions.shape}, rotations {rotations.shape}")

    # Test keypoints
    keypoints = g1_kin.compute_keypoints(positions, rotations)
    print(f"✓ Keypoints:")
    for name, pos in keypoints.items():
        print(f"  - {name}: {pos.shape}")

    # Test convenience function
    joint_pos, kps = g1_kin.forward_kinematics_with_keypoints(qpos)
    print(f"\n✓ Combined FK+keypoints: {len(kps)} keypoints")

    # Test foot contact
    contact_mask = g1_kin.compute_foot_contact(qpos)
    print(f"\n✓ Foot contact: {contact_mask.shape}")
    print(f"  Contact ratio: L={contact_mask[..., 0].mean():.1%}, R={contact_mask[..., 1].mean():.1%}")

    print("\n" + "=" * 80)
    print("All tests passed!")
