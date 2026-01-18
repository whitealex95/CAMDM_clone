"""
G1 Humanoid Forward Kinematics and Foot Contact Detection

This module provides batched PyTorch implementations for:
1. Forward Kinematics (FK) - Convert joint angles to 3D positions
2. Foot Contact Detection - Binary masks based on foot velocity

Joint Structure (30 total):
- Joint 0: Root (quaternion rotation)
- Joints 1-29: 1D revolute joints (stored in dim 0 of padded representation)

Important: The robot uses +X forward, +Y left, +Z up coordinate system.
"""

import torch
import numpy as np


# G1 Robot Skeleton Definition
# Based on g1_29dof_rev_1_0.xml and make_pose_data_g1.py

# Joint names (29 joints after root)
JOINT_NAMES = [
    "left_hip_pitch_joint",      # 1
    "left_hip_roll_joint",       # 2
    "left_hip_yaw_joint",        # 3
    "left_knee_joint",           # 4
    "left_ankle_pitch_joint",    # 5
    "left_ankle_roll_joint",     # 6
    "right_hip_pitch_joint",     # 7
    "right_hip_roll_joint",      # 8
    "right_hip_yaw_joint",       # 9
    "right_knee_joint",          # 10
    "right_ankle_pitch_joint",   # 11
    "right_ankle_roll_joint",    # 12
    "waist_yaw_joint",           # 13
    "waist_roll_joint",          # 14
    "waist_pitch_joint",         # 15
    "left_shoulder_pitch_joint", # 16
    "left_shoulder_roll_joint",  # 17
    "left_shoulder_yaw_joint",   # 18
    "left_elbow_joint",          # 19
    "left_wrist_roll_joint",     # 20
    "left_wrist_pitch_joint",    # 21
    "left_wrist_yaw_joint",      # 22
    "right_shoulder_pitch_joint",# 23
    "right_shoulder_roll_joint", # 24
    "right_shoulder_yaw_joint",  # 25
    "right_elbow_joint",         # 26
    "right_wrist_roll_joint",    # 27
    "right_wrist_pitch_joint",   # 28
    "right_wrist_yaw_joint",     # 29
]

# Parent indices (-1 for root, 0-indexed)
# Root is joint 0, joints connect as per MuJoCo body hierarchy
# Extracted from g1_29dof_rev_1_0.xml via extract_g1_skeleton.py
JOINT_PARENTS = np.array([
    -1,  # 0: root (pelvis) - no parent
    0,   # 1: left_hip_pitch
    1,   # 2: left_hip_roll
    2,   # 3: left_hip_yaw
    3,   # 4: left_knee
    4,   # 5: left_ankle_pitch
    5,   # 6: left_ankle_roll (LEFT FOOT END EFFECTOR)
    0,   # 7: right_hip_pitch
    7,   # 8: right_hip_roll
    8,   # 9: right_hip_yaw
    9,   # 10: right_knee
    10,  # 11: right_ankle_pitch
    11,  # 12: right_ankle_roll (RIGHT FOOT END EFFECTOR)
    0,   # 13: waist_yaw
    13,  # 14: waist_roll
    14,  # 15: waist_pitch (torso)
    15,  # 16: left_shoulder_pitch
    16,  # 17: left_shoulder_roll
    17,  # 18: left_shoulder_yaw
    18,  # 19: left_elbow
    19,  # 20: left_wrist_roll
    20,  # 21: left_wrist_pitch
    21,  # 22: left_wrist_yaw (left hand)
    15,  # 23: right_shoulder_pitch
    23,  # 24: right_shoulder_roll
    24,  # 25: right_shoulder_yaw
    25,  # 26: right_elbow
    26,  # 27: right_wrist_roll
    27,  # 28: right_wrist_pitch
    28,  # 29: right_wrist_yaw (right hand)
], dtype=np.int32)

# Joint offsets from parent (in parent's local frame)
# Extracted from g1_29dof_rev_1_0.xml via extract_g1_skeleton.py
JOINT_OFFSETS = np.array([
    [0.0, 0.0, 0.0],                      # 0: root (no offset)
    [0.0, 0.064452, -0.1027],             # 1: left_hip_pitch
    [0.0, 0.052, -0.030465],              # 2: left_hip_roll
    [0.025001, 0.0, -0.12412],            # 3: left_hip_yaw
    [-0.078273, 0.0021489, -0.17734],     # 4: left_knee
    [0.0, -9.4445e-05, -0.30001],         # 5: left_ankle_pitch
    [0.0, 0.0, -0.017558],                # 6: left_ankle_roll (foot)
    [0.0, -0.064452, -0.1027],            # 7: right_hip_pitch
    [0.0, -0.052, -0.030465],             # 8: right_hip_roll
    [0.025001, 0.0, -0.12412],            # 9: right_hip_yaw
    [-0.078273, -0.0021489, -0.17734],    # 10: right_knee
    [0.0, 9.4445e-05, -0.30001],          # 11: right_ankle_pitch
    [0.0, 0.0, -0.017558],                # 12: right_ankle_roll (foot)
    [0.0, 0.0, 0.0],                      # 13: waist_yaw (at pelvis)
    [-0.0039635, 0.0, 0.044],             # 14: waist_roll
    [0.0, 0.0, 0.0],                      # 15: waist_pitch (torso)
    [0.0039563, 0.10022, 0.24778],        # 16: left_shoulder_pitch
    [0.0, 0.038, -0.013831],              # 17: left_shoulder_roll
    [0.0, 0.00624, -0.1032],              # 18: left_shoulder_yaw
    [0.015783, 0.0, -0.080518],           # 19: left_elbow
    [0.1, 0.00188791, -0.01],             # 20: left_wrist_roll
    [0.038, 0.0, 0.0],                    # 21: left_wrist_pitch
    [0.046, 0.0, 0.0],                    # 22: left_wrist_yaw
    [0.0039563, -0.10021, 0.24778],       # 23: right_shoulder_pitch
    [0.0, -0.038, -0.013831],             # 24: right_shoulder_roll
    [0.0, -0.00624, -0.1032],             # 25: right_shoulder_yaw
    [0.015783, 0.0, -0.080518],           # 26: right_elbow
    [0.1, -0.00188791, -0.01],            # 27: right_wrist_roll
    [0.038, 0.0, 0.0],                    # 28: right_wrist_pitch
    [0.046, 0.0, 0.0],                    # 29: right_wrist_yaw
], dtype=np.float32)

# Joint axes (rotation axis in parent's local frame)
# Based on XML joint definitions
JOINT_AXES = np.array([
    [0, 0, 0],    # 0: root (free joint, not used)
    [0, 1, 0],    # 1: left_hip_pitch (Y-axis)
    [1, 0, 0],    # 2: left_hip_roll (X-axis)
    [0, 0, 1],    # 3: left_hip_yaw (Z-axis)
    [0, 1, 0],    # 4: left_knee (Y-axis)
    [0, 1, 0],    # 5: left_ankle_pitch (Y-axis)
    [1, 0, 0],    # 6: left_ankle_roll (X-axis)
    [0, 1, 0],    # 7: right_hip_pitch (Y-axis)
    [1, 0, 0],    # 8: right_hip_roll (X-axis)
    [0, 0, 1],    # 9: right_hip_yaw (Z-axis)
    [0, 1, 0],    # 10: right_knee (Y-axis)
    [0, 1, 0],    # 11: right_ankle_pitch (Y-axis)
    [1, 0, 0],    # 12: right_ankle_roll (X-axis)
    [0, 0, 1],    # 13: waist_yaw (Z-axis)
    [1, 0, 0],    # 14: waist_roll (X-axis)
    [0, 1, 0],    # 15: waist_pitch (Y-axis)
    [0, 1, 0],    # 16: left_shoulder_pitch (Y-axis)
    [1, 0, 0],    # 17: left_shoulder_roll (X-axis)
    [0, 0, 1],    # 18: left_shoulder_yaw (Z-axis)
    [0, 1, 0],    # 19: left_elbow (Y-axis)
    [1, 0, 0],    # 20: left_wrist_roll (X-axis)
    [0, 1, 0],    # 21: left_wrist_pitch (Y-axis)
    [0, 0, 1],    # 22: left_wrist_yaw (Z-axis)
    [0, 1, 0],    # 23: right_shoulder_pitch (Y-axis)
    [1, 0, 0],    # 24: right_shoulder_roll (X-axis)
    [0, 0, 1],    # 25: right_shoulder_yaw (Z-axis)
    [0, 1, 0],    # 26: right_elbow (Y-axis)
    [1, 0, 0],    # 27: right_wrist_roll (X-axis)
    [0, 1, 0],    # 28: right_wrist_pitch (Y-axis)
    [0, 0, 1],    # 29: right_wrist_yaw (Z-axis)
], dtype=np.float32)

# Foot joint indices
LEFT_FOOT_IDX = 6   # left_ankle_roll_link
RIGHT_FOOT_IDX = 12  # right_ankle_roll_link

# Initial body quaternion rotations (from XML)
# Some bodies have non-identity initial orientations
BODY_INIT_QUATS = np.array([
    [1, 0, 0, 0],                             # 0: root
    [1, 0, 0, 0],                             # 1: left_hip_pitch
    [0.996179, 0, -0.0873386, 0],             # 2: left_hip_roll
    [1, 0, 0, 0],                             # 3: left_hip_yaw
    [0.996179, 0, 0.0873386, 0],              # 4: left_knee
    [1, 0, 0, 0],                             # 5: left_ankle_pitch
    [1, 0, 0, 0],                             # 6: left_ankle_roll
    [1, 0, 0, 0],                             # 7: right_hip_pitch
    [0.996179, 0, -0.0873386, 0],             # 8: right_hip_roll
    [1, 0, 0, 0],                             # 9: right_hip_yaw
    [0.996179, 0, 0.0873386, 0],              # 10: right_knee
    [1, 0, 0, 0],                             # 11: right_ankle_pitch
    [1, 0, 0, 0],                             # 12: right_ankle_roll
    [1, 0, 0, 0],                             # 13: waist_yaw
    [1, 0, 0, 0],                             # 14: waist_roll
    [1, 0, 0, 0],                             # 15: waist_pitch
    [0.990264, 0.139201, 0, 0],               # 16: left_shoulder_pitch (approx)
    [0.990268, -0.139172, 0, 0],              # 17: left_shoulder_roll
    [1, 0, 0, 0],                             # 18: left_shoulder_yaw
    [1, 0, 0, 0],                             # 19: left_elbow
    [1, 0, 0, 0],                             # 20: left_wrist_roll
    [1, 0, 0, 0],                             # 21: left_wrist_pitch
    [1, 0, 0, 0],                             # 22: left_wrist_yaw
    [0.990264, -0.139201, 0, 0],              # 23: right_shoulder_pitch (approx)
    [0.990268, 0.139172, 0, 0],               # 24: right_shoulder_roll
    [1, 0, 0, 0],                             # 25: right_shoulder_yaw
    [1, 0, 0, 0],                             # 26: right_elbow
    [1, 0, 0, 0],                             # 27: right_wrist_roll
    [1, 0, 0, 0],                             # 28: right_wrist_pitch
    [1, 0, 0, 0],                             # 29: right_wrist_yaw
], dtype=np.float32)


def quat_to_rotmat(quat):
    """
    Convert quaternions (wxyz) to rotation matrices.

    Args:
        quat: [..., 4] quaternions in wxyz format

    Returns:
        [..., 3, 3] rotation matrices
    """
    # Normalize
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    w, x, y, z = quat.unbind(-1)

    # Build rotation matrix
    r_xx = 1.0 - 2.0 * (y**2 + z**2)
    r_xy = 2.0 * (x*y - z*w)
    r_xz = 2.0 * (x*z + y*w)
    r_yx = 2.0 * (x*y + z*w)
    r_yy = 1.0 - 2.0 * (x**2 + z**2)
    r_yz = 2.0 * (y*z - x*w)
    r_zx = 2.0 * (x*z - y*w)
    r_zy = 2.0 * (y*z + x*w)
    r_zz = 1.0 - 2.0 * (x**2 + y**2)

    rot_matrix = torch.stack([
        r_xx, r_xy, r_xz,
        r_yx, r_yy, r_yz,
        r_zx, r_zy, r_zz
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
        # Expand to match angle shape
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

    # Build rotation matrix using Rodrigues' formula components
    # R = I + sin(θ)[axis]_x + (1-cos(θ))[axis]_x^2
    # Expanded form:
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


def forward_kinematics_g1(qpos, parents=None, offsets=None, axes=None, init_quats=None, return_rotations=False):
    """
    Batched Forward Kinematics for G1 humanoid robot.

    Converts joint configuration (root pose + joint angles) to 3D positions of all joints.

    Args:
        qpos: [B, T, 36] or [B, 36] or [T, 36] tensor
            - qpos[..., 0:3]: root position (XYZ)
            - qpos[..., 3:7]: root quaternion (WXYZ)
            - qpos[..., 7:36]: 29 joint angles (1D revolute joints)
        parents: [30] parent indices (optional, uses default if None)
        offsets: [30, 3] joint offsets (optional, uses default if None)
        axes: [30, 3] joint rotation axes (optional, uses default if None)
        init_quats: [30, 4] initial body orientations (optional, uses default if None)
        return_rotations: If True, return (positions, rotations) tuple

    Returns:
        positions: [..., 30, 3] global 3D positions of all joints
        rotations (optional): [..., 30, 3, 3] global rotation matrices
    """
    # Handle different input shapes
    original_shape = qpos.shape
    if len(qpos.shape) == 2:  # [B, 36] or [T, 36]
        qpos = qpos.unsqueeze(1)  # [B, 1, 36]
        squeeze_time = True
    elif len(qpos.shape) == 3:  # [B, T, 36]
        squeeze_time = False
    else:
        raise ValueError(f"qpos must be 2D or 3D, got shape {qpos.shape}")

    B, T, _ = qpos.shape
    device = qpos.device

    # Load skeleton data
    if parents is None:
        parents = torch.from_numpy(JOINT_PARENTS).to(device)
    if offsets is None:
        offsets = torch.from_numpy(JOINT_OFFSETS).to(device).float()
    if axes is None:
        axes = torch.from_numpy(JOINT_AXES).to(device).float()
    if init_quats is None:
        init_quats = torch.from_numpy(BODY_INIT_QUATS).to(device).float()

    # Parse qpos
    root_pos = qpos[..., :3]  # [B, T, 3]
    root_quat = qpos[..., 3:7]  # [B, T, 4] (wxyz)
    joint_angles = qpos[..., 7:]  # [B, T, 29]

    # Prepare output
    num_joints = 30  # 1 root + 29 joints
    positions = torch.zeros(B, T, num_joints, 3, device=device, dtype=qpos.dtype)
    rotations = torch.zeros(B, T, num_joints, 3, 3, device=device, dtype=qpos.dtype)

    # Initialize root
    positions[:, :, 0] = root_pos
    rotations[:, :, 0] = quat_to_rotmat(root_quat)

    # Forward pass through kinematic chain
    for i in range(1, num_joints):
        parent_idx = parents[i].item()

        # Get parent's global rotation and position
        R_parent = rotations[:, :, parent_idx].clone()  # [B, T, 3, 3]
        p_parent = positions[:, :, parent_idx].clone()  # [B, T, 3]

        # Get initial body orientation (from XML quat field)
        init_quat = init_quats[i]  # [4] wxyz
        R_init = quat_to_rotmat(init_quat)  # [3, 3]
        # Expand to match batch and time dimensions
        R_init = R_init.unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3)  # [B, T, 3, 3]

        # Local rotation for this joint (1D revolute joint around axis)
        angle = joint_angles[:, :, i-1]  # [B, T]
        axis = axes[i]  # [3]
        R_joint = axis_angle_to_rotmat(axis, angle)  # [B, T, 3, 3]

        # Total local rotation: R_local = R_init @ R_joint
        # First apply initial orientation, then joint rotation
        R_local = torch.matmul(R_init, R_joint)  # [B, T, 3, 3]

        # Global rotation: R_global = R_parent @ R_local
        R_global = torch.matmul(R_parent, R_local)  # [B, T, 3, 3]
        rotations[:, :, i] = R_global

        # Local offset in parent's frame
        offset = offsets[i]  # [3]

        # Transform offset to global frame: offset_global = R_parent @ offset
        offset_global = torch.matmul(R_parent, offset.unsqueeze(-1)).squeeze(-1)  # [B, T, 3]

        # Global position: p_global = p_parent + offset_global
        positions[:, :, i] = p_parent + offset_global

    # Remove time dimension if it was added
    if squeeze_time:
        positions = positions.squeeze(1)  # [B, 30, 3]
        if return_rotations:
            rotations = rotations.squeeze(1)  # [B, 30, 3, 3]

    if return_rotations:
        return positions, rotations
    return positions


def compute_foot_contact_mask(foot_positions, threshold=0.01, fps=30):
    """
    Compute binary foot contact mask based on foot velocity.

    A foot is considered in contact (mask=1) when its velocity is below threshold.

    Args:
        foot_positions: [B, T, 2, 3] positions of left and right feet
            - foot_positions[..., 0, :]: left foot
            - foot_positions[..., 1, :]: right foot
        threshold: velocity threshold in m/s (default: 0.01)
        fps: frames per second (default: 30)

    Returns:
        contact_mask: [B, T-1, 2] binary mask (1 = contact, 0 = no contact)
    """
    # Compute velocity: (position[t+1] - position[t]) * fps
    velocity = (foot_positions[:, 1:] - foot_positions[:, :-1]) * fps  # [B, T-1, 2, 3]

    # Compute velocity magnitude
    vel_magnitude = torch.norm(velocity, dim=-1)  # [B, T-1, 2]

    # Binary mask: 1 if velocity < threshold, else 0
    contact_mask = (vel_magnitude <= threshold).float()  # [B, T-1, 2]

    return contact_mask


def extract_foot_positions(joint_positions, left_idx=LEFT_FOOT_IDX, right_idx=RIGHT_FOOT_IDX):
    """
    Extract left and right foot positions from full joint positions.

    Args:
        joint_positions: [B, T, 30, 3] or [B, 30, 3] all joint positions
        left_idx: index of left foot joint (default: 6)
        right_idx: index of right foot joint (default: 12)

    Returns:
        foot_positions: [B, T, 2, 3] or [B, 2, 3] stacked foot positions
    """
    left_foot = joint_positions[..., left_idx, :]
    right_foot = joint_positions[..., right_idx, :]

    # Stack along second-to-last dimension
    foot_positions = torch.stack([left_foot, right_foot], dim=-2)

    return foot_positions


# ==============================================================================
# Convenience Functions
# ==============================================================================

def qpos_to_foot_positions(qpos):
    """
    Convert qpos directly to foot positions.

    Args:
        qpos: [B, T, 36] or [B, 36] joint configuration

    Returns:
        foot_positions: [B, T, 2, 3] or [B, 2, 3] left and right foot positions
    """
    joint_positions = forward_kinematics_g1(qpos)
    foot_positions = extract_foot_positions(joint_positions)
    return foot_positions


def qpos_to_foot_contact(qpos, threshold=0.01, fps=30):
    """
    Convert qpos directly to foot contact mask.

    Args:
        qpos: [B, T, 36] joint configuration (requires T >= 2)
        threshold: velocity threshold in m/s
        fps: frames per second

    Returns:
        contact_mask: [B, T-1, 2] binary foot contact mask
    """
    foot_positions = qpos_to_foot_positions(qpos)
    contact_mask = compute_foot_contact_mask(foot_positions, threshold, fps)
    return contact_mask


# ==============================================================================
# Test Functions
# ==============================================================================

def test_forward_kinematics():
    """Test FK with random inputs."""
    print("Testing Forward Kinematics...")

    # Test case 1: Single pose
    qpos = torch.randn(1, 36)
    positions = forward_kinematics_g1(qpos)
    assert positions.shape == (1, 30, 3), f"Expected (1, 30, 3), got {positions.shape}"
    print(f"✓ Single pose: {positions.shape}")

    # Test case 2: Batch of poses
    qpos = torch.randn(16, 36)
    positions = forward_kinematics_g1(qpos)
    assert positions.shape == (16, 30, 3), f"Expected (16, 30, 3), got {positions.shape}"
    print(f"✓ Batch of poses: {positions.shape}")

    # Test case 3: Sequence of poses
    qpos = torch.randn(8, 50, 36)
    positions = forward_kinematics_g1(qpos)
    assert positions.shape == (8, 50, 30, 3), f"Expected (8, 50, 30, 3), got {positions.shape}"
    print(f"✓ Sequence of poses: {positions.shape}")

    # Test case 4: Check gradient flow
    qpos = torch.randn(2, 10, 36, requires_grad=True)
    positions = forward_kinematics_g1(qpos)
    loss = positions.sum()
    loss.backward()
    assert qpos.grad is not None, "Gradient not computed"
    print(f"✓ Gradient flow: grad shape {qpos.grad.shape}")

    print("Forward Kinematics tests passed!\n")


def test_foot_contact():
    """Test foot contact detection."""
    print("Testing Foot Contact Detection...")

    # Create dummy motion with static and moving phases
    B, T = 4, 100
    qpos = torch.zeros(B, T, 36)

    # Add some root motion
    qpos[:, :, 0] = torch.linspace(0, 1, T).unsqueeze(0).expand(B, T)  # Move forward
    qpos[:, :, 3] = 1.0  # Root quaternion w=1

    # Get foot positions
    foot_positions = qpos_to_foot_positions(qpos)
    assert foot_positions.shape == (B, T, 2, 3), f"Expected (B, T, 2, 3), got {foot_positions.shape}"
    print(f"✓ Foot positions: {foot_positions.shape}")

    # Get contact mask
    contact_mask = compute_foot_contact_mask(foot_positions)
    assert contact_mask.shape == (B, T-1, 2), f"Expected (B, T-1, 2), got {contact_mask.shape}"
    assert contact_mask.min() >= 0 and contact_mask.max() <= 1, "Mask values must be in [0, 1]"
    print(f"✓ Contact mask: {contact_mask.shape}")
    print(f"  Contact ratio: {contact_mask.mean().item():.2%}")

    # Test convenience function
    contact_mask2 = qpos_to_foot_contact(qpos)
    assert torch.allclose(contact_mask, contact_mask2), "Convenience function mismatch"
    print(f"✓ Convenience function matches")

    print("Foot Contact Detection tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("G1 Kinematics Test Suite")
    print("=" * 60)
    print()

    test_forward_kinematics()
    test_foot_contact()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
