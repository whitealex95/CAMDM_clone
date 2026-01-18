"""
Keypoint Computation Module

Provides config-based virtual keypoint computation on top of existing FK.
This module extends g1_kinematics.py with configurable keypoints.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from utils.robot_config import RobotConfig, load_robot_config
import utils.g1_kinematics as g1_kin


class KeypointComputer:
    """
    Computes virtual keypoints from joint positions using robot configuration.
    """

    def __init__(self, robot_config: RobotConfig):
        """
        Args:
            robot_config: Robot configuration with keypoint definitions
        """
        self.config = robot_config

    def compute_keypoints(
        self,
        joint_positions: torch.Tensor,
        joint_rotations: torch.Tensor,
        keypoint_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute virtual keypoint positions from joint FK results.

        Args:
            joint_positions: [... num_joints, 3] joint positions from FK
            joint_rotations: [..., num_joints, 3, 3] joint rotation matrices from FK
            keypoint_names: List of keypoint names to compute (None = all)

        Returns:
            Dictionary mapping keypoint name to position tensor [..., 3]
        """
        if keypoint_names is None:
            keypoint_names = list(self.config.keypoints.keys())

        keypoints = {}
        device = joint_positions.device

        for kp_name in keypoint_names:
            if kp_name not in self.config.keypoints:
                continue

            kp_config = self.config.keypoints[kp_name]
            parent_idx = kp_config.parent_joint_idx

            # Get parent joint position and rotation
            p_parent = joint_positions[..., parent_idx, :]  # [..., 3]
            R_parent = joint_rotations[..., parent_idx, :, :]  # [..., 3, 3]

            # Transform offset to global frame
            offset = torch.from_numpy(kp_config.offset).to(device).float()
            offset_global = torch.matmul(R_parent, offset.unsqueeze(-1)).squeeze(-1)  # [..., 3]

            # Keypoint position
            kp_pos = p_parent + offset_global
            keypoints[kp_name] = kp_pos

        return keypoints

    def qpos_to_keypoints(
        self,
        qpos: torch.Tensor,
        keypoint_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience function: compute keypoints directly from qpos.

        Args:
            qpos: [..., qpos_dim] joint configuration
            keypoint_names: List of keypoint names (None = all)

        Returns:
            Dictionary mapping keypoint name to position tensor
        """
        # Use existing g1_kinematics FK with rotation output
        positions, rotations = g1_kin.forward_kinematics_g1(qpos, return_rotations=True)

        return self.compute_keypoints(positions, rotations, keypoint_names)


# Global instance for backward compatibility
_default_keypoint_computer = None


def get_default_keypoint_computer() -> KeypointComputer:
    """Get or create default keypoint computer for G1."""
    global _default_keypoint_computer
    if _default_keypoint_computer is None:
        config = load_robot_config("g1")
        _default_keypoint_computer = KeypointComputer(config)
    return _default_keypoint_computer


def compute_keypoints_from_qpos(
    qpos: torch.Tensor,
    keypoint_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute keypoints from qpos using default G1 config.

    Args:
        qpos: [..., qpos_dim] joint configuration
        keypoint_names: List of keypoint names (None = all)

    Returns:
        Dictionary of keypoint positions
    """
    computer = get_default_keypoint_computer()
    return computer.qpos_to_keypoints(qpos, keypoint_names)


def get_foot_keypoint_positions(qpos: torch.Tensor) -> torch.Tensor:
    """
    Get foot keypoint positions for contact detection.

    Args:
        qpos: [..., qpos_dim] joint configuration

    Returns:
        foot_positions: [..., num_foot_keypoints, 3]
            Order: left_heel, left_toe, right_heel, right_toe
    """
    foot_names = ["left_heel", "left_toe", "right_heel", "right_toe"]
    keypoints = compute_keypoints_from_qpos(qpos, foot_names)

    # Stack in consistent order
    return torch.stack([keypoints[name] for name in foot_names], dim=-2)


def get_hand_keypoint_positions(qpos: torch.Tensor) -> torch.Tensor:
    """
    Get hand keypoint positions.

    Args:
        qpos: [..., qpos_dim] joint configuration

    Returns:
        hand_positions: [..., 2, 3]
            Order: left_hand, right_hand
    """
    hand_names = ["left_hand", "right_hand"]
    keypoints = compute_keypoints_from_qpos(qpos, hand_names)

    return torch.stack([keypoints[name] for name in hand_names], dim=-2)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Keypoint Computation Test")
    print("=" * 80)

    # Test with random qpos
    B, T = 2, 10
    qpos = torch.randn(B, T, 36)
    qpos[..., 3] = 1.0  # Set quaternion w=1
    qpos[..., 4:7] = 0.0

    # Compute all keypoints
    print("\nComputing all keypoints...")
    keypoints = compute_keypoints_from_qpos(qpos)
    for name, pos in keypoints.items():
        print(f"  {name}: {pos.shape}")

    # Get foot keypoints
    print("\nGetting foot keypoints...")
    foot_kps = get_foot_keypoint_positions(qpos)
    print(f"  Foot keypoints: {foot_kps.shape}")

    # Get hand keypoints
    print("\nGetting hand keypoints...")
    hand_kps = get_hand_keypoint_positions(qpos)
    print(f"  Hand keypoints: {hand_kps.shape}")

    print("\n✓ All tests passed!")
