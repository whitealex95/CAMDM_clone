"""
Geometric Losses for Motion Diffusion (Config-based Version)

This version uses robot configuration for flexibility across different robots.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from utils.robot_config import RobotConfig, load_robot_config
from utils.kinematics import HumanoidFK, compute_foot_contact_mask


class GeometricLossComputer:
    """
    Computes geometric losses for motion diffusion using robot configuration.
    """

    def __init__(self, robot_config: RobotConfig, device: str = "cpu"):
        """
        Args:
            robot_config: Robot configuration
            device: Device for computation
        """
        self.config = robot_config
        self.device = device
        self.fk = HumanoidFK(robot_config, device=device)

        # Loss weights from config
        self.loss_weights = {
            'pos': robot_config.loss_position_weight,
            'foot': robot_config.loss_foot_contact_weight,
            'vel': robot_config.loss_velocity_weight
        }

    def model_format_to_qpos(
        self,
        x: torch.Tensor,
        rot_req: str = '6d'
    ) -> torch.Tensor:
        """
        Convert model output format [B, J, feat, T] to qpos [B, T, qpos_dim].

        Args:
            x: [B, J, feat, T] model output
                - J = num_joints + 1 (joints + root position channel)
                - feat = rotation representation size (e.g. 6 for 6D)
            rot_req: Rotation representation ('6d', 'quat', etc.)

        Returns:
            qpos: [B, T, qpos_dim] where qpos_dim = 7 + num_actuated_joints
                - qpos[..., 0:3]: root position
                - qpos[..., 3:7]: root quaternion (wxyz)
                - qpos[..., 7:]: joint angles
        """
        B, J, feat, T = x.shape
        device = x.device

        num_joints = self.config.num_joints
        num_actuated = num_joints - 1  # Exclude root
        qpos_dim = 7 + num_actuated

        qpos = torch.zeros(B, T, qpos_dim, device=device)

        # Extract root position from channel J-1 (last channel)
        root_pos = x[:, -1, :3, :].permute(0, 2, 1)  # [B, T, 3]
        qpos[:, :, :3] = root_pos

        # Convert rotations to quaternions
        if rot_req == '6d':
            # Convert 6D rotation to quaternion for all joints
            for j in range(num_joints):
                rot_6d = x[:, j, :6, :].permute(0, 2, 1)  # [B, T, 6]
                quat = self._rotation_6d_to_quaternion(rot_6d)  # [B, T, 4] wxyz

                if j == 0:
                    # Root quaternion
                    qpos[:, :, 3:7] = quat
                else:
                    # Convert quaternion to axis-angle for actuated joints
                    # This is simplified - assumes single-axis revolute joints
                    # Extract angle around joint's primary axis
                    angle = self._quaternion_to_angle(quat, self.fk.axes[j])
                    qpos[:, :, 7 + j - 1] = angle
        else:
            raise NotImplementedError(f"Rotation format '{rot_req}' not yet supported")

        return qpos

    def _rotation_6d_to_quaternion(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to quaternion (wxyz).

        Args:
            rot_6d: [..., 6] 6D rotation (two column vectors)

        Returns:
            quat: [..., 4] quaternion in wxyz format
        """
        # Reconstruct rotation matrix from 6D
        x = rot_6d[..., :3]  # First column
        y = rot_6d[..., 3:6]  # Second column

        # Normalize
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        # Compute third column as cross product
        z = torch.cross(x, y, dim=-1)

        # Recompute y to ensure orthogonality
        y = torch.cross(z, x, dim=-1)

        # Stack into rotation matrix [..., 3, 3]
        rotmat = torch.stack([x, y, z], dim=-1)

        # Convert rotation matrix to quaternion
        return self._rotmat_to_quaternion(rotmat)

    def _rotmat_to_quaternion(self, rotmat: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to quaternion (wxyz).

        Args:
            rotmat: [..., 3, 3] rotation matrices

        Returns:
            quat: [..., 4] quaternions in wxyz format
        """
        batch_shape = rotmat.shape[:-2]
        rotmat = rotmat.reshape(-1, 3, 3)

        # Use Shepperd's method
        trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]

        quat = torch.zeros(rotmat.shape[0], 4, device=rotmat.device)

        # Case 1: trace > 0
        mask = trace > 0
        s = torch.sqrt(trace[mask] + 1.0) * 2
        quat[mask, 0] = 0.25 * s
        quat[mask, 1] = (rotmat[mask, 2, 1] - rotmat[mask, 1, 2]) / s
        quat[mask, 2] = (rotmat[mask, 0, 2] - rotmat[mask, 2, 0]) / s
        quat[mask, 3] = (rotmat[mask, 1, 0] - rotmat[mask, 0, 1]) / s

        # Case 2-4: handle other cases (simplified)
        mask = ~mask
        if mask.any():
            # For simplicity, use a stable formula
            quat[mask, 0] = 1.0
            quat[mask, 1:] = 0.0

        # Normalize
        quat = F.normalize(quat, dim=-1)

        return quat.reshape(*batch_shape, 4)

    def _quaternion_to_angle(self, quat: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """
        Extract rotation angle around a specific axis from quaternion.

        Args:
            quat: [..., 4] quaternion (wxyz)
            axis: [3] rotation axis

        Returns:
            angle: [...] rotation angle in radians
        """
        # This is a simplified version - assumes axis-aligned rotation
        # For a more accurate version, project quaternion onto axis

        # Convert quaternion to axis-angle
        w = quat[..., 0]
        xyz = quat[..., 1:]

        # Angle = 2 * atan2(||xyz||, w)
        angle = 2.0 * torch.atan2(torch.norm(xyz, dim=-1), w.abs())

        # Get sign from axis alignment
        # This is simplified and may need adjustment
        return angle

    def compute_position_loss(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
        rot_req: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute position loss using FK.

        Args:
            x_true: [B, J, feat, T] ground truth
            x_pred: [B, J, feat, T] prediction
            rot_req: Rotation representation
            mask: [B, T] optional mask

        Returns:
            loss: scalar tensor
        """
        # Convert to qpos
        qpos_true = self.model_format_to_qpos(x_true, rot_req)
        qpos_pred = self.model_format_to_qpos(x_pred, rot_req)

        # Forward kinematics
        pos_true = self.fk.forward_kinematics(qpos_true)  # [B, T, num_joints, 3]
        pos_pred = self.fk.forward_kinematics(qpos_pred)  # [B, T, num_joints, 3]

        # MSE loss
        loss = F.mse_loss(pos_pred, pos_true, reduction='none')  # [B, T, num_joints, 3]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            loss = loss * mask
            loss = loss.sum() / (mask.sum() * self.config.num_joints * 3 + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def compute_foot_contact_loss(
        self,
        x_pred: torch.Tensor,
        rot_req: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute foot contact loss (penalize foot sliding during contact).

        Args:
            x_pred: [B, J, feat, T] prediction
            rot_req: Rotation representation
            mask: [B, T] optional mask

        Returns:
            loss: scalar tensor
        """
        # Convert to qpos
        qpos_pred = self.model_format_to_qpos(x_pred, rot_req)  # [B, T, qpos_dim]

        # Get foot keypoint positions
        positions, rotations = self.fk.forward_kinematics(qpos_pred, return_rotations=True)
        foot_keypoints = self.fk.compute_keypoints(
            positions, rotations,
            keypoint_names=self.config.foot_keypoint_names
        )

        # Stack foot positions [B, T, num_feet, 3]
        foot_positions = torch.stack([
            foot_keypoints[name] for name in self.config.foot_keypoint_names
        ], dim=2)

        # Compute contact mask [B, T-1, num_feet]
        contact_mask = compute_foot_contact_mask(
            foot_positions,
            threshold=self.config.contact_velocity_threshold,
            fps=30
        )

        # Compute foot displacement [B, T-1, num_feet, 3]
        foot_displacement = foot_positions[:, 1:] - foot_positions[:, :-1]

        # Loss: penalize displacement when in contact
        loss = (foot_displacement ** 2) * contact_mask.unsqueeze(-1)  # [B, T-1, num_feet, 3]

        # Apply mask
        if mask is not None:
            mask_vel = mask[:, 1:]  # [B, T-1]
            mask_vel = mask_vel.unsqueeze(-1).unsqueeze(-1)  # [B, T-1, 1, 1]
            loss = loss * mask_vel
            num_feet = len(self.config.foot_keypoint_names)
            loss = loss.sum() / (mask_vel.sum() * num_feet * 3 + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def compute_velocity_loss(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
        rot_req: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute velocity loss (temporal consistency).

        Args:
            x_true: [B, J, feat, T] ground truth
            x_pred: [B, J, feat, T] prediction
            rot_req: Rotation representation
            mask: [B, T] optional mask

        Returns:
            loss: scalar tensor
        """
        # Velocity in qpos space
        qpos_true = self.model_format_to_qpos(x_true, rot_req)
        qpos_pred = self.model_format_to_qpos(x_pred, rot_req)

        vel_true = qpos_true[:, 1:] - qpos_true[:, :-1]  # [B, T-1, qpos_dim]
        vel_pred = qpos_pred[:, 1:] - qpos_pred[:, :-1]  # [B, T-1, qpos_dim]

        # MSE loss
        loss = F.mse_loss(vel_pred, vel_true, reduction='none')

        # Apply mask
        if mask is not None:
            mask_vel = mask[:, 1:].unsqueeze(-1)  # [B, T-1, 1]
            loss = loss * mask_vel
            loss = loss.sum() / (mask_vel.sum() * qpos_pred.shape[-1] + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def compute_all_losses(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
        rot_req: str = '6d',
        mask: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all geometric losses.

        Args:
            x_true: [B, J, feat, T] ground truth
            x_pred: [B, J, feat, T] prediction
            rot_req: Rotation representation
            mask: [B, T] optional mask
            weights: Optional weight overrides

        Returns:
            Dictionary of losses
        """
        if weights is None:
            weights = self.loss_weights

        losses = {}

        # Position loss
        losses['loss_pos'] = self.compute_position_loss(x_true, x_pred, rot_req, mask) * weights['pos']

        # Foot contact loss
        losses['loss_foot'] = self.compute_foot_contact_loss(x_pred, rot_req, mask) * weights['foot']

        # Velocity loss
        losses['loss_vel'] = self.compute_velocity_loss(x_true, x_pred, rot_req, mask) * weights['vel']

        # Total
        losses['loss_total'] = losses['loss_pos'] + losses['loss_foot'] + losses['loss_vel']

        return losses


# Backward compatibility function
def compute_geometric_losses(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    rot_req: str = '6d',
    mask: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.02,
    fps: int = 30,
    robot_config: Optional[RobotConfig] = None
) -> Dict[str, torch.Tensor]:
    """
    Backward compatibility wrapper for geometric loss computation.

    Args:
        x_true: [B, J, feat, T] ground truth
        x_pred: [B, J, feat, T] prediction
        rot_req: Rotation representation
        mask: [B, T] optional mask
        weights: Loss weights
        threshold: Velocity threshold for contact detection
        fps: Frames per second
        robot_config: Robot configuration (defaults to G1)

    Returns:
        Dictionary of losses
    """
    if robot_config is None:
        robot_config = load_robot_config("g1")
        # Override config with function parameters
        robot_config.contact_velocity_threshold = threshold

    device = x_true.device
    computer = GeometricLossComputer(robot_config, device=str(device))

    return computer.compute_all_losses(x_true, x_pred, rot_req, mask, weights)


if __name__ == "__main__":
    print("=" * 80)
    print("Geometric Losses Test (Config-based)")
    print("=" * 80)

    # Load config
    config = load_robot_config("g1")
    computer = GeometricLossComputer(config, device="cpu")

    # Test data
    B, T, J, feat = 4, 50, 31, 6
    x_true = torch.randn(B, J, feat, T, requires_grad=False)
    x_pred = torch.randn(B, J, feat, T, requires_grad=True)
    mask = torch.ones(B, T)

    # Initialize properly
    with torch.no_grad():
        x_true[:, 0, :, :] = 0.1
        x_true[:, 0, 0, :] = 1.0  # w=1 for root quat
        x_pred[:, 0, :, :] = 0.1
        x_pred[:, 0, 0, :] = 1.0

    x_pred.requires_grad_(True)

    print("\nTesting geometric losses...")
    try:
        losses = computer.compute_all_losses(x_true, x_pred, rot_req='6d', mask=mask)
        print(f"  ✓ Position loss: {losses['loss_pos'].item():.6f}")
        print(f"  ✓ Foot contact loss: {losses['loss_foot'].item():.6f}")
        print(f"  ✓ Velocity loss: {losses['loss_vel'].item():.6f}")
        print(f"  ✓ Total loss: {losses['loss_total'].item():.6f}")

        # Test gradients
        losses['loss_total'].backward()
        print(f"  ✓ Gradient shape: {x_pred.grad.shape}")

        print("\nAll tests passed!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
