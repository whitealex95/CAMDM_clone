"""
Geometric Losses for Motion Diffusion Model

Implements three geometric loss terms for improving motion quality:
1. Position Loss (L_pos): MSE between FK-transformed GT and prediction
2. Foot Contact Loss (L_foot): Penalizes foot sliding during contact
3. Velocity Loss (L_vel): Ensures similar temporal dynamics

Reference model format:
- Model output shape: [B, J, feat, T] where J=31 (30 joints + 1 root pos)
  - Joints 0-29: Rotation representation (6D for rot_req='6d')
  - Joint 30: Root position (XYZ in first 3 dims)
"""

import torch
import torch.nn.functional as F
from utils.g1_kinematics import G1Kinematics
import utils.nn_transforms as nn_transforms


def model_format_to_qpos(model_output, rot_req='6d'):
    """
    Convert model output format to qpos format for FK computation.

    Args:
        model_output: [B, J, feat, T] or [B, T, J, feat] model output
            - Expected: [B, 31, feat, T] in first format
        rot_req: rotation representation ('6d', 'q', etc.)

    Returns:
        qpos: [B, T, 36] joint configuration
            - qpos[..., 0:3]: root position (XYZ)
            - qpos[..., 3:7]: root quaternion (WXYZ)
            - qpos[..., 7:36]: 29 joint angles
    """
    # Handle different input formats
    if model_output.dim() == 4:
        if model_output.shape[1] == 31:  # [B, 31, feat, T]
            model_output = model_output.permute(0, 3, 1, 2)  # [B, T, 31, feat]
        # else assume [B, T, J, feat]

    B, T, J, feat = model_output.shape
    assert J == 31, f"Expected 31 joints, got {J}"

    device = model_output.device
    qpos = torch.zeros(B, T, 36, device=device, dtype=model_output.dtype)

    # Extract root position from joint 30
    qpos[:, :, :3] = model_output[:, :, 30, :3]  # Root XYZ

    # Extract root rotation from joint 0
    root_rot_repr = model_output[:, :, 0, :]  # [B, T, feat]

    # Convert rotation representation to quaternion
    if rot_req == '6d':
        # Reshape to [B*T, feat] for conversion
        root_rot_flat = root_rot_repr.reshape(-1, feat)
        root_quat_flat = nn_transforms.repr6d2quat(root_rot_flat)  # [B*T, 4] wxyz
        root_quat = root_quat_flat.reshape(B, T, 4)
    elif rot_req == 'q':
        root_quat = root_rot_repr[:, :, :4]  # Already quaternion
    else:
        raise NotImplementedError(f"Rotation type '{rot_req}' not implemented")

    qpos[:, :, 3:7] = root_quat  # Root quaternion (wxyz)

    # Extract joint angles (29 joints, stored in dim 0 of padded representation)
    joint_angles = model_output[:, :, 1:30, 0]  # [B, T, 29]
    qpos[:, :, 7:] = joint_angles

    return qpos


def compute_position_loss(pos_pred, pos_true, mask=None):
    """
    Position Loss (L_pos): MSE between FK-transformed GT and prediction.

    Formula:
        L_pos = Mean over N frames of || pos_true - pos_pred ||^2

    Args:
        pos_pred: [B, T, 30, 3] predicted joint positions from FK
        pos_true: [B, T, 30, 3] ground truth joint positions from FK
        mask: [B, T] optional binary mask (1 = valid frame)

    Returns:
        loss: [B,] tensor representing average position loss per batch element
    """
    # Compute MSE
    mse = F.mse_loss(pos_pred, pos_true, reduction='none')  # [B, T, 30, 3]

    # Average over joints and spatial dimensions -> [B, T]
    mse = mse.mean(dim=[2, 3])  # [B, T]

    # Apply mask if provided
    if mask is not None:
        mse = mse * mask  # [B, T]
        # Average over time per batch element
        loss = mse.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B,]
    else:
        loss = mse.mean(dim=1)  # [B,]

    return loss



def compute_foot_contact_loss(foot_pos_pred, fc_mask, mask=None):
    """
    Foot Contact Loss (L_foot): Penalizes foot sliding during contact phases.

    Aligned with MotionTrainingPortal implementation:
    - Computes predicted foot velocity
    - Zeros out velocity where NOT in contact (using fc_mask)
    - Computes MSE between masked velocity and zeros

    Args:
        foot_pos_pred: [B, T, 2, 3] predicted foot positions (left, right)
        fc_mask: [B, 2, 3, T-1] binary contact mask (1 = in contact, velocity should be 0)
        mask: [B, 1, 1, T] optional binary validity mask (1 = valid frame)

    Returns:
        loss: [B,] tensor representing average sliding velocity during contact
    """
    # Compute predicted foot velocity: [B, T-1, 2, 3]
    pred_vel = foot_pos_pred[:, 1:] - foot_pos_pred[:, :-1]

    # Permute to match MotionTrainingPortal format: [B, 2, 3, T-1]
    pred_vel = pred_vel.permute(0, 2, 3, 1)

    # Zero out velocity where NOT in contact (fc_mask=False means no contact)
    pred_vel = pred_vel.clone()
    pred_vel[~fc_mask] = 0

    # Target is zeros (foot should not move during contact)
    target = torch.zeros_like(pred_vel)

    # Compute MSE: [B, 2, 3, T-1]
    mse = F.mse_loss(pred_vel, target, reduction='none')

    # Apply temporal mask if provided
    if mask is not None:
        temporal_mask = mask[..., 1:]  # [B, 1, 1, T-1]
        mse = mse * temporal_mask
        # Sum over (feet, xyz, time), divide by valid elements
        n_entries = mse.shape[1] * mse.shape[2]  # 2 * 3 = 6
        loss = mse.sum(dim=[1, 2, 3]) / (temporal_mask.sum(dim=[1, 2, 3]) * n_entries + 1e-8)  # [B,]
    else:
        loss = mse.mean(dim=[1, 2, 3])  # [B,]

    return loss



def compute_geometric_losses(x_true, x_pred, rot_req='6d', mask=None,
                             weights=None, vel_threshold=0.01, g1_kin=None):
    """
    Compute all geometric losses and return as a dictionary.

    Aligned with MotionTrainingPortal implementation.
    Efficiently computes FK only once and reuses results for all losses.

    Args:
        x_true: [B, J, feat, T] or [B, T, J, feat] ground truth
        x_pred: [B, J, feat, T] or [B, T, J, feat] prediction
        rot_req: rotation representation type
        mask: [B, T] optional binary mask (1 = valid frame)
        weights: dict with keys 'geo_loss_weight_pos', 'geo_loss_weight_foot', 'geo_loss_weight_vel'
        vel_threshold: velocity threshold for foot contact (per frame, default 0.01 matching MotionTrainingPortal)
        g1_kin: G1Kinematics instance (optional, creates new if None)

    Returns:
        losses: dict with keys 'loss_geo_pos', 'loss_geo_vel', 'loss_geo_foot' (each [B,] tensor)
    """
    if weights is None:
        weights = {'geo_loss_weight_pos': 1.0, 'geo_loss_weight_foot': 0.5, 'geo_loss_weight_vel': 0.1}

    # Create kinematics instance if not provided
    if g1_kin is None:
        g1_kin = G1Kinematics()

    losses = {}

    # Get batch size
    B = x_pred.shape[0]

    # Convert to qpos format once
    qpos_true = model_format_to_qpos(x_true, rot_req)  # [B, T, 36]
    qpos_pred = model_format_to_qpos(x_pred, rot_req)  # [B, T, 36]

    # Compute FK once for both true and pred
    with torch.no_grad():
        pos_true = g1_kin.forward_kinematics(qpos_true)  # [B, T, 30, 3]
    pos_pred = g1_kin.forward_kinematics(qpos_pred)  # [B, T, 30, 3]

    # Extract foot positions once
    # left_ankle_roll=6, right_ankle_roll=12
    foot_pos_pred = torch.stack([pos_pred[..., 6, :], pos_pred[..., 12, :]], dim=-2)  # [B, T, 2, 3]

    # Position loss
    if weights.get('geo_loss_weight_pos', 0) > 0:
        losses['loss_geo_pos'] = compute_position_loss(pos_pred, pos_true, mask)
    else:
        losses['loss_geo_pos'] = torch.zeros(B, device=x_pred.device)

    # Foot contact loss (aligned with MotionTrainingPortal)
    if weights.get('geo_loss_weight_foot', 0) > 0:
        # Compute GT foot positions and velocity
        with torch.no_grad():
            foot_pos_true = torch.stack([pos_true[..., 6, :], pos_true[..., 12, :]], dim=-2)  # [B, T, 2, 3]
            # Permute to [B, 2, 3, T] to match MotionTrainingPortal
            gt_foot_xyz = foot_pos_true.permute(0, 2, 3, 1)  # [B, 2, 3, T]
            # Compute GT foot velocity norm
            gt_foot_vel = torch.linalg.norm(gt_foot_xyz[..., 1:] - gt_foot_xyz[..., :-1], dim=2)  # [B, 2, T-1]
            # Contact mask: velocity <= threshold (same as MotionTrainingPortal's 0.01)
            fc_mask = (gt_foot_vel <= vel_threshold).unsqueeze(2).expand(-1, -1, 3, -1)  # [B, 2, 3, T-1]

        # Reshape mask for foot contact loss: [B, T] -> [B, 1, 1, T]
        foot_mask = mask.view(B, 1, 1, -1) if mask is not None else None
        losses['loss_geo_foot'] = compute_foot_contact_loss(foot_pos_pred, fc_mask, foot_mask)
    else:
        losses['loss_geo_foot'] = torch.zeros(B, device=x_pred.device)

    # Velocity loss (computed on FK positions, not raw representation)
    if weights.get('geo_loss_weight_vel', 0) > 0:
        # Compute velocities from FK positions
        vel_true = pos_true[:, 1:] - pos_true[:, :-1]  # [B, T-1, 30, 3]
        vel_pred = pos_pred[:, 1:] - pos_pred[:, :-1]  # [B, T-1, 30, 3]

        # MSE between velocities
        mse = F.mse_loss(vel_pred, vel_true, reduction='none')  # [B, T-1, 30, 3]
        mse = mse.mean(dim=[2, 3])  # [B, T-1]

        # Apply mask if provided
        if mask is not None:
            temporal_mask = mask[:, :-1]  # [B, T-1]
            mse = mse * temporal_mask
            # Average over time per batch element
            losses['loss_geo_vel'] = mse.sum(dim=1) / (temporal_mask.sum(dim=1) + 1e-8)  # [B,]
        else:
            losses['loss_geo_vel'] = mse.mean(dim=1)  # [B,]
    else:
        losses['loss_geo_vel'] = torch.zeros(B, device=x_pred.device)

    return losses


# ==============================================================================
# Test Functions
# ==============================================================================

def test_losses():
    """Test loss computation with dummy data."""
    print("Testing Geometric Losses...")

    B, T, J, feat = 4, 50, 31, 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    x_true = torch.randn(B, J, feat, T, device=device, requires_grad=False)
    x_pred = torch.randn(B, J, feat, T, device=device, requires_grad=False)
    mask = torch.ones(B, T, device=device)

    # Initialize root properly
    with torch.no_grad():
        x_true[:, 0, :, :] = 0.1  # Root rotation
        # Set root position with linear motion
        root_pos_traj = torch.linspace(0, 1, T, device=device).view(1, 1, T).expand(B, 3, T)  # [B, 3, T]
        x_true[:, 30, :3, :] = root_pos_traj  # Root pos
        x_pred[:, 0, :, :] = 0.1
        x_pred[:, 30, :3, :] = root_pos_traj

        # Set root quaternion w=1 for valid quaternions
        x_true[:, 0, 0, :] = 1.0  # Assuming 6d, first component
        x_pred[:, 0, 0, :] = 1.0

    # Re-enable gradients for x_pred
    x_pred.requires_grad_(True)

    print("Input shapes:")
    print(f"  x_true: {x_true.shape}")
    print(f"  x_pred: {x_pred.shape}")
    print(f"  mask: {mask.shape}")
    print()

    # Test combined losses (this now computes FK only once)
    print("Testing Combined Losses (efficient FK computation)...")
    losses = compute_geometric_losses(
        x_true, x_pred, rot_req='6d', mask=mask,
        weights={'pos': 1.0, 'foot': 0.5, 'vel': 0.1}
    )
    print(f"  Position: {losses['loss_pos'].item():.6f}")
    print(f"  Foot Contact: {losses['loss_foot'].item():.6f}")
    print(f"  Velocity: {losses['loss_vel'].item():.6f}")
    print(f"  Total: {losses['loss_total'].item():.6f}")
    assert losses['loss_pos'].item() >= 0, "Position loss must be non-negative"
    assert losses['loss_foot'].item() >= 0, "Foot contact loss must be non-negative"
    assert losses['loss_vel'].item() >= 0, "Velocity loss must be non-negative"
    print("  ✓ Combined losses computed (FK called only once)")
    print()

    # Test gradient flow
    print("Testing Gradient Flow...")
    losses['loss_total'].backward()
    assert x_pred.grad is not None, "Gradient not computed"
    print(f"  Gradient shape: {x_pred.grad.shape}")
    print(f"  Gradient norm: {x_pred.grad.norm().item():.6f}")
    print("  ✓ Gradient flow verified")
    print()

    print("All loss tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Geometric Losses Test Suite")
    print("=" * 60)
    print()

    test_losses()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
