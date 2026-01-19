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
import utils.g1_kinematics as g1_kin
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


def compute_position_loss(x_true, x_pred, rot_req='6d', mask=None):
    """
    Position Loss (L_pos): MSE between FK-transformed GT and prediction.

    Formula:
        L_pos = Mean over N frames of || FK(x_true) - FK(x_pred) ||^2

    Args:
        x_true: [B, J, feat, T] or [B, T, J, feat] ground truth
        x_pred: [B, J, feat, T] or [B, T, J, feat] prediction
        rot_req: rotation representation type
        mask: [B, T] optional binary mask (1 = valid frame)

    Returns:
        loss: scalar tensor
    """
    # Convert to qpos format
    qpos_true = model_format_to_qpos(x_true, rot_req)  # [B, T, 36]
    qpos_pred = model_format_to_qpos(x_pred, rot_req)  # [B, T, 36]

    # Compute FK
    with torch.no_grad() if x_true.requires_grad else torch.enable_grad():
        pos_true = g1_kin.forward_kinematics_g1(qpos_true)  # [B, T, 30, 3]

    pos_pred = g1_kin.forward_kinematics_g1(qpos_pred)  # [B, T, 30, 3]

    # Compute MSE
    mse = F.mse_loss(pos_pred, pos_true, reduction='none')  # [B, T, 30, 3]

    # Average over joints and spatial dimensions
    mse = mse.mean(dim=[2, 3])  # [B, T]

    # Apply mask if provided
    if mask is not None:
        mse = mse * mask
        loss = mse.sum() / (mask.sum() + 1e-8)
    else:
        loss = mse.mean()

    return loss



def compute_foot_contact_loss(x_pred, x_gt, rot_req='6d', mask=None, vel_threshold=0.02, fps=30):
    """
    Foot Contact Loss (L_foot): Penalizes foot sliding during contact phases.

    This loss identifies contact frames using the Ground Truth (x_gt) motion,
    and then penalizes the velocity of the Predicted (x_pred) motion during those frames.

    Formula:
        L_foot = Mean( || Vel_pred[t] ||^2 * Mask_gt[t] )

    Args:
        x_pred: [B, J, feat, T] or [B, T, J, feat] predicted motion
        x_gt: [B, J, feat, T] or [B, T, J, feat] ground truth motion (used for mask)
        rot_req: rotation representation type
        mask: [B, T] optional binary validity mask (1 = valid frame)
        vel_threshold: velocity threshold for contact detection (m/s)
        fps: frames per second

    Returns:
        loss: scalar tensor representing average sliding velocity squared
    """
    # -----------------------------------------------------------
    # 1. Process Prediction (to get the velocity we want to penalize)
    # -----------------------------------------------------------
    qpos_pred = model_format_to_qpos(x_pred, rot_req)
    pos_pred = g1_kin.forward_kinematics_g1(qpos_pred)    # [B, T, J, 3]
    foot_pos_pred = g1_kin.extract_foot_positions(pos_pred) # [B, T, 2, 3]

    # Compute predicted velocity
    # [B, T-1, 2, 3]
    pred_velocity = (foot_pos_pred[:, 1:] - foot_pos_pred[:, :-1]) * fps

    # -----------------------------------------------------------
    # 2. Process Ground Truth (to determine WHEN we should penalize)
    # -----------------------------------------------------------
    with torch.no_grad():
        qpos_gt = model_format_to_qpos(x_gt, rot_req)
        pos_gt = g1_kin.forward_kinematics_g1(qpos_gt)
        foot_pos_gt = g1_kin.extract_foot_positions(pos_gt)

        # Compute mask based on GT stability
        # [B, T-1, 2]
        gt_contact_mask = g1_kin.compute_foot_contact_mask(
            foot_pos_gt, 
            vel_threshold=vel_threshold, 
            fps=fps
        )

    # -----------------------------------------------------------
    # 3. Compute Loss
    # -----------------------------------------------------------
    # Zero out velocity when GT says there is no contact
    # gt_contact_mask: [B, T-1, 2] -> [B, T-1, 2, 1]
    masked_velocity = pred_velocity * gt_contact_mask.unsqueeze(-1)

    # Sum squared error over x,y,z and both feet
    loss_per_frame = (masked_velocity ** 2).sum(dim=[2, 3])  # [B, T-1]

    # Apply global temporal mask if provided
    if mask is not None:
        temporal_mask = mask[:, :-1]  # Align length [B, T] -> [B, T-1]
        loss_per_frame = loss_per_frame * temporal_mask
        loss = loss_per_frame.sum() / (temporal_mask.sum() + 1e-8)
    else:
        loss = loss_per_frame.mean()

    return loss

def compute_velocity_loss(x_true, x_pred, rot_req='6d', mask=None):
    """
    Velocity Loss (L_vel): Ensures similar temporal dynamics between GT and prediction.

    Formula:
        L_vel = Mean over (N-1) frames of || (x_true[i+1] - x_true[i]) - (x_pred[i+1] - x_pred[i]) ||^2

    Args:
        x_true: [B, J, feat, T] or [B, T, J, feat] ground truth
        x_pred: [B, J, feat, T] or [B, T, J, feat] prediction
        rot_req: rotation representation type (not used, operates directly on representation)
        mask: [B, T] optional binary mask (1 = valid frame)

    Returns:
        loss: scalar tensor
    """
    # Normalize input format to [B, T, J, feat]
    if x_true.dim() == 4 and x_true.shape[1] != x_true.shape[3]:
        if x_true.shape[1] < x_true.shape[3]:  # [B, J, feat, T]
            x_true = x_true.permute(0, 3, 1, 2)  # [B, T, J, feat]
            x_pred = x_pred.permute(0, 3, 1, 2)

    B, T, J, feat = x_true.shape

    # Compute velocities (finite differences)
    vel_true = x_true[:, 1:] - x_true[:, :-1]  # [B, T-1, J, feat]
    vel_pred = x_pred[:, 1:] - x_pred[:, :-1]  # [B, T-1, J, feat]

    # Compute MSE between velocities
    mse = F.mse_loss(vel_pred, vel_true, reduction='none')  # [B, T-1, J, feat]

    # Average over joints and features
    mse = mse.mean(dim=[2, 3])  # [B, T-1]

    # Apply mask if provided
    if mask is not None:
        temporal_mask = mask[:, :-1]  # [B, T-1]
        mse = mse * temporal_mask
        loss = mse.sum() / (temporal_mask.sum() + 1e-8)
    else:
        loss = mse.mean()

    return loss


def compute_geometric_losses(x_true, x_pred, rot_req='6d', mask=None,
                             weights=None, threshold=0.02, fps=30):
    """
    Compute all geometric losses and return as a dictionary.

    Args:
        x_true: [B, J, feat, T] or [B, T, J, feat] ground truth
        x_pred: [B, J, feat, T] or [B, T, J, feat] prediction
        rot_req: rotation representation type
        mask: [B, T] optional binary mask (1 = valid frame)
        weights: dict with keys 'pos', 'foot', 'vel' for loss weighting
        threshold: velocity threshold for foot contact (m/s)
        fps: frames per second

    Returns:
        losses: dict with keys 'loss_pos', 'loss_foot', 'loss_vel', 'loss_total'
    """
    if weights is None:
        weights = {'pos': 1.0, 'foot': 0.5, 'vel': 0.1}

    losses = {}

    # Position loss
    if weights.get('pos', 0) > 0:
        losses['loss_pos'] = compute_position_loss(x_true, x_pred, rot_req, mask)
    else:
        losses['loss_pos'] = torch.tensor(0.0, device=x_pred.device)

    # Foot contact loss
    if weights.get('foot', 0) > 0:
        losses['loss_foot'] = compute_foot_contact_loss(x_pred, rot_req, mask, threshold, fps)
    else:
        losses['loss_foot'] = torch.tensor(0.0, device=x_pred.device)

    # Velocity loss
    if weights.get('vel', 0) > 0:
        losses['loss_vel'] = compute_velocity_loss(x_true, x_pred, rot_req, mask)
    else:
        losses['loss_vel'] = torch.tensor(0.0, device=x_pred.device)

    # Total weighted loss
    losses['loss_total'] = (
        weights.get('pos', 0) * losses['loss_pos'] +
        weights.get('foot', 0) * losses['loss_foot'] +
        weights.get('vel', 0) * losses['loss_vel']
    )

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

    # Test position loss
    print("Testing Position Loss...")
    loss_pos = compute_position_loss(x_true, x_pred, rot_req='6d', mask=mask)
    print(f"  Loss: {loss_pos.item():.6f}")
    assert loss_pos.item() >= 0, "Position loss must be non-negative"
    print("  ✓ Position loss computed")
    print()

    # Test foot contact loss
    print("Testing Foot Contact Loss...")
    loss_foot = compute_foot_contact_loss(x_pred, rot_req='6d', mask=mask)
    print(f"  Loss: {loss_foot.item():.6f}")
    assert loss_foot.item() >= 0, "Foot contact loss must be non-negative"
    print("  ✓ Foot contact loss computed")
    print()

    # Test velocity loss
    print("Testing Velocity Loss...")
    loss_vel = compute_velocity_loss(x_true, x_pred, rot_req='6d', mask=mask)
    print(f"  Loss: {loss_vel.item():.6f}")
    assert loss_vel.item() >= 0, "Velocity loss must be non-negative"
    print("  ✓ Velocity loss computed")
    print()

    # Test combined losses
    print("Testing Combined Losses...")
    losses = compute_geometric_losses(
        x_true, x_pred, rot_req='6d', mask=mask,
        weights={'pos': 1.0, 'foot': 0.5, 'vel': 0.1}
    )
    print(f"  Position: {losses['loss_pos'].item():.6f}")
    print(f"  Foot Contact: {losses['loss_foot'].item():.6f}")
    print(f"  Velocity: {losses['loss_vel'].item():.6f}")
    print(f"  Total: {losses['loss_total'].item():.6f}")
    print("  ✓ Combined losses computed")
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
