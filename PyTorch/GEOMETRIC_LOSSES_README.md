# Geometric Losses for G1 Humanoid Motion Generation

This document describes the implementation of geometric losses for improving the quality of kinematic motion generation using diffusion models.

## Overview

The system implements three geometric loss terms in addition to the vanilla diffusion loss:

1. **Position Loss (L_pos)**: MSE between FK-transformed ground truth and prediction
2. **Foot Contact Loss (L_foot)**: Penalizes foot sliding during contact
3. **Velocity Loss (L_vel)**: Ensures similar temporal dynamics

## File Structure

```
PyTorch/
├── utils/
│   └── g1_kinematics.py          # Batch FK and foot contact detection
├── network/
│   ├── geometric_losses.py        # Geometric loss implementations
│   └── training.py                # Training with geometric losses
└── visualize/
    └── visualize_fk_contact.py    # MuJoCo visualization
```

## 1. Forward Kinematics (FK) Implementation

Located in: [`utils/g1_kinematics.py`](utils/g1_kinematics.py)

### Features

- **Batched PyTorch FK**: Differentiable forward kinematics for G1 humanoid
- **Foot Contact Detection**: Binary mask based on foot velocity
- **Fully Vectorized**: Efficient batch and sequence processing

### API

```python
import utils.g1_kinematics as g1_kin

# Forward Kinematics
qpos = torch.randn(B, T, 36)  # [batch, time, qpos_dim]
positions = g1_kin.forward_kinematics_g1(qpos)  # [B, T, 30, 3]

# Foot Contact Detection
foot_positions = g1_kin.extract_foot_positions(positions)  # [B, T, 2, 3]
contact_mask = g1_kin.compute_foot_contact_mask(foot_positions)  # [B, T-1, 2]

# Convenience Functions
foot_positions = g1_kin.qpos_to_foot_positions(qpos)
contact_mask = g1_kin.qpos_to_foot_contact(qpos, threshold=0.02, fps=30)
```

### Joint Configuration (qpos)

The qpos format is `[B, T, 36]`:
- `qpos[..., 0:3]`: Root position (XYZ)
- `qpos[..., 3:7]`: Root quaternion (WXYZ)
- `qpos[..., 7:36]`: 29 joint angles (1D revolute joints)

### Robot Structure

The G1 humanoid has 30 joints (1 root + 29 actuated):
- **Legs**: 6 DOF per leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
- **Torso**: 3 DOF (waist yaw/roll/pitch)
- **Arms**: 7 DOF per arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)

Foot joint indices:
- Left foot: joint 6 (`left_ankle_roll`)
- Right foot: joint 12 (`right_ankle_roll`)

## 2. Geometric Losses

Located in: [`network/geometric_losses.py`](network/geometric_losses.py)

### Loss Formulas

#### Position Loss (L_pos)
Reconstructs 3D joint positions using FK and computes MSE:

```
L_pos = Mean over N frames of || FK(x_true) - FK(x_pred) ||²
```

#### Foot Contact Loss (L_foot)
Penalizes foot motion when foot is in contact with ground:

```
L_foot = Mean over (N-1) frames of || (FK(x_pred[i+1]) - FK(x_pred[i])) * f_i ||²
```

Where `f_i` is a binary contact mask (1 = contact, 0 = no contact).

Contact is detected when foot velocity < threshold (default: 0.02 m/s).

#### Velocity Loss (L_vel)
Ensures temporal consistency between GT and prediction:

```
L_vel = Mean over (N-1) frames of || (x_true[i+1] - x_true[i]) - (x_pred[i+1] - x_pred[i]) ||²
```

### API

```python
from network.geometric_losses import compute_geometric_losses

# Model output format: [B, J, feat, T] where J=31 (30 joints + 1 root pos)
losses = compute_geometric_losses(
    x_true=target,       # [B, J, feat, T]
    x_pred=prediction,   # [B, J, feat, T]
    rot_req='6d',        # Rotation representation
    mask=mask,           # [B, T] optional
    weights={'pos': 1.0, 'foot': 0.5, 'vel': 0.1},
    threshold=0.02,      # m/s for foot contact
    fps=30
)

# Returns dict with:
# - losses['loss_pos']: Position loss
# - losses['loss_foot']: Foot contact loss
# - losses['loss_vel']: Velocity loss
# - losses['loss_total']: Weighted sum
```

## 3. Training Integration

Located in: [`network/training.py`](network/training.py:314-423)

### Configuration

Add these parameters to your config JSON:

```json
{
  "trainer": {
    "use_geometric_losses": true,
    "geo_loss_weight_pos": 1.0,
    "geo_loss_weight_foot": 0.5,
    "geo_loss_weight_vel": 0.1
  }
}
```

### Training Command

```bash
# Without geometric losses (default)
python train_g1.py -n camdm_g1 --epoch 10000 --batch_size 512 --diffusion_steps 4 --data data/pkls/lafan1_g1_motion27.pkl

# With geometric losses (add to config)
# Edit config/default_g1.json to add geometric loss settings above, then:
python train_g1.py -n camdm_g1_geo --epoch 10000 --batch_size 512 --diffusion_steps 4 --data data/pkls/lafan1_g1_motion27.pkl
```

### Loss Terms in Logs

During training, you'll see these loss components:

- `loss_data`: Vanilla diffusion MSE loss
- `loss_data_vel`: Diffusion velocity loss
- `loss_geo_pos`: FK-based position loss
- `loss_geo_foot`: Foot contact loss
- `loss_geo_vel`: Geometric velocity loss
- `loss`: Total weighted sum

## 4. Visualization

Located in: [`visualize/visualize_fk_contact.py`](visualize/visualize_fk_contact.py)

### Usage

```bash
python visualize/visualize_fk_contact.py --dataset lafan1_g1_motion27 --motion 0
```

### Controls

- `SPACE`: Pause/Resume
- `UP/DOWN`: Previous/Next motion clip
- `F`: Toggle FK joint visualization (spheres)
- `K`: Toggle FK skeleton visualization (wireframe)
- `C`: Toggle foot contact visualization
- `1-9`: Set playback speed
- `S`: Print status
- `ESC`: Exit

### Visualization Legend

**FK Joint Spheres (toggle with F):**
- **Blue spheres**: FK computed joint positions (legs)
- **Purple spheres**: FK computed joint positions (waist)
- **Orange spheres**: FK computed joint positions (arms)
- **Bright green sphere**: Left hand position (highlighted)
- **Bright orange sphere**: Right hand position (highlighted)

**FK Skeleton Wireframe (toggle with K):**
- **Cyan lines**: Left leg connections
- **Pink lines**: Right leg connections
- **Yellow lines**: Torso connections
- **Green lines**: Left arm connections
- **Orange lines**: Right arm connections

**Foot Contact (toggle with C):**
- **Red marker + cylinder**: Foot in contact with ground
- **Green marker**: Foot not in contact

## Testing

### Test FK Implementation

```bash
python utils/g1_kinematics.py
```

Expected output:
```
============================================================
G1 Kinematics Test Suite
============================================================

Testing Forward Kinematics...
✓ Single pose: torch.Size([1, 30, 3])
✓ Batch of poses: torch.Size([16, 30, 3])
✓ Sequence of poses: torch.Size([8, 50, 30, 3])
✓ Gradient flow: grad shape torch.Size([2, 10, 36])
Forward Kinematics tests passed!

Testing Foot Contact Detection...
✓ Foot positions: torch.Size([4, 100, 2, 3])
✓ Contact mask: torch.Size([4, 99, 2])
✓ Convenience function matches
Foot Contact Detection tests passed!

All tests passed!
```

### Test Geometric Losses

```bash
python network/geometric_losses.py
```

Expected output:
```
============================================================
Geometric Losses Test Suite
============================================================

Testing Geometric Losses...
Testing Position Loss...
  ✓ Position loss computed
Testing Foot Contact Loss...
  ✓ Foot contact loss computed
Testing Velocity Loss...
  ✓ Velocity loss computed
Testing Combined Losses...
  ✓ Combined losses computed
Testing Gradient Flow...
  ✓ Gradient flow verified

All tests passed!
```

## Implementation Details

### Coordinate System

The G1 robot uses:
- **+X**: Forward direction
- **+Y**: Left direction
- **+Z**: Up direction

### Model Format

The diffusion model works with `[B, J, feat, T]` format:
- `J = 31`: 30 joints + 1 root position channel
- For rot_req='6d': `feat = 6`
- Joints 0-29: Rotation representation (6D)
- Joint 30: Root position (XYZ in first 3 dims)

### Conversion Functions

```python
from network.geometric_losses import model_format_to_qpos

# Convert model output to qpos for FK
qpos = model_format_to_qpos(model_output, rot_req='6d')  # [B, T, 36]
```

## Recommended Hyperparameters

Based on typical motion generation tasks:

| Loss Term | Weight | Description |
|-----------|--------|-------------|
| Position  | 1.0    | Primary geometric constraint |
| Foot Contact | 0.5 | Prevent sliding artifacts |
| Velocity  | 0.1    | Temporal smoothness |

Foot contact threshold: 0.02 m/s (2 cm/s)

## Troubleshooting

### Issue: Foot contact loss is always zero

**Solution**: Check that your motion has actual foot contacts. The mask is computed from ground truth velocity, so if feet are always moving, there will be no contacts.

### Issue: Position loss is too high

**Solution**: Ensure your FK implementation matches the robot URDF/XML. Check joint offsets and parent indices in [`g1_kinematics.py`](utils/g1_kinematics.py:65-130).

### Issue: Gradient issues during training

**Solution**: The FK function uses `.clone()` to avoid in-place operations. If you still have issues, check that your data format matches expectations.

## Future Improvements

Potential extensions:

1. **Joint Limit Constraints**: Add losses to enforce physical joint limits
2. **Balance Loss**: Penalize center-of-mass outside support polygon
3. **Energy Minimization**: Encourage efficient motions
4. **Trajectory Alignment**: Ensure generated motion follows target trajectory more closely

## Citation

If you use this implementation, please cite:

```bibtex
@misc{g1_geometric_losses,
  title={Geometric Losses for Humanoid Motion Diffusion},
  author={Your Name},
  year={2026}
}
```

## References

- G1 Humanoid Robot Specifications: [Unitree Robotics](https://www.unitree.com/)
- Forward Kinematics: Modern Robotics (Lynch & Park, 2017)
- Foot Contact Detection: Derivation from motion capture best practices
