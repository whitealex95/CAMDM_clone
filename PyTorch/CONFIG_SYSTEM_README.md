# Config-Based Robot System

This document describes the new configuration-based system for working with different humanoid robots.

## Overview

The system allows you to easily adapt the codebase to different humanoid robots by creating a YAML configuration file. Key benefits:

1. **Robot-agnostic FK**: Forward kinematics works with any robot defined in a config file
2. **Virtual keypoints**: Define additional points (heel, toe, hands) without modifying MuJoCo XML
3. **Easy customization**: Change visualization colors, contact thresholds, loss weights via config
4. **Backward compatible**: Existing code continues to work with G1 robot

## File Structure

```
PyTorch/
├── config/
│   └── robots/
│       └── g1.yaml                    # G1 robot configuration
├── utils/
│   ├── robot_config.py                # Config loader
│   ├── kinematics.py                  # Generic FK implementation
│   ├── keypoints.py                   # Keypoint computation
│   └── g1_kinematics.py              # (Legacy) G1-specific FK
└── network/
    ├── geometric_losses_v2.py         # Config-based losses (future)
    └── geometric_losses.py            # (Current) G1-specific losses
```

## Quick Start

### 1. Using with G1 Robot (Default)

```python
from utils.keypoints import compute_keypoints_from_qpos, get_foot_keypoint_positions

# Compute all keypoints from qpos
qpos = torch.randn(batch, time, 36)
keypoints = compute_keypoints_from_qpos(qpos)
# Returns: {"left_heel": [...], "left_toe": [...], "right_heel": [...], ...}

# Get just foot keypoints for contact detection
foot_positions = get_foot_keypoint_positions(qpos)  # [B, T, 4, 3]
```

### 2. Using Generic FK

```python
from utils.robot_config import load_robot_config
from utils.kinematics import HumanoidFK

# Load robot config
config = load_robot_config("g1")

# Create FK computer
fk = HumanoidFK(config, device="cuda")

# Compute FK
positions, rotations = fk.forward_kinematics(qpos, return_rotations=True)

# Compute keypoints
keypoints = fk.compute_keypoints(positions, rotations)
```

### 3. Creating a Custom Robot Config

Create `config/robots/my_robot.yaml`:

```yaml
robot:
  name: "My Robot"
  mjcf_path: "path/to/robot.xml"

  root:
    body_name: "pelvis"
    default_height: 0.8

  joints:
    names:
      - "joint_1"
      - "joint_2"
      # ... list all actuated joints

keypoints:
  left_heel:
    parent_joint: "left_ankle_joint"
    offset: [0.0, 0.0, -0.05]
    type: "foot_contact"

  # Add more keypoints...

visualization:
  keypoint_colors:
    left_heel: [0.0, 0.5, 1.0, 0.8]

contact:
  foot_keypoints: ["left_heel", "right_heel"]
  velocity_threshold: 0.02

losses:
  position_weight: 1.0
  foot_contact_weight: 0.5
  velocity_weight: 0.1
```

Then use it:

```python
config = load_robot_config("my_robot")
fk = HumanoidFK(config)
```

## Configuration File Format

### Robot Section

```yaml
robot:
  name: "Robot display name"
  mjcf_path: "relative/path/to/scene.xml"  # Relative to project root

  root:
    body_name: "name_of_root_body_in_xml"
    default_height: 0.793  # meters

  joints:
    names:
      # List of actuated joint names in order
      # These should match joint names in the MuJoCo XML
      - "joint_1"
      - "joint_2"
      # ...
```

The skeleton structure (parents, offsets, axes, initial quaternions) is **automatically extracted** from the MuJoCo XML file. You don't need to manually specify these.

### Keypoints Section

Define virtual keypoints that don't exist in the MuJoCo XML:

```yaml
keypoints:
  keypoint_name:
    parent_joint: "name_of_parent_joint"  # Must be in joints.names
    offset: [x, y, z]  # Offset in parent joint's local frame (meters)
    type: "foot_contact" | "end_effector" | "tracking"
```

**Common keypoint types:**
- `foot_contact`: For contact detection (heel, toe)
- `end_effector`: For manipulation tasks (hand, gripper)
- `tracking`: For motion tracking (head, pelvis)

### Visualization Section

```yaml
visualization:
  joint_colors:
    legs: [r, g, b, a]    # RGBA color for leg joints
    torso: [r, g, b, a]
    arms: [r, g, b, a]

  keypoint_colors:
    left_heel: [r, g, b, a]
    # ... one per keypoint

  skeleton_colors:
    left_leg: [r, g, b, a]
    right_leg: [r, g, b, a]
    # ...

  joint_ranges:
    legs: [0, 12]   # Joint indices for legs
    torso: [13, 15]
    arms: [16, 29]
```

### Contact Section

```yaml
contact:
  foot_keypoints:  # List of keypoint names to use for contact detection
    - "left_heel"
    - "left_toe"
    - "right_heel"
    - "right_toe"

  velocity_threshold: 0.02  # m/s - below this is contact
  height_threshold: 0.05    # m - for additional contact heuristics

  contact_color: [1.0, 0.0, 0.0, 0.8]      # Red when in contact
  no_contact_color: [0.2, 1.0, 0.2, 0.3]   # Faint green when not
```

### Losses Section

```yaml
losses:
  position_weight: 1.0        # Weight for FK position loss
  foot_contact_weight: 0.5    # Weight for foot sliding loss
  velocity_weight: 0.1        # Weight for velocity consistency loss
```

## API Reference

### robot_config.py

```python
load_robot_config(config_name: str) -> RobotConfig
```
Load a robot configuration from `config/robots/{config_name}.yaml`.

**Returns:** `RobotConfig` object with all robot parameters.

### kinematics.py

```python
class HumanoidFK:
    def __init__(self, robot_config: RobotConfig, device: str = "cpu")

    def forward_kinematics(
        self,
        qpos: torch.Tensor,
        return_rotations: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
```

Batched FK for any humanoid robot.

**Args:**
- `qpos`: `[..., qpos_dim]` - joint configuration
  - `qpos[..., 0:3]`: root position (XYZ)
  - `qpos[..., 3:7]`: root quaternion (WXYZ)
  - `qpos[..., 7:]`: joint angles

**Returns:**
- `positions`: `[..., num_joints, 3]` - joint positions
- `rotations` (optional): `[..., num_joints, 3, 3]` - rotation matrices

```python
    def compute_keypoints(
        self,
        joint_positions: torch.Tensor,
        joint_rotations: torch.Tensor,
        keypoint_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]
```

Compute virtual keypoints from FK results.

**Returns:** Dictionary mapping keypoint name to position tensor.

### keypoints.py

Convenience functions for working with keypoints:

```python
compute_keypoints_from_qpos(
    qpos: torch.Tensor,
    keypoint_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]
```

One-step keypoint computation from qpos.

```python
get_foot_keypoint_positions(qpos: torch.Tensor) -> torch.Tensor
```

Get foot keypoints in order: `[left_heel, left_toe, right_heel, right_toe]`.

**Returns:** `[..., 4, 3]` tensor

```python
get_hand_keypoint_positions(qpos: torch.Tensor) -> torch.Tensor
```

Get hand keypoints in order: `[left_hand, right_hand]`.

**Returns:** `[..., 2, 3]` tensor

## Examples

### Example 1: Visualizing Custom Keypoints

```python
from utils.keypoints import compute_keypoints_from_qpos

# Your motion data
qpos = load_motion_data()  # [B, T, 36]

# Compute all keypoints
keypoints = compute_keypoints_from_qpos(qpos)

# Access specific keypoints
left_hand = keypoints["left_hand"]  # [B, T, 3]
head = keypoints["head"]            # [B, T, 3]

# Visualize in MuJoCo
# (keypoints can be rendered as spheres - see visualize/visualize_fk_contact.py)
```

### Example 2: Foot Contact Detection with Multiple Keypoints

```python
from utils.keypoints import get_foot_keypoint_positions
from utils.kinematics import compute_foot_contact_mask

# Get foot keypoints
foot_positions = get_foot_keypoint_positions(qpos)  # [B, T, 4, 3]

# Compute contact mask
contact_mask = compute_foot_contact_mask(
    foot_positions,
    threshold=0.02,  # 2 cm/s
    fps=30
)  # [B, T-1, 4]

# contact_mask[:, :, 0] = left heel contact
# contact_mask[:, :, 1] = left toe contact
# contact_mask[:, :, 2] = right heel contact
# contact_mask[:, :, 3] = right toe contact
```

### Example 3: Adding a New Robot

1. Create `config/robots/atlas.yaml`:

```yaml
robot:
  name: "Boston Dynamics Atlas"
  mjcf_path: "robots/atlas/atlas.xml"

  root:
    body_name: "pelvis"
    default_height: 0.9

  joints:
    names:
      - "back_bkz"
      - "back_bky"
      - "back_bkx"
      - "l_arm_shz"
      # ... etc
```

2. Use it:

```python
from utils.robot_config import load_robot_config
from utils.kinematics import HumanoidFK

config = load_robot_config("atlas")
fk = HumanoidFK(config, device="cuda")

positions = fk.forward_kinematics(qpos)
```

## Backward Compatibility

All existing code continues to work:

```python
# Old way (still works)
import utils.g1_kinematics as g1_kin
positions = g1_kin.forward_kinematics_g1(qpos)

# New way (recommended)
from utils.kinematics import forward_kinematics_g1  # Wrapper
positions = forward_kinematics_g1(qpos)
```

The `geometric_losses.py` module will continue to use `g1_kinematics.py` until we migrate it.

## Migration Guide

To migrate existing code to use the config system:

### Before:
```python
import utils.g1_kinematics as g1_kin

positions = g1_kin.forward_kinematics_g1(qpos)
foot_pos = g1_kin.extract_foot_positions(positions)
contact = g1_kin.compute_foot_contact_mask(foot_pos)
```

### After:
```python
from utils.keypoints import get_foot_keypoint_positions
from utils.kinematics import compute_foot_contact_mask

foot_pos = get_foot_keypoint_positions(qpos)
contact = compute_foot_contact_mask(foot_pos)
```

## Troubleshooting

### Issue: "Joint 'xyz' not found in MuJoCo model"

**Solution:** Ensure joint names in your YAML config exactly match the joint names in your MuJoCo XML file. Check with:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("path/to/model.xml")
for i in range(model.njnt):
    print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
```

### Issue: "Keypoint parent joint not found"

**Solution:** The `parent_joint` specified in a keypoint config must be one of the joints listed in `robot.joints.names`.

### Issue: FK positions are wrong

**Solution:** Check that:
1. MuJoCo XML path is correct
2. Joint order in config matches expected qpos format
3. Root body name is correct

Run the extraction script to verify:
```bash
python utils/robot_config.py
```

## Future Enhancements

Planned improvements:

1. **Auto-detect joint order** from MuJoCo XML
2. **Full FK rotation tracking** for accurate keypoint orientations
3. **Config-based geometric losses** (`geometric_losses_v2.py`)
4. **Visualization config integration** in `visualize_fk_contact.py`
5. **Multi-robot training** support

## Citation

If you use this config system, please cite:

```bibtex
@misc{humanoid_config_system,
  title={Config-Based Humanoid Robot System for Motion Generation},
  author={Your Name},
  year={2026}
}
```
