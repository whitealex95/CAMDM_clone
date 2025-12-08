# Trajectory Visualization Feature

## Overview

The trajectory visualization feature displays the past and future motion trajectories used by the diffusion model for conditioning. This helps you understand:
- Where the character has been (past trajectory)
- Where the character is going (future trajectory)
- How well the trajectory matches the actual motion

## Usage

### Command Line Arguments

```bash
# Enable trajectory visualization at startup
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory

# Customize trajectory length
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory \
    --past-frames 10 --future-frames 45

# Default values match training configuration
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory \
    --past-frames 10 --future-frames 45  # Same as config/default_g1.json
```

### Interactive Control

Press `T` key during playback to toggle trajectory visualization on/off.

## Visualization Details

### Color Coding
- **Blue lines**: Past trajectory (where the character has been)
- **Red lines**: Future trajectory (where the character is going)

### Trajectory Source

Trajectories are extracted from the motion data using the same process as training:
1. Root positions are smoothed using Gaussian filters (kernel sizes 5 and 10)
2. XY positions are extracted (Z comes from actual root height)
3. Forward directions are computed from root orientations
4. Two smoothed versions are stored (using different kernels)

From `make_pose_data_g1.py`:
```python
def extract_traj(root_positions, forward_vectors, kernels=[5, 10]):
    traj_trans, traj_pose = [], []
    canonical_forward = AXIS_FORWARD   # forward = +X in world

    for k in kernels:
        smooth_xy = gaussian_filter1d(root_positions[:, [0, 1]], k, axis=0)
        traj_trans.append(smooth_xy)
        # ... quaternion computation for orientation
```

### Data Format

The trajectory data in pickle files:
```python
motion_dict = {
    "traj": [
        array(T, 2),  # XY positions with kernel=5
        array(T, 2)   # XY positions with kernel=10
    ],
    "traj_pose": [
        array(T, 4),  # WXYZ quaternions with kernel=5
        array(T, 4)   # WXYZ quaternions with kernel=10
    ]
}
```

The visualization uses kernel index 0 (kernel size=5) by default.

## Configuration Alignment

The default values match your training configuration:

**From `config/default_g1.json`:**
```json
{
    "arch": {
        "past_frame": 10,
        "future_frame": 45,
        ...
    }
}
```

**Command line defaults:**
```bash
--past-frames 10
--future-frames 45
```

This ensures you see the exact same trajectory context that the model uses during training.

## Implementation Details

### In `motion_loader.py`

Added `get_trajectory()` method to `MotionData` class:
```python
def get_trajectory(self, frame_idx, past_frames=10, future_frames=45, kernel_idx=0):
    """
    Get past and future trajectory for a specific frame.
    
    Returns:
        past_traj: (past_frames, 3) - XYZ positions of past trajectory
        future_traj: (future_frames, 3) - XYZ positions of future trajectory
        past_orient: (past_frames, 4) - WXYZ quaternions
        future_orient: (future_frames, 4) - WXYZ quaternions
    """
```

### In `step2_visualize_data.py`

1. **Initialization**: Player accepts trajectory visualization parameters
2. **Update loop**: Trajectory is extracted for current frame
3. **Rendering**: Lines are drawn using MuJoCo's geometry API
4. **Interactive toggle**: Press `T` to enable/disable

### Rendering Code

```python
# Draw trajectory lines using MuJoCo scene geoms
for i in range(len(trajectory) - 1):
    p1 = trajectory[i]
    p2 = trajectory[i + 1]
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_LINE,
        ...
    )
```

## Verification Checklist

When trajectory visualization is enabled, verify:

- [ ] Blue lines show smooth path behind the robot
- [ ] Red lines show path ahead of the robot
- [ ] Trajectory matches the direction of robot movement
- [ ] Past trajectory stays connected to current position
- [ ] Future trajectory predicts upcoming motion direction
- [ ] Trajectory appears on ground plane (Z from root height)
- [ ] Toggle (T key) works to show/hide trajectories
- [ ] Different motions show different trajectory patterns

## Use Cases

### 1. Data Verification
Check that trajectories were extracted correctly from motion data:
```bash
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory
```
- Verify trajectories are smooth
- Check they align with actual motion

### 2. Understanding Model Conditioning
See what trajectory information the model receives:
```bash
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory \
    --past-frames 10 --future-frames 45
```
- Past frames provide context
- Future frames specify desired path

### 3. Style Analysis
Compare trajectories across different styles:
```bash
python visualize/step2_visualize_data.py --dataset 100style --trajectory
```
- Different styles may have different trajectory characteristics
- Walk vs run will show different trajectory patterns

## Troubleshooting

### Trajectories not appearing
- Ensure `--trajectory` flag is set or press `T` key
- Check that pickle file contains trajectory data (`traj` and `traj_pose` fields)
- Verify data was processed with `make_pose_data_g1.py`

### Trajectories look wrong
- Check coordinate system (forward = +X axis)
- Verify trajectory extraction uses correct smoothing kernels
- Compare with training data processing in `dataset_g1.py`

### Performance issues
- Reduce `--past-frames` and `--future-frames` values
- Trajectory rendering adds overhead for line drawing

## Next Steps

This trajectory visualization prepares you for:
1. **Step 3**: Visualizing generated motions with trajectory conditioning
2. **Step 4**: Interactive control where you specify trajectories in real-time
3. Understanding how trajectory shapes motion generation

The trajectory you see here is exactly what the diffusion model uses for conditioning during training and inference.
