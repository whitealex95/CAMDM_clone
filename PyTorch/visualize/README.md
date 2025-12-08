# CAMDM Motion Visualization Guide

This guide walks through creating MuJoCo visualizations for the CAMDM motion generation project.

## Overview

Three types of visualization:
1. **Training Data Visualization** - View motion clips from dataset
2. **Generated Motion Visualization** - View model-generated motions
3. **Interactive Control** - Real-time motion generation with WASD control

## Prerequisites

```bash
pip install mujoco numpy scipy
```

## File Structure

```
visualize/
├── __init__.py
├── motion_loader.py           # Data loading utilities
├── step1_test_mujoco.py       # Step 1: Test MuJoCo setup
├── step2_visualize_data.py    # Step 2: Visualize training data
├── step3_visualize_generated.py    # Step 3: Visualize generated motions
├── step4_interactive_control.py    # Step 4: Interactive WASD control
└── assets/
    ├── scene.xml              # MuJoCo scene
    └── g1_29dof_rev_1_0.xml   # G1 robot model
```

## Step-by-Step Usage

### Step 1: Test MuJoCo Setup ✓

Verify that MuJoCo can load your robot model correctly.

```bash
python visualize/step1_test_mujoco.py
```

**What it does:**
- Loads the G1 robot model from `scene.xml`
- Verifies joint names and order match expected configuration
- Confirms qpos size is 36 DOF (7 root + 29 joints)
- Opens viewer showing robot in neutral pose

**Expected output:**
- Joint order verification
- Model statistics (36 DOF)
- Interactive viewer window

---

### Step 2: Visualize Training Data ✓

Play back actual motion clips from your training dataset.

```bash
# Visualize Lafan1 G1 dataset
python visualize/step2_visualize_data.py --dataset lafan1_g1

# Visualize 100STYLE dataset
python visualize/step2_visualize_data.py --dataset 100style

# Start from specific motion
python visualize/step2_visualize_data.py --dataset lafan1_g1 --motion 5
```

**What it does:**
- Loads motion clips from `.pkl` files
- Plays animations at 30 FPS
- Allows browsing through different clips and styles

**Controls:**
- `SPACE`: Pause/Resume
- `UP/DOWN`: Switch between motion clips
- `LEFT/RIGHT`: Step through frames (when paused)
- `R`: Reset to first frame
- `T`: Toggle trajectory visualization on/off
- `1-9`: Change playback speed (1=0.25x, 5=1x, 9=2x)
- `S`: Print current status
- `ESC`: Exit

**Trajectory Visualization:**
- Blue line: Past trajectory (default 10 frames)
- Red line: Future trajectory (default 45 frames)
- Toggle on/off with `T` key or use `--trajectory` flag at startup

**What to verify:**
- Motions look natural and smooth
- No joint limit violations
- Root position/orientation correct
- Different styles are distinguishable
- Trajectories align with robot motion (if enabled)

---

### Step 3: Visualize Generated Motions (TODO)

Load a trained model and visualize generated motions.

```bash
python visualize/step3_visualize_generated.py \
    --checkpoint save/camdm_g1_lafan1_g1/best.pt \
    --dataset lafan1_g1 \
    --style walk
```

**What it will do:**
- Load trained diffusion model
- Generate motions conditioned on trajectory + style
- Compare ground truth vs generated side-by-side
- Save generated sequences

---

### Step 4: Interactive Control (TODO)

Real-time motion generation with keyboard control.

```bash
python visualize/step4_interactive_control.py \
    --checkpoint save/camdm_g1_lafan1_g1/best.pt
```

**What it will do:**
- WASD controls for trajectory (W=forward, S=back, A=left, D=right)
- Real-time diffusion sampling (using 4 steps for speed)
- Number keys (1-9) to switch styles
- Generate smooth responsive motions

---

## Data Format Reference

From `data/make_pose_data_g1.py`:

### Joint Order (30 total)
```
0: root_joint (7 DOF: XYZ + XYZW quaternion)
1-6: left leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
7-12: right leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
13-15: waist (yaw, roll, pitch)
16-22: left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
23-29: right arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
```

### Pickle File Structure
```python
{
    "parents": None,
    "offsets": None,
    "names": None,
    "motions": [
        {
            "filepath": str,
            "local_joint_rotations": (T, 30, 4),  # T frames, 30 joints
            "global_root_positions": (T, 3),
            "traj": [...],  # trajectory translations
            "traj_pose": [...],  # trajectory orientations
            "style": str,
            "text": str
        }
    ]
}
```

### MuJoCo qpos Format (36 DOF)
```
[0:3]   - root position XYZ
[3:7]   - root quaternion WXYZ (MuJoCo convention)
[7:36]  - 29 joint angles (1 DOF each)
```

## Troubleshooting

### MuJoCo Import Error
```bash
pip install mujoco
```

### Missing Dataset
Ensure `.pkl` files exist in `data/pkls/`:
```bash
python data/make_pose_data_g1.py
```

### Robot Falls Through Ground
Check that `scene.xml` has proper floor collision geometry.

### Joint Limits Violated
Verify joint ranges in `g1_29dof_rev_1_0.xml` match training data ranges.

### Quaternion Convention Issues
- Training data uses **XYZW** format (stored as WXYZ after conversion)
- MuJoCo uses **WXYZ** format
- Conversion handled in `motion_loader.py`

## Next Steps

After verifying Steps 1-2:
1. Implement Step 3 (generated motion visualization)
2. Implement Step 4 (interactive control)
3. Add trajectory overlay visualization
4. Add ground contact force indicators
5. Export video rendering functionality
