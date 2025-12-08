# CAMDM MuJoCo Visualization - Quick Start Guide

## 📋 What We've Created

A step-by-step visualization system for your motion generation project with three main capabilities:

1. ✅ **Visualize Training Data** - View and verify motion clips from dataset
2. ⏳ **Visualize Generated Motions** - Compare model outputs with ground truth
3. ⏳ **Interactive Control** - Real-time motion generation with WASD control

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install mujoco numpy scipy
```

### 2. Run Step-by-Step Tests
```bash
python visualize/run_tests.py
```

This will guide you through:
- Step 1: Test MuJoCo model loading
- Step 2: Test motion data loading
- Step 3: Visualize training data

OR run steps individually:

### Step 1: Test MuJoCo Setup
```bash
python visualize/step1_test_mujoco.py
```
**Verify:** Robot appears in viewer, 36 DOF confirmed, joint order correct

### Step 2: Test Data Loader
```bash
python visualize/test_loader.py
```
**Verify:** Loads .pkl files, qpos shape is (36,), no errors

### Step 3: Visualize Training Data
```bash
python visualize/step2_visualize_data.py --dataset lafan1_g1
```
**Verify:** Smooth animations, natural movements, controls work

## 🎮 Controls (Step 3)

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume animation |
| `↑` `↓` | Switch between motion clips |
| `←` `→` | Step through frames (when paused) |
| `R` | Reset to first frame |
| `T` | Toggle trajectory visualization |
| `1-9` | Playback speed (1=0.25x, 5=1x, 9=2x) |
| `S` | Print current status |
| `ESC` | Exit viewer |

## 📁 Files Created

```
visualize/
├── README.md                    # Detailed documentation
├── CHECKLIST.md                 # Verification checklist
├── run_tests.py                 # Automated test runner ⭐
│
├── __init__.py
├── motion_loader.py             # Data loading utilities
├── test_loader.py               # Test data loading
│
├── step1_test_mujoco.py        # Step 1: Test MuJoCo ✅
├── step2_visualize_data.py     # Step 2: View training data ✅
│
└── assets/
    ├── scene.xml                # (You created this)
    └── g1_29dof_rev_1_0.xml    # (You created this)
```

## 🔍 What to Verify at Each Step

### Step 1: MuJoCo Loading
- [ ] Viewer opens without errors
- [ ] G1 robot visible in standing pose
- [ ] Console shows 36 DOF (7 root + 29 joints)
- [ ] All 30 joints listed correctly

### Step 2: Data Loading
- [ ] Finds pickle files in `data/pkls/`
- [ ] Shows dataset summary (number of clips, styles)
- [ ] qpos shape is (36,) for single frame
- [ ] Test passes with ✓ marks

### Step 3: Animation Playback
- [ ] Robot animates smoothly
- [ ] Movements look natural
- [ ] Can pause and step through frames
- [ ] Can switch between different motions
- [ ] No penetration through floor
- [ ] Joints stay within reasonable ranges
- [ ] Forward direction is correct (+X axis)

## 🐛 Troubleshooting

### "Import mujoco could not be resolved"
```bash
pip install mujoco
```

### "Dataset not found"
Generate the pickle files:
```bash
python data/make_pose_data_g1.py
```

### Robot appears upside down / rotated wrong
Check quaternion convention in `motion_loader.py` (should be WXYZ for MuJoCo)

### Robot falls through floor
Uncomment floor geometry in `scene.xml`:
```xml
<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
```

### Joints move strangely
Verify joint order in XML matches data order in `make_pose_data_g1.py`

### Animation too fast/slow
FPS is hardcoded to 30 in `step2_visualize_data.py` - check if data was recorded at different rate

## 📊 Data Format Reference

Your motion data structure:
```python
qpos[0:3]   # Root XYZ position
qpos[3:7]   # Root quaternion (WXYZ for MuJoCo)
qpos[7:36]  # 29 joint angles:
            #   [7:13]   left leg (6 joints)
            #   [13:19]  right leg (6 joints)
            #   [19:22]  waist (3 joints)
            #   [22:29]  left arm (7 joints)
            #   [29:36]  right arm (7 joints)
```

## ✨ Next Steps

After verifying Steps 1-3 work correctly:

### Implement Step 4: Generated Motion Visualization
Create `step3_visualize_generated.py` to:
- Load trained diffusion model from `save/camdm_g1_*/best.pt`
- Generate motions with different style/trajectory conditions
- Compare ground truth vs generated side-by-side
- Quantitative evaluation metrics

### Implement Step 5: Interactive Control
Create `step4_interactive_control.py` to:
- WASD keyboard control for trajectory
- Real-time diffusion sampling (4 steps)
- Style switching with number keys
- Trajectory path overlay visualization

### Additional Features
- Export video recordings (MP4)
- Trajectory deviation plots
- Ground contact force visualization
- Batch evaluation mode
- Multi-model comparison

## Usage Examples

```bash
# View all motions in lafan1_g1 dataset
python visualize/step2_visualize_data.py --dataset lafan1_g1

# View with trajectory visualization
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory

# View 100style dataset
python visualize/step2_visualize_data.py --dataset 100style

# Start from specific motion (index 10) with trajectory
python visualize/step2_visualize_data.py --dataset lafan1_g1 --motion 10 --trajectory

# Custom trajectory length (20 past, 60 future frames)
python visualize/step2_visualize_data.py --dataset lafan1_g1 --trajectory --past-frames 20 --future-frames 60

# Run all tests in sequence
python visualize/run_tests.py
```

## 📞 Getting Help

If you encounter issues:
1. Check `CHECKLIST.md` for verification steps
2. See `README.md` for detailed documentation
3. Review `motion_loader.py` for data format details
4. Check that mesh files exist in `visualize/assets/meshes/`

## 🎯 Current Status

- ✅ Step 1: MuJoCo model loading - **READY TO TEST**
- ✅ Step 2: Motion data loading - **READY TO TEST**
- ✅ Step 3: Training data visualization - **READY TO TEST**
- ⏳ Step 4: Generated motion visualization - **TODO**
- ⏳ Step 5: Interactive control - **TODO**

**You can now run the tests and verify each step works with your data!**
