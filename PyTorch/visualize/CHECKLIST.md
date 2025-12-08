# Visualization Implementation Checklist

## Setup Phase
- [x] Created `visualize/` directory structure
- [x] Created `visualize/assets/` with robot models
- [x] Placed `scene.xml` and `g1_29dof_rev_1_0.xml`

## Step 1: Test MuJoCo Setup
**File:** `visualize/step1_test_mujoco.py`

**To run:**
```bash
python visualize/step1_test_mujoco.py
```

**What to verify:**
- [ ] Script runs without errors
- [ ] MuJoCo viewer window opens
- [ ] Robot appears in viewer (standing pose)
- [ ] Joint order matches expected (29 joints + floating base)
- [ ] qpos size is 36 DOF
- [ ] All joint names are correct

**If issues:**
- Install MuJoCo: `pip install mujoco`
- Check that meshes directory exists in `visualize/assets/meshes/`
- Verify scene.xml path is correct

---

## Step 2: Test Motion Data Loading
**File:** `visualize/test_loader.py`

**To run:**
```bash
python visualize/test_loader.py
```

**What to verify:**
- [ ] Finds .pkl files in `data/pkls/`
- [ ] Loads dataset without errors
- [ ] Shows correct number of motions
- [ ] qpos shape is (36,) for single frame
- [ ] All frames qpos shape is (T, 36)

**If issues:**
- Ensure pickle files exist: run `python data/make_pose_data_g1.py`
- Check data format matches expected structure
- Verify quaternion order (should be WXYZ in qpos)

---

## Step 3: Visualize Training Data
**File:** `visualize/step2_visualize_data.py`

**To run:**
```bash
# Default (lafan1_g1)
python visualize/step2_visualize_data.py

# Or specify dataset
python visualize/step2_visualize_data.py --dataset 100style
```

**What to verify:**
- [ ] Viewer opens and shows robot
- [ ] Animation plays smoothly at 30 FPS
- [ ] Robot movements look natural
- [ ] Can pause/resume with SPACE
- [ ] Can switch motions with UP/DOWN arrows
- [ ] Can step through frames with LEFT/RIGHT arrows
- [ ] Speed control (1-9 keys) works
- [ ] Different motion styles are distinguishable
- [ ] No joint limit violations
- [ ] Root position stays reasonable (doesn't fly away)

**Controls to test:**
- [ ] SPACE (pause/resume)
- [ ] UP/DOWN (switch clips)
- [ ] LEFT/RIGHT (step frames when paused)
- [ ] R (reset)
- [ ] 1-9 (speed control)
- [ ] S (print status)

**Visual checks:**
- [ ] Feet contact ground properly
- [ ] No penetration through floor
- [ ] Joints move within reasonable ranges
- [ ] Forward direction is correct (+X axis)
- [ ] Motions loop smoothly

**If issues:**
- Robot appears in wrong position: Check root position values
- Robot rotated incorrectly: Check quaternion conversion (XYZW vs WXYZ)
- Joints in wrong positions: Verify joint order in XML matches data
- Animation too fast/slow: Check FPS setting (should be 30)
- Robot falls through ground: Add floor geometry to scene.xml

---

## Step 4: Visualize Generated Motions (TODO)
**File:** `visualize/step3_visualize_generated.py` (to be created)

**Requirements:**
- [ ] Trained model checkpoint exists in `save/`
- [ ] Can load model architecture from config
- [ ] Can run inference with trajectory + style conditions

**Features to implement:**
- [ ] Load trained diffusion model
- [ ] Generate motion from random/specified trajectory
- [ ] Side-by-side comparison (ground truth vs generated)
- [ ] Save generated sequences to file
- [ ] Visualize trajectory overlay

---

## Step 5: Interactive Control (TODO)
**File:** `visualize/step4_interactive_control.py` (to be created)

**Requirements:**
- [ ] Real-time diffusion sampling (4 steps)
- [ ] WASD keyboard input handling
- [ ] Trajectory prediction from keyboard input
- [ ] Style selection with number keys

**Features to implement:**
- [ ] W/A/S/D for directional control
- [ ] Number keys (1-9) for style selection
- [ ] Smooth trajectory generation
- [ ] Real-time motion generation
- [ ] Trajectory path visualization
- [ ] Speed control (shift key for running)

---

## Additional Features (Future)

### Visualization Enhancements
- [ ] Draw trajectory path overlay
- [ ] Show ground contact forces
- [ ] Highlight active foot contacts
- [ ] Display current style label
- [ ] Show confidence/quality metrics

### Recording & Analysis
- [ ] Export videos (MP4)
- [ ] Save generated sequences
- [ ] Batch visualization mode
- [ ] Comparison mode (multiple models)
- [ ] Quantitative metrics display

### Debugging Tools
- [ ] Joint angle plots
- [ ] Velocity/acceleration visualization
- [ ] Trajectory deviation visualization
- [ ] Style embedding visualization

---

## Testing Notes

### Date: _________

**Step 1 Results:**
- Status: ☐ Pass ☐ Fail
- Notes:

**Step 2 (Data Loader) Results:**
- Status: ☐ Pass ☐ Fail
- Notes:

**Step 3 (Visualize Data) Results:**
- Status: ☐ Pass ☐ Fail
- Dataset tested: _________
- Motions look correct: ☐ Yes ☐ No
- Issues found:

**Next Actions:**
