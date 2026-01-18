"""
Verify that our FK implementation matches MuJoCo's FK.
"""

import numpy as np
import torch
import mujoco
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.g1_kinematics as g1_kin

# Load MuJoCo model
scene_path = "visualize/assets/scene.xml"
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

print("="*80)
print("FK Verification: Comparing our implementation with MuJoCo")
print("="*80)
print()

# Test 1: Zero configuration
print("Test 1: Zero configuration")
print("-"*80)
qpos_zero = np.zeros(36)
qpos_zero[3] = 1.0  # quaternion w=1

# MuJoCo FK
data.qpos[:] = qpos_zero
mujoco.mj_forward(model, data)
mujoco_positions = data.xpos[1:31].copy()  # Skip world, take bodies 1-30

# Our FK
qpos_torch = torch.from_numpy(qpos_zero).float().unsqueeze(0)  # [1, 36]
our_positions = g1_kin.forward_kinematics_g1(qpos_torch).squeeze(0).numpy()  # [30, 3]

# Compare
max_error = np.abs(mujoco_positions - our_positions).max()
mean_error = np.abs(mujoco_positions - our_positions).mean()

print(f"Max error:  {max_error:.6f} m")
print(f"Mean error: {mean_error:.6f} m")

if max_error < 0.01:  # 1cm tolerance
    print("✓ PASS: FK matches MuJoCo within 1cm")
else:
    print("✗ FAIL: FK does not match MuJoCo")
    print("\nDetailed comparison:")
    for i in range(min(30, mujoco_positions.shape[0])):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i+1)
        mj_pos = mujoco_positions[i]
        our_pos = our_positions[i]
        error = np.linalg.norm(mj_pos - our_pos)
        if error > 0.001:  # Print if error > 1mm
            print(f"{i:2d}. {body_name:30s} MJ={mj_pos} Our={our_pos} Error={error:.6f}")

print()

# Test 2: Random configuration
print("Test 2: Random configuration")
print("-"*80)
np.random.seed(42)
qpos_random = np.random.randn(36) * 0.5
qpos_random[:3] = [0, 0, 0.8]  # Root position
qpos_random[3:7] = [1, 0, 0, 0]  # Root quaternion

# MuJoCo FK
data.qpos[:] = qpos_random
mujoco.mj_forward(model, data)
mujoco_positions = data.xpos[1:31].copy()

# Our FK
qpos_torch = torch.from_numpy(qpos_random).float().unsqueeze(0)
our_positions = g1_kin.forward_kinematics_g1(qpos_torch).squeeze(0).numpy()

# Compare
max_error = np.abs(mujoco_positions - our_positions).max()
mean_error = np.abs(mujoco_positions - our_positions).mean()

print(f"Max error:  {max_error:.6f} m")
print(f"Mean error: {mean_error:.6f} m")

if max_error < 0.01:
    print("✓ PASS: FK matches MuJoCo within 1cm")
else:
    print("✗ FAIL: FK does not match MuJoCo")
    print("\nDetailed comparison (errors > 1mm):")
    for i in range(min(30, mujoco_positions.shape[0])):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i+1)
        mj_pos = mujoco_positions[i]
        our_pos = our_positions[i]
        error = np.linalg.norm(mj_pos - our_pos)
        if error > 0.001:
            print(f"{i:2d}. {body_name:30s} Error={error:.6f} MJ={mj_pos} Our={our_pos}")

print()
print("="*80)
print("Verification complete")
print("="*80)
