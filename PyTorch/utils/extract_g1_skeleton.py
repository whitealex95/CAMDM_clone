"""
Extract G1 skeleton parameters directly from MuJoCo model.

This script uses MuJoCo's forward kinematics to extract the correct
parent indices, offsets, and axes for the G1 humanoid robot.
"""

import mujoco
import numpy as np
import os

# Load MuJoCo model
scene_path = "visualize/assets/scene.xml"
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

# Get joint names from qpos
print("="*60)
print("G1 Robot Joint Structure")
print("="*60)
print()

# The first 7 DOF are the floating base (free joint)
# qpos layout: [x, y, z, qw, qx, qy, qz, joint1_angle, ..., joint29_angle]

print("Total qpos dimension:", model.nq)
print("Total joints (including free):", model.njnt)
print("Number of bodies:", model.nbody)
print()

# Print joint information
print("Joint Information:")
print("-"*60)
for i in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    jnt_qposadr = model.jnt_qposadr[i]
    jnt_dofadr = model.jnt_dofadr[i]

    type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
    print(f"{i:2d}. {jnt_name:30s} type={type_names.get(jnt_type, jnt_type)} qposadr={jnt_qposadr} dofadr={jnt_dofadr}")

print()

# Print body information and parent relationships
print("Body Information:")
print("-"*60)
body_to_joint = {}
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    parent_id = model.body_parentid[i]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id >= 0 else "world"
    body_pos = model.body_pos[i]
    body_quat = model.body_quat[i]  # (w, x, y, z)

    # Find joint attached to this body
    jnt_id = -1
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == i:
            jnt_id = j
            break

    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) if jnt_id >= 0 else "none"

    print(f"{i:2d}. {body_name:30s} parent={parent_id:2d} ({parent_name:20s}) pos={body_pos} quat={body_quat}")
    if jnt_id >= 0:
        print(f"     -> joint: {jnt_name}")
        body_to_joint[i] = jnt_id

print()

# Extract skeleton for the 29 actuated joints
print("Extracting skeleton for 29 actuated joints:")
print("-"*60)

joint_names_ordered = []
parent_indices = []
offsets = []
axes = []

# Start from joint 1 (skip the free joint at index 0)
for i in range(1, model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_names_ordered.append(jnt_name)

    # Get body this joint is attached to
    body_id = model.jnt_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    # Get parent body
    parent_body_id = model.body_parentid[body_id]

    # Find which joint corresponds to parent body (or -1 if root)
    parent_joint_idx = -1
    if parent_body_id > 0:  # Not world, not pelvis
        # Find joint attached to parent body
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == parent_body_id:
                # Convert to our joint indexing (subtract 1 for free joint)
                parent_joint_idx = j - 1 if j > 0 else -1
                break

    parent_indices.append(parent_joint_idx)

    # Get offset (body position relative to parent)
    offset = model.body_pos[body_id]
    offsets.append(offset)

    # Get joint axis
    axis = model.jnt_axis[i]
    axes.append(axis)

    parent_jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, parent_joint_idx + 1) if parent_joint_idx >= 0 else "root"
    print(f"{i-1:2d}. {jnt_name:30s} parent={parent_joint_idx:2d} ({parent_jnt_name:20s}) offset={offset} axis={axis}")

print()
print("="*60)
print("Python arrays for g1_kinematics.py:")
print("="*60)
print()

# Print as Python code
print("JOINT_NAMES = [")
for name in joint_names_ordered:
    print(f'    "{name}",')
print("]")
print()

print("JOINT_PARENTS = np.array([")
print("    -1,  # 0: root (pelvis) - no parent")
for i, (name, parent) in enumerate(zip(joint_names_ordered, parent_indices), 1):
    comment = f"# {i}: {name}"
    print(f"    {parent:2d},  {comment}")
print("], dtype=np.int32)")
print()

print("JOINT_OFFSETS = np.array([")
print("    [0.0, 0.0, 0.0],  # 0: root (no offset)")
for i, (name, offset) in enumerate(zip(joint_names_ordered, offsets), 1):
    comment = f"# {i}: {name}"
    print(f"    [{offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f}],  {comment}")
print("], dtype=np.float32)")
print()

print("JOINT_AXES = np.array([")
print("    [0, 0, 0],  # 0: root (free joint, not used)")
for i, (name, axis) in enumerate(zip(joint_names_ordered, axes), 1):
    comment = f"# {i}: {name}"
    # Round to avoid floating point noise
    axis_clean = [int(round(a)) if abs(a) > 0.5 else 0 for a in axis]
    print(f"    [{axis_clean[0]}, {axis_clean[1]}, {axis_clean[2]}],  {comment}")
print("], dtype=np.float32)")
print()

# Verify with FK test
print("="*60)
print("Verification: Testing FK with zero configuration")
print("="*60)

# Reset to zero configuration
data.qpos[:] = 0
data.qpos[3] = 1.0  # Set quaternion w=1
mujoco.mj_forward(model, data)

print("\nBody positions from MuJoCo FK (zero config):")
for i in range(min(15, model.nbody)):  # Print first 15 bodies
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_xpos = data.xpos[i]  # Global position
    print(f"{i:2d}. {body_name:30s} pos={body_xpos}")
