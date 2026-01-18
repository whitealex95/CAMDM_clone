"""
Test that keypoints properly rotate with their parent joints.
"""

import torch
import numpy as np
from utils.keypoints import compute_keypoints_from_qpos

# Create two poses: one with ankle at 0 degrees, one rotated
B, T = 1, 2
qpos = torch.zeros(B, T, 36)

# Set root quaternion (identity)
qpos[:, :, 3] = 1.0  # w=1

# Pose 1: Neutral stance
qpos[:, 0, 2] = 0.75  # Height

# Pose 2: Rotate left ankle pitch by 30 degrees (0.524 radians)
qpos[:, 1, 2] = 0.75  # Same height
qpos[:, 1, 11] = 0.524  # left_ankle_pitch_joint (index 5 -> qpos[11])

# Compute keypoints
keypoints = compute_keypoints_from_qpos(qpos)

print("=" * 80)
print("Keypoint Rotation Test")
print("=" * 80)

print("\nPose 1 (Neutral):")
print(f"  Left heel: {keypoints['left_heel'][0, 0].numpy()}")
print(f"  Left toe:  {keypoints['left_toe'][0, 0].numpy()}")

print("\nPose 2 (Ankle pitch rotated 30°):")
print(f"  Left heel: {keypoints['left_heel'][0, 1].numpy()}")
print(f"  Left toe:  {keypoints['left_toe'][0, 1].numpy()}")

# Check that heel and toe moved (rotated)
heel_diff = keypoints['left_heel'][0, 1] - keypoints['left_heel'][0, 0]
toe_diff = keypoints['left_toe'][0, 1] - keypoints['left_toe'][0, 0]

print("\nDifference due to rotation:")
print(f"  Heel moved: {heel_diff.numpy()}")
print(f"  Toe moved:  {toe_diff.numpy()}")

# The heel should move up and back, toe should move down and forward
# when ankle pitches forward (positive rotation)
if abs(heel_diff[2]) > 0.001 and abs(toe_diff[2]) > 0.001:
    print("\n✓ Keypoints properly rotate with parent joint!")
    print(f"  Heel Z change: {heel_diff[2]:.4f} m (should be positive)")
    print(f"  Toe Z change: {toe_diff[2]:.4f} m (should be negative)")
else:
    print("\n✗ Keypoints did NOT rotate properly")
