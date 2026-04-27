"""
HumanoidEnvMotionDataset – G1 dataset with NSM Polar Environment Sensor readings.

Extends the base G1 dataset (``dataset_g1.py``) by loading the augmented pkl
produced by ``visualize/step2_visualize_data_env2d.py --create-dataset``.

Each motion in the pkl must carry:

* ``sensor_readings``      : (T, env_sensor_dim) float32 occupancy values.
* ``traj_trans_per_frame`` : (T, future_frame, 2) float32 — at detour frames
                             the yellow command, at non-detour frames the
                             gt (red) trajectory.
* ``traj_pose_per_frame``  : (T, future_frame, 4) wxyz quaternion, same
                             detour/non-detour split as ``traj_trans_per_frame``.

``__getitem__`` returns the future window of motion plus a condition dict
containing past motion, the per-frame trajectory condition, the current
frame's sensor snapshot, and the style index. Sensor readings and the
trajectory are co-rotated by the same global heading augmentation.
"""

import sys
sys.path.append('./')

import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from network.dataset_g1 import HumanoidMotionDataset


class HumanoidEnvMotionDataset(HumanoidMotionDataset):
    """
    G1 humanoid motion dataset augmented with environment-sensor readings
    and a per-frame command trajectory.
    """

    def __init__(self, pkl_path, rot_req, offset_frame,
                 past_frame, future_frame, dtype=np.float32, limited_num=None,
                 min_start_velocity: float = None,
                 rotation_aug: bool = True,
                 legacy_rotation_aug: bool = False):

        # ---- base class init (loads rotations, traj, styles, etc.) ----
        super().__init__(
            pkl_path, rot_req, offset_frame,
            past_frame, future_frame, dtype=dtype, limited_num=limited_num,
            min_start_velocity=min_start_velocity,
            rotation_aug=rotation_aug,
            legacy_rotation_aug=legacy_rotation_aug,
        )

        data_source = pickle.load(open(pkl_path, "rb"))

        self.env_sensor_dim = int(data_source["env_sensor_dim"])
        sensor_resolution   = int(data_source["sensor_resolution"])
        sensor_max_range    = float(data_source["sensor_max_range"])

        # Recompute per-ring slices using the same formula as EnvironmentSensor.
        ring_slices: list = []
        n_pts = 0
        for z in range(sensor_resolution):
            count = int(round(2.0 * np.pi * z)) if z > 0 else 0
            ring_slices.append((n_pts, n_pts + count))
            n_pts += count
        self._ring_slices = ring_slices

        window_size = past_frame + future_frame

        self.sensor_readings_list      = []
        self.traj_trans_per_frame_list = []
        self.traj_pose_per_frame_list  = []
        for motion in data_source["motions"][:limited_num]:
            N = motion["local_joint_rotations"].shape[0]
            if N < window_size:
                continue
            self.sensor_readings_list.append(
                np.asarray(motion["sensor_readings"], dtype=dtype)
            )
            self.traj_trans_per_frame_list.append(
                np.asarray(motion["traj_trans_per_frame"], dtype=dtype)
            )
            self.traj_pose_per_frame_list.append(
                np.asarray(motion["traj_pose_per_frame"], dtype=dtype)
            )

        print(f"[HumanoidEnvMotionDataset] env_sensor_dim={self.env_sensor_dim}, "
              f"resolution={sensor_resolution}, max_range={sensor_max_range}m, "
              f"{len(self.sensor_readings_list)} clips loaded.")

    # ------------------------------------------------------------------
    # __getitem__ – base-class output + sensor + per-frame trajectory
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        item = self.item_frame_indices[idx]
        motion_idx, frame_ids = item[0], item[1:]

        rotations = self.rotations_list[motion_idx][frame_ids].copy()  # (TW, 30, 4)
        root_pos  = self.root_pos_list[motion_idx][frame_ids].copy()   # (TW, 3)
        root_pos[:, [0, 1]] -= root_pos[self.reference_frame_idx - 1, [0, 1]]

        current_frame = int(frame_ids[self.reference_frame_idx - 1])

        # Per-frame command trajectory (yellow at detour frames, gt at
        # non-detour). Stored in world frame, so centre at the current
        # frame's world XY before passing it on.
        traj_pos = self.traj_trans_per_frame_list[motion_idx][current_frame].copy()
        traj_rot = self.traj_pose_per_frame_list[motion_idx][current_frame].copy()
        ref_xy   = self.root_pos_list[motion_idx][current_frame, :2].astype(traj_pos.dtype)
        traj_pos = traj_pos - ref_xy                                     # (TF, 2)

        sensor_current = self.sensor_readings_list[motion_idx][current_frame].copy()

        # ----------------------------------------------------------
        # GLOBAL ROTATION AUGMENTATION (same as base class)
        # ----------------------------------------------------------
        if self.rotation_aug:
            rot_xyzw     = rotations[..., [1, 2, 3, 0]]
            trajrot_xyzw = traj_rot[..., [1, 2, 3, 0]]

            theta = np.random.uniform(0, 2 * np.pi)
            theta_arr = np.full(rotations.shape[0], theta)
            rot_vec = R.from_rotvec(np.stack([0 * theta_arr, 0 * theta_arr, theta_arr], axis=-1))

            rotations[:, 0] = (
                rot_vec * R.from_quat(rot_xyzw[:, 0])
            ).as_quat()[..., [3, 0, 1, 2]]

            traj_rot = (
                rot_vec[self.reference_frame_idx:] * R.from_quat(trajrot_xyzw)
            ).as_quat()[..., [3, 0, 1, 2]]

            root_pos = rot_vec.apply(root_pos)

            # traj_pos is world-frame XY so it must rotate with everything else.
            if not self.legacy_rotation_aug:
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R2 = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=traj_pos.dtype)
                traj_pos = (R2 @ traj_pos.T).T

            # Rotate the sensor reading to match the new heading.
            # Each ring z has count_z = round(2π*z) spheres uniformly
            # distributed over 2π. A heading rotation of theta shifts ring z's
            # angular index by k_z = -round(theta * count_z / (2π)).
            rotated = sensor_current.copy()
            for (start, end) in self._ring_slices:
                count = end - start
                if count == 0:
                    continue
                k = -int(round(theta * count / (2.0 * np.pi)))
                if k != 0:
                    rotated[start:end] = np.roll(sensor_current[start:end], k)
            sensor_current = rotated

        # ----------------------------------------------------------
        # TORCH CONVERSION (same as base class)
        # ----------------------------------------------------------
        rotations = torch.from_numpy(rotations.astype(self.dtype))  # (TW, 30, 4)
        traj_pos  = torch.from_numpy(traj_pos.astype(self.dtype))   # (TF, 2)
        traj_rot  = torch.from_numpy(traj_rot.astype(self.dtype))   # (TF, 4)
        traj_rot  = self.convert_rot(traj_rot)                       # (TF, per_rot_feat)

        root_quat    = rotations[:, 0]
        root_repr    = self.convert_rot(root_quat)                   # (TW, per_rot_feat)
        joints       = rotations[:, 1:]                              # (TW, 29, 4)
        joints_repr  = self.pad_joint_angle(joints)                  # (TW, 29, per_rot_feat)

        root_repr       = root_repr.unsqueeze(1)                     # (TW, 1, per_rot_feat)
        rotations_full  = torch.cat([root_repr, joints_repr], dim=1) # (TW, 30, per_rot_feat)

        root_pos_pad             = torch.zeros((root_pos.shape[0], 1, self.per_rot_feat))
        root_pos_pad[..., :3]    = torch.from_numpy(root_pos[:, None].astype(self.dtype))
        rotations_w_root         = torch.cat([rotations_full, root_pos_pad], dim=1)  # (TW, 31, per_rot_feat)

        future = rotations_w_root[self.reference_frame_idx:]
        past   = rotations_w_root[:self.reference_frame_idx]

        style_idx_val = float(self.style_set.index(
            self.global_conds["style"][motion_idx]
        ))

        sensor_tensor = torch.from_numpy(sensor_current.astype(self.dtype))  # (env_sensor_dim,)

        return {
            "data": future,
            "conditions": {
                "past_motion": past,               # (TP, 31, per_rot_feat)
                "traj_pose":   traj_rot,           # (TF, per_rot_feat)
                "traj_trans":  traj_pos,           # (TF, 2)
                "sensor":      sensor_tensor,      # (env_sensor_dim,)
                "style":       self.global_conds["style"][motion_idx],
                "style_idx":   style_idx_val,
                "mask":        self.mask,
            },
        }
