"""
HumanoidEnvMotionDataset – G1 dataset with NSM Polar Environment Sensor readings.

Extends the base G1 dataset (dataset_g1.py) by loading pre-computed continuous
occupancy readings from an augmented pkl file (produced by
visualize/step2_visualize_data_env2d.py --create-dataset).

Key addition
------------
* ``sensor_readings_list``: per-clip (T, feature_dim) float32 arrays of
  continuous occupancy values in [0, 1].
* ``__getitem__`` slices the future window of sensor readings and applies the
  same global-rotation-augmentation cyclic shift that is applied to the
  trajectory / pose data.  The shift is applied per-ring (n_r rings × n_theta
  angles) so that the polar grid rotates correctly.

The returned condition dict gains:
    ``sensor``: (future_frames, feature_dim) float32 tensor
which is permuted to (bs, feature_dim, future_frames) in the training loop.
"""

import sys
sys.path.append('./')

import pickle
import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import utils.nn_transforms as nn_transforms
from network.dataset_g1 import HumanoidMotionDataset


class HumanoidEnvMotionDataset(HumanoidMotionDataset):
    """
    G1 humanoid motion dataset augmented with environment-sensor readings.

    The pkl file must have a ``sensor_readings`` field (shape T × n_rays)
    in each motion dict.  If the field is missing for a clip, the sensor is
    assumed to be all-zero (free) for that clip with a warning.

    Args (additions):
        (none – all parameters forwarded to HumanoidMotionDataset)
    """

    def __init__(self, pkl_path, rot_req, offset_frame,
                 past_frame, future_frame, dtype=np.float32, limited_num=None,
                 min_start_velocity: float = None):

        # ---- base class init (loads rotations, traj, styles, etc.) ----
        super().__init__(
            pkl_path, rot_req, offset_frame,
            past_frame, future_frame, dtype=dtype, limited_num=limited_num,
            min_start_velocity=min_start_velocity,
        )

        # ---- load sensor metadata and readings ----
        data_source = pickle.load(open(pkl_path, "rb"))

        # Support new (sensor_resolution / sensor_max_range) and legacy pkls
        self.sensor_feature_dim = int(
            data_source.get("sensor_feature_dim",
            data_source.get("sensor_n_rays", 36))
        )
        sensor_resolution = int(data_source.get("sensor_resolution", 10))
        sensor_max_range  = float(data_source.get("sensor_max_range", 0.5))

        # Recompute per-ring slices using the same formula as EnvironmentSensor
        size     = 2.0 * sensor_max_range
        coverage = 0.5 * size / max(sensor_resolution - 1, 1)
        ring_slices: list = []
        n_pts = 0
        for z in range(sensor_resolution):
            count = int(round(2.0 * np.pi * z)) if z > 0 else 0
            ring_slices.append((n_pts, n_pts + count))
            n_pts += count
        self._ring_slices = ring_slices

        window_size = past_frame + future_frame
        n_missing = 0

        self.sensor_readings_list = []
        for motion in data_source["motions"][:limited_num]:
            N = motion["local_joint_rotations"].shape[0]
            if N < window_size:
                continue
            if "sensor_readings" in motion:
                readings = np.asarray(motion["sensor_readings"], dtype=dtype)
            else:
                n_missing += 1
                readings = np.zeros((N, self.sensor_feature_dim), dtype=dtype)
            self.sensor_readings_list.append(readings)

        if n_missing:
            print(f"[HumanoidEnvMotionDataset] WARNING: {n_missing} clips had no "
                  f"sensor_readings field; using all-zero readings for those clips.")

        print(f"[HumanoidEnvMotionDataset] sensor_feature_dim={self.sensor_feature_dim}, "
              f"resolution={sensor_resolution}, max_range={sensor_max_range}m, "
              f"{len(self.sensor_readings_list)} clips loaded.")

    # ------------------------------------------------------------------
    # __getitem__ – identical to base class but adds sensor condition
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        import random

        item = self.item_frame_indices[idx]
        motion_idx, frame_ids = item[0], item[1:]

        rotations = self.rotations_list[motion_idx][frame_ids].copy()  # (TW, 30, 4)
        root_pos  = self.root_pos_list[motion_idx][frame_ids].copy()   # (TW, 3)

        # Normalize XY
        root_pos[:, [0, 1]] -= root_pos[self.reference_frame_idx - 1, [0, 1]]

        # Randomly choose trajectory version
        traj_rot = self.local_conds["traj_pose"][motion_idx][
            random.choice(self.traj_aug_indexs1)
        ][frame_ids]
        traj_pos = self.local_conds["traj_trans"][motion_idx][
            random.choice(self.traj_aug_indexs2)
        ][frame_ids]

        # Extra trajectory smoothing
        r_aug = np.random.rand()
        if r_aug < 0.75:
            k = 5 if r_aug < 0.5 else 10
            traj_pos = gaussian_filter1d(traj_pos, k, axis=0)

        traj_pos -= traj_pos[self.reference_frame_idx - 1]
        traj_pos = traj_pos[self.reference_frame_idx:]   # (TF, 2)
        traj_rot = traj_rot[self.reference_frame_idx:]   # (TF, 4) wxyz

        # ---- Sensor readings for future frames ----
        sensor_future = self.sensor_readings_list[motion_idx][
            frame_ids[self.reference_frame_idx:]
        ].copy()  # (TF, n_rays)

        # ----------------------------------------------------------
        # GLOBAL ROTATION AUGMENTATION (same as base class)
        # ----------------------------------------------------------
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

        # ---- Rotate sensor readings to match new heading ----
        # Each ring z has count_z = round(2π*z) spheres uniformly distributed
        # over 2π.  A heading rotation of theta shifts ring z's angular index by
        #   k_z = -round(theta * count_z / (2π))
        # We apply the roll per-ring independently (variable count per ring).
        rotated = sensor_future.copy()
        for (start, end) in self._ring_slices:
            count = end - start
            if count == 0:
                continue
            k = -int(round(theta * count / (2.0 * np.pi)))
            if k != 0:
                rotated[:, start:end] = np.roll(sensor_future[:, start:end], k, axis=1)
        sensor_future = rotated

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

        sensor_tensor = torch.from_numpy(sensor_future.astype(self.dtype))  # (TF, n_rays)

        return {
            "data": future,
            "conditions": {
                "past_motion": past,               # (TP, 31, per_rot_feat)
                "traj_pose":   traj_rot,           # (TF, per_rot_feat)
                "traj_trans":  traj_pos,           # (TF, 2)
                "sensor":      sensor_tensor,      # (TF, n_rays)
                "style":       self.global_conds["style"][motion_idx],
                "style_idx":   style_idx_val,
                "mask":        self.mask,
            },
        }
