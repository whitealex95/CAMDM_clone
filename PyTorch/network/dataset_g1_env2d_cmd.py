"""
HumanoidEnvCmdMotionDataset – like HumanoidEnvMotionDataset, but replaces
the per-frame trajectory condition with a single body-frame twist command
``(vx, vy, omega)`` summarising the future window.

The base dataset stores ``traj_trans_per_frame`` and ``traj_pose_per_frame``
already canonicalised to the robot-local (yaw-aligned) frame at the
current frame, with the current frame at the origin facing ``+X``. We
collapse those per-frame paths into a single twist:

    vx, vy   = traj_trans_per_frame[current, -1] / (future_frame * frame_dt)
    omega    = yaw_of(traj_pose_per_frame[current, -1]) / (future_frame * frame_dt)

i.e. average body-frame velocity over the future window. The yaw is
taken from the final-frame quaternion (already in the robot-local frame,
so it's the yaw delta from 0).

Returned ``conditions`` dict drops the per-frame ``traj_pose`` and
``traj_trans`` keys; everything else (past motion, sensor, style, mask)
matches the env2d dataset.
"""

import numpy as np
import torch

from network.dataset_g1_env2d import HumanoidEnvMotionDataset


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Yaw angle (radians) of a wxyz quaternion, in [-pi, pi]."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class HumanoidEnvCmdMotionDataset(HumanoidEnvMotionDataset):
    """
    Drop-in replacement for HumanoidEnvMotionDataset that exposes a
    single ``command`` instead of per-frame trajectory.
    """

    DEFAULT_FRAME_DT = 1.0 / 30.0   # lafan1 is 30 fps

    def __init__(self, *args, frame_dt: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_dt = float(frame_dt) if frame_dt is not None else self.DEFAULT_FRAME_DT
        # Base class stores past_frame but not future_frame explicitly; derive
        # it from the per-future-frame mask, which is shape (TF,).
        self.future_frame = int(self.mask.shape[0])
        self.window_seconds = float(self.future_frame) * self.frame_dt
        print(
            f"[HumanoidEnvCmdMotionDataset] frame_dt={self.frame_dt:.4f}s, "
            f"future_frame={self.future_frame}, "
            f"window={self.window_seconds:.3f}s -> command=(vx,vy,omega) per clip"
        )

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        cond = item["conditions"]

        # The base class already canonicalised traj_pose / traj_trans
        # into the robot-local frame at the current frame (current frame
        # at origin, facing +X). Both arrays are NOT rotated by the
        # global heading augmentation. Convert to body-frame twist.
        traj_trans = cond["traj_trans"]            # (TF, 2)   torch
        traj_pose  = cond["traj_pose"]             # (TF, F)   torch (rot repr)

        # Recover the final-frame world XY from the rotated/canonicalised
        # storage. traj_trans is already body-frame XY relative to the
        # current frame, so we can use it directly.
        last_xy = traj_trans[-1].detach().cpu().numpy().astype(np.float32)  # (2,)

        # For yaw, the converted rotation in traj_pose can be 6d/quaternion/etc.
        # The simplest source of yaw is the raw per-frame quaternion stored
        # by the base class. Re-derive it here.
        traj_quat = self.traj_pose_per_frame_list[
            int(self.item_frame_indices[idx][0])
        ][int(self.item_frame_indices[idx][self.reference_frame_idx])]  # (TF, 4) wxyz
        final_yaw = _yaw_from_quat_wxyz(traj_quat[-1])                   # scalar

        vx = last_xy[0] / self.window_seconds
        vy = last_xy[1] / self.window_seconds
        omega = final_yaw / self.window_seconds

        command = torch.tensor([vx, vy, omega], dtype=traj_trans.dtype)  # (3,)

        cond = dict(cond)        # shallow copy to avoid mutating cached dicts
        cond.pop("traj_pose", None)
        cond.pop("traj_trans", None)
        cond["command"] = command

        return {"data": item["data"], "conditions": cond}
