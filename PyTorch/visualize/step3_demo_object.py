import os
import sys
import time
import pickle
import argparse
from collections import deque

import numpy as np
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.common as common
import utils.nn_transforms as nn_transforms
from diffusion.create_diffusion import create_gaussian_diffusion
from network.models_object import MotionDiffusionObject

from visualize.utils.geometry import draw_trajectory
from visualize.utils.trajectory import align_trajectory_to_pose, blend_trajectory, extend_future_traj_heusristic
from visualize.utils.trajectory import match_future_horizon


def quat_wxyz_to_mat(q):
    return R.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def quat_wxyz_to_yaw_mat(q):
    yaw = R.from_quat(q[[1, 2, 3, 0]]).as_euler("zyx")[0]
    return R.from_euler("z", yaw).as_matrix()


def mat_to_quat_wxyz(m):
    q = R.from_matrix(m).as_quat()  # xyzw
    return q[[3, 0, 1, 2]]


def compute_object_pose_relative_frame(qpos43, frame_mode="root"):
    root_pos = qpos43[:3]
    root_q = qpos43[3:7]
    obj_pos = qpos43[36:39]
    obj_q = qpos43[39:43]

    if frame_mode != "root_yaw_gravity_aligned":
        raise ValueError("Unsupported object_pose_relative_frame. Expected 'root_yaw_gravity_aligned'.")
    root_r = quat_wxyz_to_yaw_mat(root_q)
    obj_r = quat_wxyz_to_mat(obj_q)
    root_r_inv = root_r.T
    p_rel = root_r_inv @ (obj_pos - root_pos)
    r_rel = root_r_inv @ obj_r
    return np.concatenate([p_rel, r_rel.reshape(-1)], axis=0)


def object_relative_to_world(root_pos, root_q_wxyz, obj_rel_12, frame_mode="root"):
    p_rel = obj_rel_12[:3]
    r_rel = obj_rel_12[3:].reshape(3, 3)
    if frame_mode != "root_yaw_gravity_aligned":
        raise ValueError("Unsupported object_pose_relative_frame. Expected 'root_yaw_gravity_aligned'.")
    root_r = quat_wxyz_to_yaw_mat(root_q_wxyz)
    obj_pos = root_pos + root_r @ p_rel
    obj_r = root_r @ r_rel
    obj_q = mat_to_quat_wxyz(obj_r)
    return obj_pos, obj_q


def qpos36_to_body_state(qpos36):
    """(T,36) -> (T,186) where 186=(31*6)."""
    T = qpos36.shape[0]
    out = np.zeros((T, 31, 6), dtype=np.float32)
    for t in range(T):
        root_pos = qpos36[t, :3]
        root_q = qpos36[t, 3:7]
        joints = qpos36[t, 7:]
        root_6d = nn_transforms.quat2repr6d(torch.from_numpy(root_q).float().unsqueeze(0)).numpy()[0]
        out[t, 0, :] = root_6d
        out[t, 1:30, 0] = joints
        out[t, 30, :3] = root_pos
    return out.reshape(T, -1)


def body_state_to_qpos36(body_state):
    """(T,186) -> (T,36)."""
    T = body_state.shape[0]
    in_view = body_state.reshape(T, 31, 6)
    qpos = np.zeros((T, 36), dtype=np.float32)
    for t in range(T):
        root_6d = in_view[t, 0, :]
        joints = in_view[t, 1:30, 0]
        root_pos = in_view[t, 30, :3]
        root_q = nn_transforms.repr6d2quat(torch.from_numpy(root_6d).float().unsqueeze(0)).numpy()[0]
        qpos[t, :3] = root_pos
        qpos[t, 3:7] = root_q
        qpos[t, 7:] = joints
    return qpos


class ObjectModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, **kwargs):
        return self.model.forward(
            x, timesteps,
            kwargs.get("past_motion"),
            kwargs.get("traj_pose"),
            kwargs.get("traj_trans"),
            kwargs.get("traj_contact"),
            kwargs.get("style_idx"),
            kwargs.get("traj_obj_pose"),
            kwargs.get("traj_obj_trans"),
        )


class ObjectMotionData:
    def __init__(self, motion_dict):
        self.filepath = motion_dict.get("filepath", "")
        self.style = motion_dict.get("style", "unknown")
        self.text = motion_dict.get("text", self.style)
        self.local_joint_rotations = motion_dict["local_joint_rotations"].astype(np.float32)
        self.global_root_positions = motion_dict["global_root_positions"].astype(np.float32)
        self.traj = motion_dict["traj"]
        self.traj_pose = motion_dict["traj_pose"]
        self.object_pose_relative = np.asarray(motion_dict.get("object_pose_relative", None), dtype=np.float32)
        self.object_contact_mask = np.asarray(motion_dict.get("object_contact_mask", None), dtype=np.float32).reshape(-1, 1)
        self.object_pose_relative_frame = motion_dict.get("object_pose_relative_frame", None)
        if self.object_pose_relative_frame != "root_yaw_gravity_aligned":
            raise ValueError(
                "Expected 'object_pose_relative_frame' == 'root_yaw_gravity_aligned'. "
                "Re-generate merged_object_motion.pkl with the latest script."
            )
        if "obj_traj" not in motion_dict or "obj_traj_pose" not in motion_dict:
            raise KeyError(
                "Missing required keys: 'obj_traj' and 'obj_traj_pose'. "
                "Re-generate merged_object_motion.pkl with the latest script."
            )
        self.obj_traj = motion_dict["obj_traj"]
        self.obj_traj_pose = motion_dict["obj_traj_pose"]
        self.has_object = bool(motion_dict.get("has_object", False))
        self.num_frames = len(self.global_root_positions)

        self.obj_traj = [np.asarray(self.obj_traj[0], dtype=np.float32), np.asarray(self.obj_traj[1], dtype=np.float32)]
        self.obj_traj_pose = [np.asarray(self.obj_traj_pose[0], dtype=np.float32), np.asarray(self.obj_traj_pose[1], dtype=np.float32)]
        if self.obj_traj[0].shape[-1] != 3 or self.obj_traj[1].shape[-1] != 3:
            raise ValueError(
                f"obj_traj must be xyz (last dim=3), got {[self.obj_traj[0].shape[-1], self.obj_traj[1].shape[-1]]}. "
                "Re-generate merged_object_motion.pkl with latest script."
            )

    def get_qpos43(self, frame_idx):
        i = min(frame_idx, self.num_frames - 1)
        root_pos = self.global_root_positions[i]
        root_q = self.local_joint_rotations[i, 0]
        joints = self.local_joint_rotations[i, 1:, 0]
        qpos36 = np.concatenate([root_pos, root_q, joints], axis=0)
        if self.has_object:
            obj_pos, obj_q = object_relative_to_world(
                root_pos, root_q, self.object_pose_relative[i], frame_mode=self.object_pose_relative_frame
            )
        else:
            obj_pos = np.array([0.0, 0.0, -10.0], dtype=np.float32)
            obj_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([qpos36, obj_pos, obj_q], axis=0).astype(np.float32)

    def get_contact(self, frame_idx):
        i = min(frame_idx, self.num_frames - 1)
        return float(self.object_contact_mask[i, 0])

    def get_trajectory(self, frame_idx, past_frames=10, future_frames=45):
        traj_xy = self.traj[0]
        traj_q = self.traj_pose[0]
        ps = max(0, frame_idx - past_frames)
        pe = frame_idx
        fs = frame_idx
        fe = min(self.num_frames, frame_idx + future_frames)

        past_xy = traj_xy[ps:pe]
        fut_xy = traj_xy[fs:fe]
        past_q = traj_q[ps:pe]
        fut_q = traj_q[fs:fe]

        past = np.concatenate([past_xy, np.zeros((len(past_xy), 1), dtype=np.float32)], axis=-1)
        fut = np.concatenate([fut_xy, np.zeros((len(fut_xy), 1), dtype=np.float32)], axis=-1)
        return past, fut, past_q, fut_q

    def get_object_trajectory(self, frame_idx, past_frames=10, future_frames=45):
        obj_xyz = self.obj_traj[0]
        obj_q = self.obj_traj_pose[0]
        fs = frame_idx
        fe = min(self.num_frames, frame_idx + future_frames)
        return obj_xyz[fs:fe], obj_q[fs:fe]


class ObjectMotionDataset:
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            src = pickle.load(f)
        self.motions = [ObjectMotionData(m) for m in src["motions"]]
        self.styles = sorted(list(set(m.style for m in self.motions)))
        self.style_to_motions = {s: [] for s in self.styles}
        for i, m in enumerate(self.motions):
            self.style_to_motions[m.style].append(i)
            m.style_idx = self.styles.index(m.style)
        print(f"Loaded {len(self.motions)} motions, styles={self.styles}")

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return self.motions[idx]


class MotionGeneratorObject:
    def __init__(self, model, diffusion, config, device="cuda", sampler="ddpm", cfg_scale=1.0):
        self.model = ObjectModelWrapper(model)
        self.diffusion = diffusion
        self.device = device
        self.sampler = sampler.lower()
        self.cfg_scale = float(cfg_scale)

        self.future_frames = config.arch.future_frame
        self.rot_req = config.arch.rot_req
        self.state_dim = 199
        self.body_dim = 186

    def generate_motion(
        self, past_qpos43, past_contact, traj_trans, traj_pose, traj_contact,
        traj_obj_trans, traj_obj_pose, style_idx, cfg_scale=None,
        frame_mode="root_yaw_gravity_aligned",
        has_object=True,
    ):
        curr_root_xy = past_qpos43[-1, :2].copy()

        # center XY as in training
        past_q = past_qpos43.copy()
        past_q[:, [0, 1]] -= curr_root_xy[None, :]
        past_q[:, [36, 37]] -= curr_root_xy[None, :]

        past_body = qpos36_to_body_state(past_q[:, :36])         # [Tp,186]
        if has_object:
            past_obj_rel = np.stack([compute_object_pose_relative_frame(x, frame_mode=frame_mode) for x in past_q], axis=0)  # [Tp,12]
        else:
            # Match walk padding used in training data.
            past_obj_rel = np.zeros((past_q.shape[0], 12), dtype=np.float32)
        past_state = np.concatenate([past_body, past_obj_rel, past_contact.reshape(-1, 1)], axis=-1)  # [Tp,199]
        past_state = past_state[..., None]                        # [Tp,199,1]

        traj_trans_centered = traj_trans.copy()
        traj_trans_centered -= curr_root_xy[None, :]

        traj_pose_repr = nn_transforms.get_rotation(torch.from_numpy(traj_pose).float(), self.rot_req).numpy()
        if has_object:
            traj_obj_pose_repr = nn_transforms.get_rotation(torch.from_numpy(traj_obj_pose).float(), self.rot_req).numpy()
            traj_obj_trans_centered = traj_obj_trans.copy()
            traj_obj_trans_centered[:, :2] -= curr_root_xy[None, :]
        else:
            # Neutral object condition for walk style.
            traj_obj_trans_centered = np.zeros((traj_trans_centered.shape[0], 3), dtype=np.float32)
            id_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (traj_pose.shape[0], 1))
            traj_obj_pose_repr = nn_transforms.get_rotation(torch.from_numpy(id_quat).float(), self.rot_req).numpy()

        past_motion_t = torch.from_numpy(past_state).float().unsqueeze(0).permute(0, 2, 3, 1).to(self.device)
        traj_trans_t = torch.from_numpy(traj_trans_centered).float().unsqueeze(0).permute(0, 2, 1).to(self.device)
        traj_pose_t = torch.from_numpy(traj_pose_repr).float().unsqueeze(0).permute(0, 2, 1).to(self.device)
        traj_obj_pose_t = torch.from_numpy(traj_obj_pose_repr).float().unsqueeze(0).permute(0, 2, 1).to(self.device)
        traj_obj_trans_t = torch.from_numpy(traj_obj_trans_centered).float().unsqueeze(0).permute(0, 2, 1).to(self.device)
        traj_contact_t = torch.from_numpy(traj_contact.astype(np.float32)).float().reshape(1, -1, 1).permute(0, 2, 1).to(self.device)
        style_idx_t = torch.tensor([style_idx], device=self.device)

        model_kwargs = {
            "past_motion": past_motion_t,
            "traj_trans": traj_trans_t,
            "traj_pose": traj_pose_t,
            "traj_contact": traj_contact_t,
            "traj_obj_pose": traj_obj_pose_t,
            "traj_obj_trans": traj_obj_trans_t,
            "style_idx": style_idx_t,
            "y": {},
        }
        uncond_kwargs = {
            "past_motion": torch.zeros_like(past_motion_t),
            "traj_trans": traj_trans_t,
            "traj_pose": traj_pose_t,
            "traj_contact": torch.zeros_like(traj_contact_t),
            "traj_obj_pose": torch.zeros_like(traj_obj_pose_t),
            "traj_obj_trans": torch.zeros_like(traj_obj_trans_t),
            "style_idx": style_idx_t,
            "y": {},
        }

        guidance_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if guidance_scale == 1.0:
            sampling_model = self.model
            sampling_kwargs = model_kwargs
        else:
            class _CFGWrapper(torch.nn.Module):
                def __init__(self, cond_model, uncond_kwargs, scale):
                    super().__init__()
                    self.cond_model = cond_model
                    self.uncond_kwargs = uncond_kwargs
                    self.scale = scale

                def forward(self, x, timesteps, **kwargs):
                    pred_cond = self.cond_model(x, timesteps, **kwargs)
                    pred_uncond = self.cond_model(x, timesteps, **self.uncond_kwargs)
                    return pred_uncond + self.scale * (pred_cond - pred_uncond)

            sampling_model = _CFGWrapper(self.model, uncond_kwargs, guidance_scale)
            sampling_kwargs = model_kwargs

        shape = (1, self.state_dim, 1, self.future_frames)
        if self.sampler == "ddim":
            sample = self.diffusion.ddim_sample_loop(
                sampling_model, shape, clip_denoised=False, model_kwargs=sampling_kwargs,
                progress=False, eta=0.0, device=self.device
            )
        else:
            sample = self.diffusion.p_sample_loop(
                sampling_model, shape, clip_denoised=False, model_kwargs=sampling_kwargs,
                progress=False, device=self.device
            )

        sample_np = sample.squeeze(0).permute(2, 0, 1).cpu().numpy()[:, :, 0]  # [Tf,199]
        pred_body = sample_np[:, :self.body_dim]
        pred_obj_rel = sample_np[:, self.body_dim:self.body_dim + 12]
        pred_contact = sample_np[:, self.body_dim + 12:self.body_dim + 13]

        pred_qpos36 = body_state_to_qpos36(pred_body)
        pred_qpos36[:, :2] += curr_root_xy[None, :]
        if not has_object:
            pred_obj_rel[:] = 0.0
            pred_contact[:] = 0.0
        return pred_qpos36.astype(np.float32), pred_obj_rel.astype(np.float32), pred_contact.squeeze(-1).astype(np.float32)


class DemoPlayerObject:
    def __init__(
        self, model, data, dataset, motion_generator, show_trajectory=True,
        past_frames=10, future_frames=45, applyframes=15, cfg_count=2,
        traj_bias_pos=0.4, traj_bias_rot=2.2
    ):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.motion_generator = motion_generator
        self.show_trajectory = show_trajectory
        self.camera_follow = True

        self.past_frames = past_frames
        self.future_frames = future_frames
        self.apply_generated_frames = int(applyframes)
        self.cfg_count_cache = int(cfg_count)
        self.cfg_count = int(cfg_count)
        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)

        self.current_motion_idx = 0
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()
        self.fps = 30
        self.frame_dt = 1.0 / self.fps

        self.generated_qpos36 = None
        self.generated_obj_rel = None
        self.generated_contact = None
        self.generated_frame_idx = 0
        self.prev_style_idx = None

        self.command_contact_target = 0.0
        self.qpos_history = deque(maxlen=self.past_frames)
        self.contact_history = deque(maxlen=self.past_frames)
        self.current_obj_pose_world = None

        self.load_motion(0)

    def load_motion(self, motion_idx):
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion_data = self.dataset[self.current_motion_idx]
        self.object_pose_relative_frame = self.current_motion_data.object_pose_relative_frame
        self.current_frame = 0
        if self.prev_style_idx is None or self.current_motion_data.style_idx != self.prev_style_idx:
            self.cfg_count = self.cfg_count_cache
        self.prev_style_idx = self.current_motion_data.style_idx

        q = self.current_motion_data.get_qpos43(self.current_frame)
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

        self.qpos_history.clear()
        self.contact_history.clear()
        init_contact = self.current_motion_data.get_contact(self.current_frame)
        for _ in range(self.past_frames):
            self.qpos_history.append(q.copy())
            self.contact_history.append(init_contact)

        self.current_obj_pose_world = q[36:43].copy()
        self.generated_qpos36 = None
        self.generated_obj_rel = None
        self.generated_contact = None
        self.generated_frame_idx = 0
        self.update_past_trajectory()
        self.update_future_trajectory()

        print(f"\nMotion {self.current_motion_idx + 1}/{len(self.dataset)} | style={self.current_motion_data.style}")

    def update_past_trajectory(self):
        qh = np.array(self.qpos_history)
        self.past_traj = qh[:, :3]
        self.past_orient = qh[:, 3:7]

    def _load_trajectory_from_dataset(self):
        p, f, po, fo = self.current_motion_data.get_trajectory(self.current_frame, self.past_frames, self.future_frames)
        return p[:, :2], f[:, :2], po, fo

    def load_future_trajectory(self):
        _, future_traj_dataset, _, future_orient_dataset = self._load_trajectory_from_dataset()
        ref_q = self.current_motion_data.get_qpos43(self.current_frame)
        curr_q = self.data.qpos.copy()
        aligned_traj, aligned_orient = align_trajectory_to_pose(future_traj_dataset, future_orient_dataset, ref_q[:36], curr_q[:36])
        self.future_traj_dataset, self.future_orient_dataset = match_future_horizon(
            aligned_traj, aligned_orient, self.future_frames
        )

        # Object target future trajectory (same alignment transform as root trajectory).
        obj_traj_dataset, obj_orient_dataset = self.current_motion_data.get_object_trajectory(
            self.current_frame, self.past_frames, self.future_frames
        )
        obj_z = obj_traj_dataset[:, 2:3]
        aligned_obj_traj, aligned_obj_orient = align_trajectory_to_pose(
            obj_traj_dataset[:, :2], obj_orient_dataset, ref_q[:36], curr_q[:36]
        )
        self.future_obj_traj_dataset, self.future_obj_orient_dataset = match_future_horizon(
            aligned_obj_traj, aligned_obj_orient, self.future_frames
        )
        if obj_z.shape[0] == 0:
            obj_z = np.zeros((1, 1), dtype=np.float32)
        if obj_z.shape[0] < self.future_frames:
            pad_n = self.future_frames - obj_z.shape[0]
            obj_z = np.concatenate([obj_z, np.repeat(obj_z[-1:], pad_n, axis=0)], axis=0)
        else:
            obj_z = obj_z[: self.future_frames]
        self.future_obj_traj_dataset = np.concatenate([self.future_obj_traj_dataset, obj_z], axis=-1).astype(np.float32)

    def update_future_trajectory(self):
        self.load_future_trajectory()
        if self.generated_qpos36 is None:
            self.future_traj = self.future_traj_dataset
            self.future_orient = self.future_orient_dataset
            self.future_obj_traj = self.future_obj_traj_dataset
            self.future_obj_orient = self.future_obj_orient_dataset
            return

        t_cur = self.generated_frame_idx
        pred_traj = self.generated_qpos36[:, :3]
        pred_orient = self.generated_qpos36[:, 3:7]
        model_pred_future_traj = pred_traj[t_cur + 1 :, :2]
        model_pred_future_orient = pred_orient[t_cur + 1 :]
        ext_traj, ext_orient = extend_future_traj_heusristic(model_pred_future_traj, model_pred_future_orient, self.future_frames, K=1)
        blend_traj, blend_orient = blend_trajectory(
            ext_traj, ext_orient, self.future_traj_dataset, self.future_orient_dataset,
            blend=self.traj_bias_pos, blend_rot=self.traj_bias_rot
        )
        self.future_traj = blend_traj
        self.future_orient = blend_orient
        self.future_obj_traj = self.future_obj_traj_dataset
        self.future_obj_orient = self.future_obj_orient_dataset

    def build_desired_contact_traj(self):
        start = float(self.contact_history[-1]) if len(self.contact_history) > 0 else 0.0
        target = float(self.command_contact_target)
        horizon = self.future_frames
        out = np.full((horizon,), target, dtype=np.float32)
        ramp = min(10, horizon)
        if ramp > 1:
            out[:ramp] = np.linspace(start, target, ramp, dtype=np.float32)
        return out

    def generate_motion(self):
        past_q = np.array(self.qpos_history)  # [Tp,43]
        past_c = np.array(self.contact_history, dtype=np.float32)  # [Tp]
        desired_contact = self.build_desired_contact_traj()  # [Tf]
        style_idx = self.current_motion_data.style_idx
        effective_cfg_scale = self.motion_generator.cfg_scale if self.cfg_count > 0 else 1.0

        q36, obj_rel, pred_c = self.motion_generator.generate_motion(
            past_q, past_c, self.future_traj, self.future_orient, desired_contact,
            self.future_obj_traj, self.future_obj_orient,
            style_idx, cfg_scale=effective_cfg_scale,
            frame_mode=self.object_pose_relative_frame,
            has_object=self.current_motion_data.has_object,
        )
        if self.cfg_count > 0:
            self.cfg_count -= 1
        return q36, obj_rel, pred_c

    def update_pose(self):
        if self.generated_frame_idx == 0:
            self.generated_qpos36, self.generated_obj_rel, self.generated_contact = self.generate_motion()

        i = self.generated_frame_idx
        q36 = self.generated_qpos36[i]
        obj_rel = self.generated_obj_rel[i]
        c = float(self.generated_contact[i])

        # Update object world pose only when predicted contact is active.
        if self.current_motion_data.has_object and c > 0.5:
            obj_pos, obj_q = object_relative_to_world(
                q36[:3], q36[3:7], obj_rel, frame_mode=self.object_pose_relative_frame
            )
            self.current_obj_pose_world = np.concatenate([obj_pos, obj_q], axis=0).astype(np.float32)
        elif self.current_obj_pose_world is None or not self.current_motion_data.has_object:
            self.current_obj_pose_world = np.array([0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        q43 = np.concatenate([q36, self.current_obj_pose_world], axis=0).astype(np.float32)

        self.qpos_history.append(q43.copy())
        self.contact_history.append(np.clip(c, 0.0, 1.0))
        self.update_past_trajectory()
        self.update_future_trajectory()
        self.data.qpos[:] = q43
        mujoco.mj_forward(self.model, self.data)

        self.generated_frame_idx = (self.generated_frame_idx + 1) % self.apply_generated_frames

    def step(self):
        if not self.playing:
            return
        dt = time.time() - self.last_update_time
        if dt >= self.frame_dt / self.playback_speed:
            self.current_frame += 1
            if self.current_frame >= self.current_motion_data.num_frames:
                self.current_frame = 0
            self.update_pose()
            self.last_update_time = time.time()

    def set_pick_command(self):
        self.command_contact_target = 1.0
        print("Command: PICK (desired future contact -> 1)")

    def set_drop_command(self):
        self.command_contact_target = 0.0
        print("Command: DROP (desired future contact -> 0)")

    def next_motion(self):
        self.load_motion(self.current_motion_idx + 1)

    def prev_motion(self):
        self.load_motion(self.current_motion_idx - 1)

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def reset(self):
        self.load_motion(self.current_motion_idx)
        print("Reset")

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow
        print(f"Camera follow: {'ON' if self.camera_follow else 'OFF'}")

    def set_speed(self, speed):
        self.playback_speed = speed
        print(f"Playback speed: {speed}x")

    def print_status(self):
        print(
            f"Motion {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"style={self.current_motion_data.style} | "
            f"frame={self.current_frame}/{self.current_motion_data.num_frames} | "
            f"cmd_contact_target={self.command_contact_target:.1f} | "
            f"pred_contact={float(self.contact_history[-1]):.2f}"
        )

    def render_trajectory(self, scene):
        if not self.show_trajectory:
            return
        if len(self.past_traj) > 0:
            draw_trajectory(scene, self.past_traj, self.past_orient, color=[0.2, 0.5, 1.0, 1.0])
        if len(self.future_traj_dataset) > 0:
            draw_trajectory(scene, self.future_traj_dataset, self.future_orient_dataset, color=[1.0, 0.2, 0.2, 1.0])
            draw_trajectory(scene, self.future_traj, self.future_orient, color=[0.2, 1.0, 0.2, 1.0])


def get_args():
    parser = argparse.ArgumentParser(description="Object demo with pick/drop contact commands")
    parser.add_argument("--dataset", type=str, default="data/pkls/merged_object_motion.pkl")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--past-frames", type=int, default=10)
    parser.add_argument("--future-frames", type=int, default=45)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--cfg-scale", type=float, default=0.5)
    parser.add_argument("--cfg-count", type=int, default=2)
    parser.add_argument("--applyframes", type=int, default=15)
    parser.add_argument("--traj-bias-pos", type=float, default=0.4)
    parser.add_argument("--traj-bias-rot", type=float, default=2.2)
    return parser.parse_args()


def print_instruction():
    print("\n" + "=" * 68)
    print("Controls:")
    print("SPACE pause | UP/DOWN motion | T trajectory | C camera | 1-9 speed")
    print("P pick command (future contact->1) | O drop command (future contact->0)")
    print("S status | R reset | ESC exit")
    print("=" * 68 + "\n")


def key_callback(player, keycode):
    if keycode == 32:
        player.toggle_pause()
    elif keycode == 265:
        player.next_motion()
    elif keycode == 264:
        player.prev_motion()
    elif keycode in (ord("r"), ord("R")):
        player.reset()
    elif keycode in (ord("t"), ord("T")):
        player.toggle_trajectory()
    elif keycode in (ord("c"), ord("C")):
        player.toggle_camera_follow()
    elif keycode in (ord("s"), ord("S")):
        player.print_status()
    elif keycode in (ord("p"), ord("P")):
        player.set_pick_command()
    elif keycode in (ord("o"), ord("O")):
        player.set_drop_command()
    elif ord("1") <= keycode <= ord("9"):
        speed_map = {
            ord("1"): 0.25, ord("2"): 0.5, ord("3"): 0.75, ord("4"): 0.9, ord("5"): 1.0,
            ord("6"): 1.25, ord("7"): 1.5, ord("8"): 1.75, ord("9"): 2.0,
        }
        player.set_speed(speed_map[keycode])


def main():
    args = get_args()
    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene_object.xml")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    dataset = ObjectMotionDataset(args.dataset)
    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    diffusion = create_gaussian_diffusion(config)
    diffusion_model = MotionDiffusionObject(
        input_feats=199,
        nstyles=len(dataset.styles),
        njoints=199,
        nfeats=1,
        rot_req=config.arch.rot_req,
        clip_len=config.arch.clip_len,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        traj_pose_feats=6,
        traj_trans_feats=2,
        traj_contact_feats=1,
        traj_obj_pose_feats=6,
        traj_obj_trans_feats=3,
        device=device,
    ).to(device)
    diffusion_model.load_state_dict(checkpoint["state_dict"], strict=True)
    diffusion_model.eval()

    generator = MotionGeneratorObject(
        diffusion_model, diffusion, config, device=device, sampler=args.sampler, cfg_scale=args.cfg_scale
    )

    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)
    player = DemoPlayerObject(
        mj_model, mj_data, dataset, generator,
        show_trajectory=True, past_frames=args.past_frames, future_frames=args.future_frames,
        applyframes=args.applyframes, cfg_count=args.cfg_count,
        traj_bias_pos=args.traj_bias_pos, traj_bias_rot=args.traj_bias_rot,
    )

    print_instruction()
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=lambda kc: key_callback(player, kc)) as viewer:
        viewer.sync()
        while viewer.is_running():
            player.step()
            viewer.user_scn.ngeom = 0
            player.render_trajectory(viewer.user_scn)
            if player.camera_follow:
                viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
