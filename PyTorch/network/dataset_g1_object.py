import sys
sys.path.append('./')

import random
import pickle
import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.data import Dataset

import utils.nn_transforms as nn_transforms


def _quat_wxyz_to_yaw_mat_batch(q_wxyz):
    """(T,4) wxyz -> (T,3,3) yaw-only rotation matrices."""
    rot = R.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    forward = rot.apply(np.tile([1.0, 0.0, 0.0], (len(q_wxyz), 1)))
    forward[:, 2] = 0.0
    norm = np.linalg.norm(forward[:, :2], axis=1, keepdims=True)
    forward[:, :2] /= np.maximum(norm, 1e-8)
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    return R.from_euler("z", yaw).as_matrix()


def _obj_rel_to_world_centered(obj_pose_rel, root_pos_centered, root_q_wxyz):
    """Convert root-yaw-relative obj pose to world-frame-centered.

    obj_pose_rel:       (T,12) = [p_rel(3), R_rel_flat(9)]
    root_pos_centered:  (T,3)  XY already centered by ref frame
    root_q_wxyz:        (T,4)  root quaternions (before rotation aug)
    Returns:            (T,12) = [p_world_centered(3), R_world_flat(9)]
    """
    root_r = _quat_wxyz_to_yaw_mat_batch(root_q_wxyz)          # (T,3,3)
    p_rel = obj_pose_rel[:, :3]
    R_rel = obj_pose_rel[:, 3:].reshape(-1, 3, 3)
    p_world = root_pos_centered + np.einsum("tij,tj->ti", root_r, p_rel)
    R_world = np.einsum("tij,tjk->tik", root_r, R_rel)
    return np.concatenate([p_world, R_world.reshape(-1, 9)], axis=-1).astype(np.float32)


class SingleObjectMotionDataset(Dataset):
    """
    G1 + single-object dataset.
    Denoising target is concatenated vector:
    [full_body_state, object_pose_world_centered(12), contact_mask(1)].
    Object pose is stored in world-frame centered by the diffusion reference
    frame XY — the same convention used for body root position.
    """

    rot_feat_dim = {'q': 4, '6d': 6, 'euler': 3}

    def __init__(self, pkl_path, rot_req, offset_frame,
                 past_frame, future_frame, dtype=np.float32, limited_num=None):
        self.pkl_path = pkl_path
        self.rot_req = rot_req.lower()
        self.dtype = dtype

        window_size = past_frame + future_frame
        self.past_frame = past_frame
        self.reference_frame_idx = past_frame

        data_source = pickle.load(open(pkl_path, "rb"))

        self.rotations_list = []
        self.root_pos_list = []
        self.object_pose_rel_list = []
        self.object_contact_list = []
        self.has_object_list = []

        self.local_conds = {"traj_pose": [], "traj_trans": [], "obj_traj_pose": [], "obj_traj_trans": []}
        self.global_conds = {"style": []}
        item_indices = []
        motion_idx = 0

        for motion in tqdm(data_source["motions"][:limited_num]):
            N = motion["local_joint_rotations"].shape[0]
            if N < window_size:
                continue

            rotations = motion["local_joint_rotations"].astype(dtype)
            root_pos = motion["global_root_positions"].astype(dtype)
            obj_rel = np.asarray(motion.get("object_pose_relative", np.zeros((N, 12), dtype=dtype)), dtype=dtype)
            obj_contact = np.asarray(motion.get("object_contact_mask", np.zeros((N, 1), dtype=dtype)), dtype=dtype).reshape(N, 1)
            if "obj_traj" not in motion or "obj_traj_pose" not in motion:
                raise KeyError(
                    "Missing required object future conditions in pkl motion: "
                    "'obj_traj' and 'obj_traj_pose'. Re-generate merged_object_motion.pkl with the latest script."
                )
            if motion.get("object_pose_relative_frame", None) != "root_yaw_gravity_aligned":
                raise ValueError(
                    "object_pose_relative_frame must be 'root_yaw_gravity_aligned'. "
                    "Re-generate merged_object_motion.pkl with the latest script."
                )
            obj_traj = np.array(motion["obj_traj"], dtype=dtype)
            obj_traj_pose = np.array(motion["obj_traj_pose"], dtype=dtype)
            if obj_traj.shape[-1] != 3:
                raise ValueError(
                    f"obj_traj must have shape (..., 3) for xyz conditioning, got last dim={obj_traj.shape[-1]}. "
                    "Re-generate merged_object_motion.pkl with latest script."
                )

            self.rotations_list.append(rotations)
            self.root_pos_list.append(root_pos)
            self.object_pose_rel_list.append(obj_rel)
            self.object_contact_list.append(obj_contact)
            self.has_object_list.append(bool(motion.get("has_object", False)))

            self.local_conds["traj_pose"].append(np.array(motion["traj_pose"], dtype=dtype))
            self.local_conds["traj_trans"].append(np.array(motion["traj"], dtype=dtype))
            self.local_conds["obj_traj_pose"].append(np.array(obj_traj_pose, dtype=dtype))
            self.local_conds["obj_traj_trans"].append(np.array(obj_traj, dtype=dtype))
            self.global_conds["style"].append(motion["style"])

            clips = np.arange(0, N - window_size + 1, offset_frame)[:, None] + np.arange(window_size)
            clips = np.hstack((np.full((len(clips), 1), motion_idx), clips))
            item_indices.append(clips)
            motion_idx += 1

        self.item_frame_indices = np.concatenate(item_indices, axis=0)

        self.joint_names = data_source["motions"][0].get("joint_names", [])
        self.joint_num = self.rotations_list[0].shape[1]  # 1(root)+29 joints
        self.per_rot_feat = self.rot_feat_dim[self.rot_req]

        self.traj_aug_indexs1 = list(range(self.local_conds['traj_pose'][0].shape[0]))
        self.traj_aug_indexs2 = list(range(self.local_conds['traj_trans'][0].shape[0]))
        self.traj_obj_aug_indexs1 = list(range(self.local_conds['obj_traj_pose'][0].shape[0]))
        self.traj_obj_aug_indexs2 = list(range(self.local_conds['obj_traj_trans'][0].shape[0]))

        self.mask = np.ones(window_size - past_frame, dtype=bool)
        self.style_set = sorted(set(self.global_conds["style"]))

        # body: (root+29 joints + root translation) * per_rot_feat
        self.body_state_dim = (self.joint_num + 1) * self.per_rot_feat
        self.object_state_dim = 13
        self.state_dim = self.body_state_dim + self.object_state_dim

        print(
            f"SingleObject dataset loaded: {motion_idx} motions, "
            f"{len(self.item_frame_indices)} clips, {len(self.style_set)} styles, "
            f"state_dim={self.state_dim}"
        )

    def __len__(self):
        return len(self.item_frame_indices)

    def pad_joint_angle(self, angles):
        T, J, _ = angles.shape
        out = torch.zeros((T, J, self.per_rot_feat), dtype=torch.float32)
        out[..., 0] = angles[..., 0]
        return out

    def convert_rot(self, quat_tensor):
        return nn_transforms.get_rotation(quat_tensor, self.rot_req)

    def __getitem__(self, idx):
        item = self.item_frame_indices[idx]
        motion_idx, frame_ids = item[0], item[1:]

        rotations = self.rotations_list[motion_idx][frame_ids].copy()
        root_pos = self.root_pos_list[motion_idx][frame_ids].copy()
        obj_pose_rel = self.object_pose_rel_list[motion_idx][frame_ids].copy()
        obj_contact = self.object_contact_list[motion_idx][frame_ids].copy()
        root_ref_xy = root_pos[self.reference_frame_idx - 1, [0, 1]].copy()

        root_pos[:, [0, 1]] -= root_ref_xy

        # Convert root-relative object pose to world-frame-centered BEFORE rotation aug.
        # Use root_q (rotations[:,0]) before augmentation and root_pos already centered.
        has_object = self.has_object_list[motion_idx]
        if has_object:
            obj_pose_world = _obj_rel_to_world_centered(obj_pose_rel, root_pos, rotations[:, 0])
        else:
            obj_pose_world = np.zeros_like(obj_pose_rel)  # zeros for walk, consistent with inference

        traj_rot = self.local_conds["traj_pose"][motion_idx][random.choice(self.traj_aug_indexs1)][frame_ids]
        traj_pos = self.local_conds["traj_trans"][motion_idx][random.choice(self.traj_aug_indexs2)][frame_ids]
        traj_obj_rot = self.local_conds["obj_traj_pose"][motion_idx][random.choice(self.traj_obj_aug_indexs1)][frame_ids]
        traj_obj_pos = self.local_conds["obj_traj_trans"][motion_idx][random.choice(self.traj_obj_aug_indexs2)][frame_ids]

        r = np.random.rand()
        if r < 0.75:
            k = 5 if r < 0.5 else 10
            traj_pos = gaussian_filter1d(traj_pos, k, axis=0)

        traj_pos -= traj_pos[self.reference_frame_idx - 1]
        traj_pos = traj_pos[self.reference_frame_idx:]
        traj_rot = traj_rot[self.reference_frame_idx:]
        traj_obj_pos[:, :2] -= root_ref_xy  # normalize by root reference XY
        traj_obj_pos = traj_obj_pos[self.reference_frame_idx:]
        traj_obj_rot = traj_obj_rot[self.reference_frame_idx:]

        rot_xyzw = rotations[..., [1, 2, 3, 0]]
        trajrot_xyzw = traj_rot[..., [1, 2, 3, 0]]
        traj_obj_rot_xyzw = traj_obj_rot[..., [1, 2, 3, 0]]

        theta = np.repeat(np.random.uniform(0, 2 * np.pi), rotations.shape[0])
        rot_vec = R.from_rotvec(np.stack([0 * theta, 0 * theta, theta], axis=-1))

        rotations[:, 0] = (rot_vec * R.from_quat(rot_xyzw[:, 0])).as_quat()[..., [3, 0, 1, 2]]
        traj_rot = (rot_vec[self.reference_frame_idx:] * R.from_quat(trajrot_xyzw)).as_quat()[..., [3, 0, 1, 2]]
        traj_obj_rot = (rot_vec[self.reference_frame_idx:] * R.from_quat(traj_obj_rot_xyzw)).as_quat()[..., [3, 0, 1, 2]]
        root_pos = rot_vec.apply(root_pos)
        traj_obj_pos = rot_vec[self.reference_frame_idx:].apply(traj_obj_pos)
        # Apply rotation aug to world-frame object pose (positions + rotations).
        if has_object:
            rot_mats = rot_vec.as_matrix()  # (T,3,3)
            obj_world_pos_aug = np.einsum("tij,tj->ti", rot_mats, obj_pose_world[:, :3])
            obj_world_R_aug = np.einsum("tij,tjk->tik", rot_mats, obj_pose_world[:, 3:].reshape(-1, 3, 3))
            obj_pose_world = np.concatenate([obj_world_pos_aug, obj_world_R_aug.reshape(-1, 9)], axis=-1)

        rotations = torch.from_numpy(rotations.astype(self.dtype))
        traj_pos = torch.from_numpy(traj_pos.astype(self.dtype))
        traj_rot = torch.from_numpy(traj_rot.astype(self.dtype))
        traj_rot = self.convert_rot(traj_rot)
        traj_obj_rot = torch.from_numpy(traj_obj_rot.astype(self.dtype))
        traj_obj_rot = self.convert_rot(traj_obj_rot)
        traj_obj_pos = torch.from_numpy(traj_obj_pos.astype(self.dtype))

        obj_pose_world_t = torch.from_numpy(obj_pose_world.astype(self.dtype))
        obj_contact = torch.from_numpy(obj_contact.astype(self.dtype))

        root_quat = rotations[:, 0]
        root_repr = self.convert_rot(root_quat)
        joints = rotations[:, 1:]
        joints_repr = self.pad_joint_angle(joints)

        root_repr = root_repr.unsqueeze(1)
        rotations_full = torch.cat([root_repr, joints_repr], dim=1)

        root_pos_pad = torch.zeros((root_pos.shape[0], 1, self.per_rot_feat), dtype=torch.float32)
        root_pos_pad[..., :3] = torch.from_numpy(root_pos[:, None].astype(self.dtype))
        rotations_w_root = torch.cat([rotations_full, root_pos_pad], dim=1)

        body_state = rotations_w_root.reshape(rotations_w_root.shape[0], -1)
        object_state = torch.cat([obj_pose_world_t, obj_contact], dim=-1)
        full_state = torch.cat([body_state, object_state], dim=-1).unsqueeze(-1)  # [TW, D, 1]

        future = full_state[self.reference_frame_idx:]
        past = full_state[:self.reference_frame_idx]
        traj_contact = obj_contact[self.reference_frame_idx:]

        style_idx = float(self.style_set.index(self.global_conds["style"][motion_idx]))
        return {
            "data": future,
            "conditions": {
                "past_motion": past,
                "traj_pose": traj_rot,
                "traj_trans": traj_pos,
                "traj_contact": traj_contact,
                "traj_obj_pose": traj_obj_rot,
                "traj_obj_trans": traj_obj_pos,
                "style": self.global_conds["style"][motion_idx],
                "style_idx": style_idx,
                "mask": self.mask,
            },
        }
