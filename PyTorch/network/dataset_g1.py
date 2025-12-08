import sys
sys.path.append('./')

import torch
import pickle
import random
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.data import Dataset

import utils.nn_transforms as nn_transforms


class HumanoidMotionDataset(Dataset):
    """
    G1 Dataset loader
    - root joint: quaternion (4)
    - 29 joints: 1D angles stored in dim0 of a padded vector
    - Augmentations identical to original CAMDM
    """

    rot_feat_dim = {'q': 4, '6d': 6, 'euler': 3}

    def __init__(self, pkl_path, rot_req, offset_frame,
                 past_frame, future_frame, dtype=np.float32, limited_num=-1):

        self.pkl_path = pkl_path
        self.rot_req = rot_req.lower()
        self.dtype    = dtype

        window_size = past_frame + future_frame
        self.past_frame = past_frame
        self.reference_frame_idx = past_frame

        data_source = pickle.load(open(pkl_path, "rb"))

        self.rotations_list = []
        self.root_pos_list   = []
        self.local_conds = {"traj_pose": [], "traj_trans": []}
        self.global_conds = {"style": []}

        item_indices = []
        motion_idx = 0

        # -------------------------
        # Load motions
        # -------------------------
        for motion in tqdm(data_source["motions"][:limited_num]):
            N = motion["local_joint_rotations"].shape[0]
            if N < window_size:
                continue

            self.rotations_list.append(motion["local_joint_rotations"].astype(dtype))
            self.root_pos_list.append(motion["global_root_positions"].astype(dtype))

            self.local_conds["traj_pose"].append(
                np.array(motion["traj_pose"], dtype=dtype)
            )
            self.local_conds["traj_trans"].append(
                np.array(motion["traj"], dtype=dtype)
            )
            self.global_conds["style"].append(motion["style"])

            clips = np.arange(0, N - window_size + 1, offset_frame)[:, None] \
                    + np.arange(window_size)
            clips = np.hstack((np.full((len(clips),1), motion_idx), clips))
            item_indices.append(clips)

            motion_idx += 1

        self.item_frame_indices = np.concatenate(item_indices, axis=0)

        self.joint_names = data_source["motions"][0]["joint_names"]
        self.joint_num = self.rotations_list[0].shape[1]      # 1(root)+29 joints
        self.per_rot_feat = self.rot_feat_dim[self.rot_req]   # padded length

        # Random traj augmentation restored
        self.traj_aug_indexs1 = list(range(self.local_conds['traj_pose'][0].shape[0]))
        self.traj_aug_indexs2 = list(range(self.local_conds['traj_trans'][0].shape[0]))

        self.mask = np.ones(window_size - past_frame, dtype=bool)
        self.style_set = sorted(set(self.global_conds["style"]))

        print(f"G1 Dataset loaded: {motion_idx} motions, {len(self.item_frame_indices)} clips, {len(self.style_set)} styles")


    def __len__(self):
        return len(self.item_frame_indices)


    # -----------------------------------------------------------
    # Pad single-angle joints to per_rot_feat dim
    # -----------------------------------------------------------
    def pad_joint_angle(self, angles):
        """
        Input: angles: (T, 29, 4) but only [:,:,0] contains angle
        Output: padded to (T,29,per_rot_feat)
        """
        T, J, _ = angles.shape
        out = torch.zeros((T, J, self.per_rot_feat), dtype=torch.float32)
        out[..., 0] = angles[..., 0]       # place 1D angle in channel 0
        return out


    def convert_rot(self, quat_tensor):
        """
        quat_tensor: (T,4) wxyz
        return (T, per_rot_feat)
        """
        # For q,6d,euler use nn_transforms
        return nn_transforms.get_rotation(quat_tensor, self.rot_req)


    def __getitem__(self, idx):

        item = self.item_frame_indices[idx]
        motion_idx, frame_ids = item[0], item[1:]

        rotations = self.rotations_list[motion_idx][frame_ids].copy() # (TW,1+29, 4) wxyz
        root_pos  = self.root_pos_list[motion_idx][frame_ids].copy() # (TW, 3) xyz

        # Normalize XY (subtract root_pos from last past frame)
        root_pos[:, [0,1]] -= root_pos[self.reference_frame_idx-1, [0,1]]

        # Randomly choose trajectory version (among two smoothness levels)
        traj_rot = self.local_conds["traj_pose"][motion_idx][
            random.choice(self.traj_aug_indexs1)
        ][frame_ids]

        traj_pos = self.local_conds["traj_trans"][motion_idx][
            random.choice(self.traj_aug_indexs2)
        ][frame_ids]

        # Additional smoothing on traj_pos
        r = np.random.rand()
        if r < 0.75:
            k = 5 if r < 0.5 else 10
            traj_pos = gaussian_filter1d(traj_pos, k, axis=0)

        # Slice trajectory to future frames (TW=TF+TP -> TF)
        # Subtract last past frame position to normalize
        traj_pos -= traj_pos[self.reference_frame_idx-1]
        traj_pos = traj_pos[self.reference_frame_idx:] # (TF, 2)
        traj_rot = traj_rot[self.reference_frame_idx:] # (TF, 4) wxyz

        # -----------------------------
        # GLOBAL ROTATION AUGMENTATION
        # -----------------------------
        rot_xyzw     = rotations[..., [1,2,3,0]] # (TW, 1+29, 4)
        trajrot_xyzw = traj_rot[...,     [1,2,3,0]] # (TF, 4)

        # Random global rotation around Up-axis (Z axis for G1, originally Y axis from Unity)
        theta = np.repeat(np.random.uniform(0,2*np.pi), rotations.shape[0])
        rot_vec = R.from_rotvec(np.stack([0*theta, 0*theta, theta], axis=-1))

        # Rotate first rotation(root)
        rotations[:,0] = (rot_vec * R.from_quat(rot_xyzw[:,0])) \
                            .as_quat()[..., [3,0,1,2]]

        # Rotate trajectory rotations
        traj_rot = (rot_vec[self.reference_frame_idx:] *
                    R.from_quat(trajrot_xyzw)) \
                    .as_quat()[..., [3,0,1,2]]

        # Rotate root positions
        root_pos = rot_vec.apply(root_pos)

        # -----------------------------
        # TORCH CONVERSION
        # -----------------------------
        rotations = torch.from_numpy(rotations.astype(self.dtype))  # (TW,30,4)
        traj_pos  = torch.from_numpy(traj_pos.astype(self.dtype))   # (TF,2)
        traj_rot  = torch.from_numpy(traj_rot.astype(self.dtype))   # (TF,4) wxyz
        traj_rot  = self.convert_rot(traj_rot)                      # (TF,per_rot_feat)

        # ---------------------------------------------------------
        # Convert ROOT to requested representation
        # ---------------------------------------------------------
        root_quat = rotations[:,0]    # (TW,4)
        root_repr = self.convert_rot(root_quat)   # (TW,per_rot_feat)

        # ---------------------------------------------------------
        # Convert 1D joints to padded representation
        # ---------------------------------------------------------
        joints = rotations[:,1:]      # (TW,29,4)
        joints_repr = self.pad_joint_angle(joints) # (TW,29,per_rot_feat)

        # Stick root back at index 0
        root_repr = root_repr.unsqueeze(1)         # (TW,1,per_rot_feat)
        rotations_full = torch.cat([root_repr, joints_repr], dim=1)  # (TW,1+29,per_rot_feat)

        # ---------------------------------------------------------
        # Append root translation as a joint
        # ---------------------------------------------------------
        root_pos_pad = torch.zeros((root_pos.shape[0], 1, self.per_rot_feat))
        root_pos_pad[..., :3] = torch.from_numpy(root_pos[:,None].astype(self.dtype))
        rotations_w_root = torch.cat([rotations_full, root_pos_pad], dim=1) # (TW, 31=1+29+1, per_rot_feat)

        # Slice past/future
        future = rotations_w_root[self.reference_frame_idx:]
        past   = rotations_w_root[:self.reference_frame_idx]

        style_idx = float(self.style_set.index(self.global_conds["style"][motion_idx]))

        return {
            "data": future,
            "conditions": {
                'past_motion': past, # (TP, 31=1+29+1, per_rot_feat)
                'traj_pose': traj_rot, # (TF, per_rot_feat)
                'traj_trans': traj_pos, # (TF, 2)
                'style': self.global_conds["style"][motion_idx], # string
                'style_idx': style_idx, # float index
                'mask': self.mask # (TF,) boolean, all True(=no masking)
            }
        }

# -----------------------------------------------------------
# Case test for the G1 dataset class
# -----------------------------------------------------------
if __name__ == "__main__":

    import time
    import torch

    # Path to the pickle created by make_pose_data_g1.py
    pkl_path = "data/pkls/lafan1_g1.pkl"
    
    # Only 1D rotation used for G1
    rot_req = "6d"

    # Load dataset
    print("\n=== Loading G1 Humanoid Dataset ===")
    dataset = HumanoidMotionDataset(
        pkl_path=pkl_path,
        rot_req=rot_req,
        offset_frame=1,
        past_frame=10,
        future_frame=45,
        dtype=np.float32,
        limited_num=-1
    )

    # Build DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # Inspect one batch
    print("\n=== Inspect one batch ===")
    batch = next(iter(loader))
    print("Future data shape:", batch["data"].shape)
    print("Past motion shape:", batch["conditions"]['past_motion'].shape)
    print("Traj pose shape:", batch["conditions"]['traj_pose'].shape)
    print("Traj trans shape:", batch["conditions"]['traj_trans'].shape)
    print("Mask shape:", batch["conditions"]['mask'].shape)
    print("Style:", batch["conditions"]['style'][:5])
    print("Style idx:", batch["conditions"]['style_idx'][:5])

    # Loop speed test
    print("\n=== Iteration speed test ===")
    times = []
    start_time = time.time()

    for i, batch in enumerate(loader):
        end_time = time.time()
        print(f"Iteration {i}: {end_time - start_time:.4f} sec")
        times.append(end_time - start_time)
        start_time = end_time

        if i == 10:  # test first ~10 iterations only
            break

    print("\nAverage data loading time:", sum(times) / len(times), "sec")
    print("Total tested loading time:", sum(times), "sec")
