import argparse
import glob
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


AXIS_FORWARD = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def extract_traj(root_positions: np.ndarray, forward_vectors: np.ndarray, kernels=(5, 10)):
    traj_trans, traj_pose = [], []
    canonical_forward = AXIS_FORWARD

    for k in kernels:
        smooth_xy = gaussian_filter1d(root_positions[:, [0, 1]], k, axis=0)
        traj_trans.append(smooth_xy.astype(np.float32))

        fwd = gaussian_filter1d(forward_vectors, k, axis=0)
        fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-8

        xaxis = canonical_forward.repeat(len(fwd), axis=0)
        axis = np.cross(xaxis, fwd)
        w = np.sqrt((xaxis ** 2).sum(axis=1) * (fwd ** 2).sum(axis=1)) + (xaxis * fwd).sum(axis=1)

        quat_wxyz = np.concatenate([w[:, None], axis], axis=-1)
        quat_wxyz = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_quat()[:, [3, 0, 1, 2]]
        traj_pose.append(quat_wxyz.astype(np.float32))

    return traj_trans, traj_pose


def detect_grasp_from_motion(qpos_seq: np.ndarray, motion_threshold=0.008, window_size=5) -> np.ndarray:
    """Contact logic mirrored from visualize/step2_visualize_data_object.py."""
    grasp_states = []
    box_positions = qpos_seq[:, 36:39]

    for i in range(len(qpos_seq)):
        if i < window_size:
            grasp_states.append(0.0)
        else:
            recent_positions = box_positions[i - window_size : i + 1]
            motion = np.max(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))
            in_grasp = motion >= motion_threshold or box_positions[i, 2] > 0.2
            grasp_states.append(1.0 if in_grasp else 0.0)
    return np.array(grasp_states, dtype=np.float32)[:, None]


def wxyz_to_mat(wxyz: np.ndarray) -> np.ndarray:
    return R.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix()


def build_local_joint_rotations_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """qpos=(T,43) -> local_joint_rotations=(T,30,4)."""
    T = qpos.shape[0]
    local = np.zeros((T, 30, 4), dtype=np.float32)
    local[:, 0] = qpos[:, 3:7]      # root quat in WXYZ
    local[:, 1:, 0] = qpos[:, 7:36] # 29 1-DoF joints in channel 0
    return local


def build_traj_from_root(root_pos: np.ndarray, root_quat_wxyz: np.ndarray):
    root_rot = R.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]])
    forward = root_rot.apply(AXIS_FORWARD.repeat(len(root_pos), axis=0))
    forward[:, 2] = 0.0
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8
    return extract_traj(root_pos, forward)


def compute_object_pose_relative(qpos: np.ndarray) -> np.ndarray:
    """Returns (T,12): [p_rel(3), R_rel(9 row-major)]."""
    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    obj_pos = qpos[:, 36:39]
    obj_quat = qpos[:, 39:43]

    root_rot_m = wxyz_to_mat(root_quat)
    obj_rot_m = wxyz_to_mat(obj_quat)

    root_rot_inv = np.transpose(root_rot_m, (0, 2, 1))
    p_rel = np.einsum("tij,tj->ti", root_rot_inv, obj_pos - root_pos)
    r_rel = np.einsum("tij,tjk->tik", root_rot_inv, obj_rot_m)

    return np.concatenate([p_rel, r_rel.reshape(len(qpos), 9)], axis=-1).astype(np.float32)


def load_walk_motions(walk_pkl_path: str) -> Tuple[List[Dict], List[str]]:
    data = pickle.load(open(walk_pkl_path, "rb"))
    motions = data["motions"]
    if not motions:
        raise RuntimeError(f"No motions found in walk pkl: {walk_pkl_path}")

    joint_names = motions[0].get("joint_names", [])
    out = []
    for i, motion in enumerate(motions):
        T = motion["local_joint_rotations"].shape[0]
        padded_obj_pose = np.zeros((T, 12), dtype=np.float32)
        padded_contact = np.zeros((T, 1), dtype=np.float32)

        m = dict(motion)
        m["object_pose_relative"] = padded_obj_pose
        m["object_contact_mask"] = padded_contact
        m["has_object"] = False
        m["source"] = "walk"
        m["text"] = m.get("text", m.get("style", "walk"))
        m["filepath"] = m.get("filepath", f"{walk_pkl_path}::motion_{i}")
        out.append(m)
    return out, joint_names


def load_object_motions(object_dir: str, joint_names: List[str]) -> List[Dict]:
    npz_files = sorted(glob.glob(os.path.join(object_dir, "*.npz")))
    if not npz_files:
        raise RuntimeError(f"No object npz files found in: {object_dir}")

    motions = []
    for npz_path in npz_files:
        npz = np.load(npz_path)
        qpos = npz["qpos"].astype(np.float32)  # (T,43)
        if qpos.shape[1] != 43:
            raise RuntimeError(f"Unexpected qpos dim in {npz_path}: {qpos.shape}")

        root_pos = qpos[:, 0:3].astype(np.float32)
        root_quat = qpos[:, 3:7].astype(np.float32)
        local_rot = build_local_joint_rotations_from_qpos(qpos)
        traj, traj_pose = build_traj_from_root(root_pos, root_quat)

        obj_pose_rel = compute_object_pose_relative(qpos)
        contact = detect_grasp_from_motion(qpos)

        basename = os.path.basename(npz_path)
        motions.append(
            {
                "filepath": npz_path,
                "local_joint_rotations": local_rot,
                "global_root_positions": root_pos,
                "traj": traj,
                "traj_pose": traj_pose,
                "style": "object_pick_carry_place",
                "text": basename,
                "joint_names": joint_names,
                "object_pose_relative": obj_pose_rel,
                "object_contact_mask": contact,
                "has_object": True,
                "source": "object_npz",
            }
        )
    return motions


def main():
    parser = argparse.ArgumentParser(description="Merge walk + object motions into one pkl for single-object CAMDM.")
    parser.add_argument("--walk-pkl", type=str, default="data/pkls/lafan1_g1_motion28.pkl")
    parser.add_argument("--object-dir", type=str, default="data/robot-object-mujoco")
    parser.add_argument("--output", type=str, default="data/pkls/merged_object_motion.pkl")
    args = parser.parse_args()

    walk_motions, joint_names = load_walk_motions(args.walk_pkl)
    object_motions = load_object_motions(args.object_dir, joint_names)

    merged = {
        "parents": None,
        "offsets": None,
        "names": None,
        "motions": walk_motions + object_motions,
        "metadata": {
            "description": "Walk + single-object pick/carry/place merged dataset",
            "object_pose_relative_format": "p_rel(3) + R_rel_row_major(9), relative to root frame per frame",
            "object_contact_logic": "Same as step2_visualize_data_object.py detect_grasp_from_motion",
            "walk_object_padding": "zeros for object_pose_relative/contact_mask",
            "counts": {
                "walk_motions": len(walk_motions),
                "object_motions": len(object_motions),
                "total_motions": len(walk_motions) + len(object_motions),
            },
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(merged, f)

    print("=" * 70)
    print(f"Saved merged dataset: {args.output}")
    print(f"Walk motions:   {len(walk_motions)}")
    print(f"Object motions: {len(object_motions)}")
    print(f"Total motions:  {len(merged['motions'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
