"""
Convert kimodo .npz motion files (G1Skeleton34, y-up z-forward) to G1 MuJoCo
qpos CSVs (z-up, x-forward) in the same (T, 36) layout used by
``data/Lafan1_g1/raw/*.csv``: ``root_xyz(3) + root_quat_xyzw(4) + 29 joint angles``.

Run inside the kimodo conda env (kimodo / kimodo_demo).

The .npz files are expected to contain:
  - local_rot_mats : (T, 34, 3, 3)  -- kimodo skeleton local rotations
  - root_positions : (T, 3)          -- kimodo-space root positions
"""
import argparse
import os
import sys
import numpy as np

from kimodo.skeleton import G1Skeleton34
from kimodo.exports.mujoco import MujocoQposConverter


def convert(npz_path: str, csv_path: str) -> int:
    data = np.load(npz_path, allow_pickle=True)
    local_rot_mats = data["local_rot_mats"].astype(np.float32)  # (T, 34, 3, 3)
    root_positions = data["root_positions"].astype(np.float32)  # (T, 3)

    sk = G1Skeleton34()
    conv = MujocoQposConverter(sk)

    # dict_to_qpos handles (T, J, 3, 3) by ensuring batched internally.
    qpos = conv.dict_to_qpos(
        {
            "local_rot_mats": local_rot_mats,
            "root_positions": root_positions,
        },
        device=None,
        root_quat_w_first=False,   # CSVs store x,y,z,w (see make_pose_data_g1.load_g1_csv)
        numpy=True,
        mujoco_rest_zero=False,
    )

    # dict_to_qpos returns (B, T, 36). Squeeze batch.
    qpos = np.asarray(qpos)
    if qpos.ndim == 3:
        if qpos.shape[0] != 1:
            raise ValueError(f"Expected batch=1, got {qpos.shape}")
        qpos = qpos[0]

    if qpos.shape[1] != 36:
        raise ValueError(f"qpos last dim {qpos.shape[1]} != 36")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    np.savetxt(csv_path, qpos, delimiter=",", fmt="%.6f")
    print(f"  saved {csv_path}  ({qpos.shape[0]} frames)")
    return qpos.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more .npz files to convert")
    p.add_argument("--out-dir", default="data/Lafan1_g1/raw_jump",
                   help="Output directory for CSV files")
    args = p.parse_args()

    total = 0
    for npz in args.inputs:
        stem = os.path.splitext(os.path.basename(npz))[0]
        out_csv = os.path.join(args.out_dir, f"{stem}.csv")
        print(f"[convert] {npz} -> {out_csv}")
        total += convert(npz, out_csv)
    print(f"Done. {len(args.inputs)} files, {total} frames total.")


if __name__ == "__main__":
    main()
