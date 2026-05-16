"""
Build a source motion pkl that combines:
  - lafan motion30 (data/Lafan1_g1/raw/walk1_subject5.csv)  -> style='walk'
  - three new jump motions in data/Lafan1_g1/raw_jump/*.csv -> style='jump'

The output pkl matches the schema produced by make_pose_data_g1.py so it can be
fed directly to ``visualize/step2_visualize_data_env2d.py --create-dataset``.

Output: data/pkls/lafan1_g1_motion30_jump.pkl
"""
import os
import pickle
from data.make_pose_data_g1 import load_g1_csv, JOINT_NAMES


WALK_CSV = "data/Lafan1_g1/raw/walk1_subject5.csv"            # motion30 in sorted listing
JUMP_CSVS = [
    "data/Lafan1_g1/raw_jump/walk_jump_walk.csv",
    "data/Lafan1_g1/raw_jump/walk_jump_walk2.csv",
    "data/Lafan1_g1/raw_jump/walk_jump_stop.csv",
]
OUTPUT_PKL = "data/pkls/lafan1_g1_motion30_jump.pkl"


def build_motion(csv_path: str, style: str) -> dict:
    root_pos, local_rot, traj_trans, traj_pose = load_g1_csv(csv_path)
    return {
        "filepath": csv_path,
        "local_joint_rotations": local_rot,
        "global_root_positions": root_pos,
        "traj": traj_trans,
        "traj_pose": traj_pose,
        "style": style,
        "text": style,
        "joint_names": JOINT_NAMES,
    }


def main():
    motions = []
    print(f"[walk] {WALK_CSV}")
    motions.append(build_motion(WALK_CSV, "walk"))
    for csv in JUMP_CSVS:
        print(f"[jump] {csv}")
        motions.append(build_motion(csv, "jump"))

    data = {"parents": None, "offsets": None, "names": None, "motions": motions}
    os.makedirs(os.path.dirname(OUTPUT_PKL) or ".", exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(data, f)

    print(f"Wrote {OUTPUT_PKL}")
    for m in motions:
        T = m["local_joint_rotations"].shape[0]
        print(f"  style={m['style']:<6} frames={T:5d}  file={m['filepath']}")


if __name__ == "__main__":
    main()
