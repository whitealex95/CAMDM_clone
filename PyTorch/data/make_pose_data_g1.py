import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

"""The Order of Configuration
G1: (30 FPS)
    root_joint(XYZQXQYQZQW)
    left_hip_pitch_joint
    left_hip_roll_joint
    left_hip_yaw_joint
    left_knee_joint
    left_ankle_pitch_joint
    left_ankle_roll_joint
    right_hip_pitch_joint
    right_hip_roll_joint
    right_hip_yaw_joint
    right_knee_joint
    right_ankle_pitch_joint
    right_ankle_roll_joint
    waist_yaw_joint
    waist_roll_joint
    waist_pitch_joint
    left_shoulder_pitch_joint
    left_shoulder_roll_joint
    left_shoulder_yaw_joint
    left_elbow_joint
    left_wrist_roll_joint
    left_wrist_pitch_joint
    left_wrist_yaw_joint
    right_shoulder_pitch_joint
    right_shoulder_roll_joint
    right_shoulder_yaw_joint
    right_elbow_joint
    right_wrist_roll_joint
    right_wrist_pitch_joint
    right_wrist_yaw_joint
"""

# WARNING: `make_pose_data.py` uses +Z as forward direction.
# In G1 dataset, +X is used as forward direction.
# TODO: make sure this is handled consistently in training/inference code.
AXIS_FORWARD = np.array([[1, 0, 0]])  # +X direction
AXIS_LEFT = np.array([[0, 1, 0]])  # +Y direction
AXIS_UP = np.array([[0, 0, 1]])  # +Z direction
NUM_JOINTS = 29                      # 29 joints after root
NUM_COLS   = 7 + NUM_JOINTS         # = 36
FPS = 30
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
def extract_traj(root_positions, forward_vectors, kernels=[5, 10]):
    traj_trans, traj_pose = [], []
    canonical_forward = AXIS_FORWARD   # forward = +X in world

    for k in kernels:
        smooth_xy = gaussian_filter1d(root_positions[:, [0, 1]], k, axis=0)
        traj_trans.append(smooth_xy)

        fwd = gaussian_filter1d(forward_vectors, k, axis=0)
        fwd /= (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-8)

        # quaternion that rotates x-axis to forward
        xaxis = canonical_forward.repeat(len(fwd), axis=0)
        axis = np.cross(xaxis, fwd)
        w = np.sqrt((xaxis**2).sum(axis=1) * (fwd**2).sum(axis=1)) + (xaxis * fwd).sum(axis=1)

        quat = np.concatenate([w[:,None], axis], axis=-1)
        quat = R.from_quat(quat[:, [1,2,3,0]]).as_quat()[:, [3,0,1,2]]   # wxyz
        traj_pose.append(quat) # wxyz format

    return traj_trans, traj_pose


def load_g1_csv(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.shape[1] != NUM_COLS:
        raise RuntimeError(
            f"CSV {csv_path} has {data.shape[1]} columns, expected {NUM_COLS}"
        )

    root_pos  = data[:, :3]
    root_quat_xyzw = data[:, 3:7]         # (x,y,z,w)
    joint_1d = data[:, 7:7+NUM_JOINTS]    # (N,29)

    # Convert root quaternion xyzw → wxyz
    root_quat_wxyz = root_quat_xyzw[:, [3,0,1,2]]

    N = root_pos.shape[0]

    # Build local_joint_rotations = (N, 1(root)+NUM_JOINTS, 4 dims)
    # Joint 0 = root quaternion (4 dims)
    # Joint 1–29 = 1D rotation at [:,0]
    local = np.zeros((N, 1+NUM_JOINTS, 4), dtype=np.float32)
    local[:, 0] = root_quat_wxyz
    local[:, 1:, 0] = joint_1d   # angle in dim 0

    # forward vector of the robot in ground plane
    r = R.from_quat(root_quat_wxyz[:, [1,2,3,0]])
    forward = r.apply(AXIS_FORWARD)
    # force forward vector to lie in XY plane
    forward[:, 2] = 0.0
    # renormalize
    forward /= (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)

    traj_trans, traj_pose = extract_traj(root_pos, forward)

    return root_pos, local, traj_trans, traj_pose


##############################################################################
# Main processing script
##############################################################################

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion",  type=int, default=-1,
                    help="Single motion index to export (-1 = all)")
    ap.add_argument("--motions", type=int, nargs="+", default=None,
                    help="Multiple motion indices to merge into one pkl "
                         "(e.g. --motions 24 25 29 31).  "
                         "Overrides --motion when given.")
    ap.add_argument("--input-dir", type=str, default="data/Lafan1_g1/raw")
    ap.add_argument("--output",    type=str, default=None)
    cli = ap.parse_args()

    input_dir = cli.input_dir

    # Decide which indices to include
    if cli.motions is not None:
        chosen = set(cli.motions)
        tag    = "_".join(str(m) for m in sorted(chosen))
        export_path = cli.output or f"data/pkls/lafan1_g1_motion{tag}.pkl"
    elif cli.motion >= 0:
        chosen = {cli.motion}
        export_path = cli.output or f"data/pkls/lafan1_g1_motion{cli.motion}.pkl"
    else:
        chosen = None   # all
        export_path = cli.output or "data/pkls/lafan1_g1.pkl"

    data_list = {"parents": None, "offsets": None, "names": None, "motions": []}

    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    print(f"Total CSV files found: {len(csv_files)}")
    if chosen is not None:
        print(f"Exporting motion indices: {sorted(chosen)}")
        for idx in sorted(chosen):
            if idx < len(csv_files):
                print(f"  [{idx:2d}] {csv_files[idx]}")
            else:
                print(f"  [{idx:2d}] *** INDEX OUT OF RANGE (max {len(csv_files)-1}) ***")

    for i, csv_file in enumerate(tqdm(csv_files)):
        if chosen is not None and i not in chosen:
            continue
        csv_path = os.path.join(input_dir, csv_file)

        style_name = ''.join([c for c in csv_file.split("_")[0] if not c.isdigit()])

        root_pos, local_rot, traj_trans, traj_pose = load_g1_csv(csv_path)

        motion_data = {
            "filepath": csv_path,
            "local_joint_rotations": local_rot,
            "global_root_positions": root_pos,
            "traj": traj_trans,
            "traj_pose": traj_pose,
            "style": style_name,
            "text": style_name,
            "joint_names": JOINT_NAMES,
        }
        data_list["motions"].append(motion_data)

    pickle.dump(data_list, open(export_path, "wb"))
    print("Exported:", export_path)
