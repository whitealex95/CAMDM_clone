"""
Step 2: Visualize Merged Single-Object Training Data (.pkl)
-------------------------------------------------------------
Loads merged walk + object motions from a single .pkl and renders in MuJoCo.
Supports:
- Active object manipulation clips (object visible, contact-colored)
- Walk-only padded clips (object hidden)

Usage:
    python visualize/step2_visualize_data_object_pkl.py --dataset data/pkls/merged_object_motion.pkl
"""

import argparse
import os
import pickle
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.utils.geometry import draw_trajectory


def set_box_color(model, state: str):
    if state == "grasp":
        rgba = [1.0, 0.0, 0.0, 1.0]
    elif state == "free":
        rgba = [0.8, 0.5, 0.2, 1.0]
    else:
        rgba = [0.5, 0.5, 0.5, 0.02]

    for i in range(model.ngeom):
        body_id = model.geom(i).bodyid
        if body_id < model.nbody:
            body_name = model.body(body_id).name
            if body_name and "largebox" in body_name.lower():
                model.geom_rgba[i] = rgba
                break


def _quat_wxyz_to_yaw_mat(q):
    yaw = R.from_quat(q[[1, 2, 3, 0]]).as_euler("zyx")[0]
    return R.from_euler("z", yaw).as_matrix()


def root_relative_object_to_world(root_pos, root_quat_wxyz, obj_rel_pose, frame_mode="root"):
    p_rel = obj_rel_pose[:3]
    r_rel = obj_rel_pose[3:].reshape(3, 3)

    if frame_mode != "root_yaw_gravity_aligned":
        raise ValueError("Unsupported object_pose_relative_frame. Expected 'root_yaw_gravity_aligned'.")
    r_root = _quat_wxyz_to_yaw_mat(root_quat_wxyz)
    r_obj = r_root @ r_rel
    p_obj = root_pos + r_root @ p_rel
    q_obj_xyzw = R.from_matrix(r_obj).as_quat()
    q_obj_wxyz = q_obj_xyzw[[3, 0, 1, 2]]
    return p_obj, q_obj_wxyz


class MergedObjectMotion:
    def __init__(self, motion):
        self.filepath = motion.get("filepath", "unknown")
        self.style = motion.get("style", "unknown")
        self.text = motion.get("text", self.style)
        self.source = motion.get("source", "unknown")
        self.has_object = bool(motion.get("has_object", False))
        self.object_pose_relative_frame = motion.get("object_pose_relative_frame", None)
        if self.object_pose_relative_frame != "root_yaw_gravity_aligned":
            raise ValueError(
                "Expected 'object_pose_relative_frame' == 'root_yaw_gravity_aligned'. "
                "Re-generate merged_object_motion.pkl with the latest script."
            )

        self.local_joint_rotations = motion["local_joint_rotations"].astype(np.float32)  # (T,30,4)
        self.global_root_positions = motion["global_root_positions"].astype(np.float32)  # (T,3)
        self.traj = motion.get("traj", None)
        self.traj_pose = motion.get("traj_pose", None)

        self.object_pose_relative = motion.get("object_pose_relative", None)
        if self.object_pose_relative is not None:
            self.object_pose_relative = np.asarray(self.object_pose_relative, dtype=np.float32)

        self.object_contact_mask = motion.get("object_contact_mask", None)
        if self.object_contact_mask is None:
            self.object_contact_mask = np.zeros((len(self.global_root_positions), 1), dtype=np.float32)
        else:
            self.object_contact_mask = np.asarray(self.object_contact_mask, dtype=np.float32).reshape(-1, 1)

        self.num_frames = len(self.global_root_positions)

    def get_qpos(self, frame_idx):
        frame_idx = min(frame_idx, self.num_frames - 1)
        root_pos = self.global_root_positions[frame_idx]
        root_quat = self.local_joint_rotations[frame_idx, 0]
        joint_angles = self.local_joint_rotations[frame_idx, 1:, 0]

        qpos = np.concatenate([root_pos, root_quat, joint_angles], axis=0)  # 36

        if self.has_object and self.object_pose_relative is not None:
            obj_rel = self.object_pose_relative[frame_idx]
            obj_pos, obj_quat = root_relative_object_to_world(
                root_pos, root_quat, obj_rel, frame_mode=self.object_pose_relative_frame
            )
            obj_qpos = np.concatenate([obj_pos, obj_quat], axis=0)  # 7
        else:
            obj_qpos = np.array([0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return np.concatenate([qpos, obj_qpos], axis=0)

    def get_contact(self, frame_idx):
        frame_idx = min(frame_idx, len(self.object_contact_mask) - 1)
        return float(self.object_contact_mask[frame_idx, 0])

    def get_trajectory(self, frame_idx, past_frames=10, future_frames=45):
        if self.traj is None or self.traj_pose is None:
            return None, None, None, None

        traj_xy = self.traj[0]
        traj_quat = self.traj_pose[0]

        past_start = max(0, frame_idx - past_frames)
        past_end = frame_idx
        future_start = frame_idx
        future_end = min(self.num_frames, frame_idx + future_frames)

        past_xy = traj_xy[past_start:past_end]
        future_xy = traj_xy[future_start:future_end]
        past_quat = traj_quat[past_start:past_end]
        future_quat = traj_quat[future_start:future_end]

        past_z = np.zeros((len(past_xy), 1), dtype=np.float32)
        future_z = np.zeros((len(future_xy), 1), dtype=np.float32)

        past_traj = np.concatenate([past_xy, past_z], axis=-1)
        future_traj = np.concatenate([future_xy, future_z], axis=-1)
        return past_traj, future_traj, past_quat, future_quat


class MergedObjectDataset:
    def __init__(self, pkl_path):
        print(f"Loading merged dataset: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.motions = [MergedObjectMotion(m) for m in data["motions"]]
        self.pkl_path = pkl_path

        if not self.motions:
            raise RuntimeError("No motions in dataset.")

        self.object_count = sum(int(m.has_object) for m in self.motions)
        self.walk_count = len(self.motions) - self.object_count
        print(f"Loaded {len(self.motions)} motions ({self.walk_count} walk-only, {self.object_count} object).")

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return self.motions[idx]


class MergedObjectPlayer:
    def __init__(self, model, data, dataset, show_trajectory=True, past_frames=10, future_frames=45):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.show_trajectory = show_trajectory
        self.past_frames = past_frames
        self.future_frames = future_frames

        self.current_motion_idx = 0
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()
        self.fps = 30
        self.frame_dt = 1.0 / self.fps

        self.load_motion(0)

    def load_motion(self, motion_idx):
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion = self.dataset[self.current_motion_idx]
        self.current_frame = 0

        print(f"\n{'=' * 60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Style: {self.current_motion.style}")
        print(f"Text: {self.current_motion.text}")
        print(f"Source: {self.current_motion.source}")
        print(f"Has object: {self.current_motion.has_object}")
        print(f"Frames: {self.current_motion.num_frames}")
        print(f"{'=' * 60}\n")
        self.update_pose()

    def update_pose(self):
        qpos = self.current_motion.get_qpos(self.current_frame)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        if not self.current_motion.has_object:
            set_box_color(self.model, "hidden")
        else:
            contact = self.current_motion.get_contact(self.current_frame)
            set_box_color(self.model, "grasp" if contact > 0.5 else "free")

        if self.show_trajectory:
            self._update_trajectory()

    def step(self):
        if not self.playing:
            return
        dt = time.time() - self.last_update_time
        if dt >= self.frame_dt / self.playback_speed:
            self.current_frame = (self.current_frame + 1) % self.current_motion.num_frames
            self.update_pose()
            self.last_update_time = time.time()

    def next_motion(self):
        self.load_motion(self.current_motion_idx + 1)

    def prev_motion(self):
        self.load_motion(self.current_motion_idx - 1)

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.current_motion.num_frames
        self.update_pose()

    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.current_motion.num_frames
        self.update_pose()

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def reset(self):
        self.current_frame = 0
        self.update_pose()
        print("Reset to first frame")

    def set_speed(self, speed):
        self.playback_speed = speed
        print(f"Playback speed: {speed}x")

    def _update_trajectory(self):
        out = self.current_motion.get_trajectory(self.current_frame, self.past_frames, self.future_frames)
        self.past_traj, self.future_traj, self.past_orient, self.future_orient = out

    def render_trajectory(self, scene):
        if not self.show_trajectory:
            return
        if self.past_traj is None or self.future_traj is None:
            return
        if len(self.past_traj) > 0:
            draw_trajectory(scene, self.past_traj, self.past_orient, color=[0.2, 0.5, 1.0, 1.0])
        if len(self.future_traj) > 0:
            draw_trajectory(scene, self.future_traj, self.future_orient, color=[1.0, 0.2, 0.2, 1.0])

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")

    def print_status(self):
        contact = self.current_motion.get_contact(self.current_frame)
        print(
            f"Motion: {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"Frame: {self.current_frame}/{self.current_motion.num_frames} | "
            f"Has object: {self.current_motion.has_object} | "
            f"Contact: {contact:.0f} | "
            f"{'Playing' if self.playing else 'Paused'} ({self.playback_speed}x)"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Visualize merged single-object dataset (.pkl)")
    parser.add_argument("--dataset", type=str, default="data/pkls/merged_object_motion.pkl")
    parser.add_argument("--motion", type=int, default=0)
    parser.add_argument("--past-frames", type=int, default=10)
    parser.add_argument("--future-frames", type=int, default=45)
    return parser.parse_args()


def print_instruction():
    print("\n" + "=" * 60)
    print("Controls:")
    print("-" * 60)
    print("  SPACE       : Pause/Resume")
    print("  LEFT/RIGHT  : Previous/Next frame")
    print("  UP/DOWN     : Previous/Next motion clip")
    print("  R           : Reset to first frame")
    print("  T           : Toggle trajectory visualization")
    print("  1-9         : Set playback speed")
    print("  S           : Print status")
    print("  ESC         : Exit")
    print("=" * 60 + "\n")


def key_callback(player, keycode):
    if keycode == 32:
        player.toggle_pause()
    elif keycode == 265:
        player.next_motion()
    elif keycode == 264:
        player.prev_motion()
    elif keycode == 263:
        player.prev_frame()
    elif keycode == 262:
        player.next_frame()
    elif keycode in (ord("r"), ord("R")):
        player.reset()
    elif keycode in (ord("t"), ord("T")):
        player.toggle_trajectory()
    elif keycode in (ord("s"), ord("S")):
        player.print_status()
    elif ord("1") <= keycode <= ord("9"):
        speed_map = {
            ord("1"): 0.25,
            ord("2"): 0.5,
            ord("3"): 0.75,
            ord("4"): 0.9,
            ord("5"): 1.0,
            ord("6"): 1.25,
            ord("7"): 1.5,
            ord("8"): 1.75,
            ord("9"): 2.0,
        }
        player.set_speed(speed_map[keycode])


def main():
    args = get_args()
    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene_object.xml")

    print("=" * 60)
    print("Step 2: Visualizing Merged Single-Object Training Data")
    print("=" * 60)
    print(f"Scene: {scene_path}")
    print(f"Dataset: {args.dataset}")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    dataset = MergedObjectDataset(args.dataset)

    player = MergedObjectPlayer(
        model, data, dataset, show_trajectory=True, past_frames=args.past_frames, future_frames=args.future_frames
    )
    if args.motion > 0:
        player.load_motion(args.motion)

    print_instruction()
    with mujoco.viewer.launch_passive(model, data, key_callback=lambda kc: key_callback(player, kc)) as viewer:
        viewer.sync()
        while viewer.is_running():
            player.step()
            viewer.user_scn.ngeom = 0
            player.render_trajectory(viewer.user_scn)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
