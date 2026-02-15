"""
Step 2: Visualize Training Data (Object Version)
--------------------------------------------------
Load robot-object motion clips from npz files and play them in MuJoCo viewer.
Includes trajectory arrows (past/future) and object (largebox) visualization
with grasp detection coloring.

Controls:
- SPACE: Pause/Resume
- LEFT/RIGHT: Previous/Next frame
- UP/DOWN: Previous/Next motion clip
- R: Reset to first frame
- T: Toggle trajectory visualization
- 1-9: Change playback speed
- S: Print status
- ESC: Exit

Usage:
    python visualize/step2_visualize_data_object.py [--data-dir data/robot-object-mujoco]
"""

import os
import sys
import re
import argparse
import time
import glob
import numpy as np
import mujoco
import mujoco.viewer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.utils.geometry import draw_trajectory


def _natural_sort_key(text: str):
    parts = re.findall(r"\d+|\D+", text)
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def detect_grasp_from_motion(qpos_seq, motion_threshold=0.008, window_size=5):
    """Detect grasp by analyzing box motion. Box is 'in grasp' when moving or floating."""
    grasp_states = []
    box_positions = qpos_seq[:, 36:39]

    for i in range(len(qpos_seq)):
        if i < window_size:
            grasp_states.append(False)
        else:
            recent_positions = box_positions[i - window_size:i + 1]
            motion = np.max(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))
            in_grasp = motion >= motion_threshold or box_positions[i, 2] > 0.2
            grasp_states.append(in_grasp)

    return grasp_states


def update_box_color(model, in_grasp):
    """Update box color based on grasp state (red=grasped, orange=free)."""
    for i in range(model.ngeom):
        body_id = model.geom(i).bodyid
        if body_id < model.nbody:
            body_name = model.body(body_id).name
            if body_name and 'largebox' in body_name.lower():
                if in_grasp:
                    model.geom_rgba[i] = [1.0, 0.0, 0.0, 1.0]
                else:
                    model.geom_rgba[i] = [0.8, 0.5, 0.2, 1.0]
                break


class ObjectMotionClip:
    """Container for a single robot-object motion clip loaded from npz."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(filepath))[0]

        npz = np.load(filepath)
        self.qpos_seq = npz["qpos"]  # (T, 43)
        self.fps = int(npz["fps"])
        self.num_frames = len(self.qpos_seq)

        # Pre-compute grasp states
        self.grasp_states = detect_grasp_from_motion(self.qpos_seq)

        # Compute trajectory data from root positions
        self._compute_trajectory()

    def _compute_trajectory(self):
        """Compute trajectory positions and orientations from qpos data."""
        # Root XY positions for trajectory
        self.traj_xy = self.qpos_seq[:, :2].copy()  # (T, 2)
        # Extract yaw-only quaternion from root quat (WXYZ) so arrows stay on XY plane
        self.traj_quat = self._extract_yaw_quats(self.qpos_seq[:, 3:7])  # (T, 4)

    @staticmethod
    def _extract_yaw_quats(quats_wxyz):
        """Project quaternions to yaw-only (rotation around Z axis) in WXYZ format."""
        # For WXYZ quaternion, yaw component is: w=cos(yaw/2), z=sin(yaw/2)
        w = quats_wxyz[:, 0]
        z = quats_wxyz[:, 3]
        # Normalize the yaw-only quaternion
        norm = np.sqrt(w**2 + z**2)
        yaw_quats = np.zeros_like(quats_wxyz)
        yaw_quats[:, 0] = w / norm  # w
        yaw_quats[:, 3] = z / norm  # z
        return yaw_quats

    def get_qpos(self, frame_idx):
        frame_idx = min(frame_idx, self.num_frames - 1)
        return self.qpos_seq[frame_idx]

    def get_trajectory(self, frame_idx, past_frames=10, future_frames=45):
        """Get past and future trajectory for visualization."""
        past_start = max(0, frame_idx - past_frames)
        past_end = frame_idx

        future_start = frame_idx
        future_end = min(self.num_frames, frame_idx + future_frames)

        # XY trajectory with Z=0
        past_xy = self.traj_xy[past_start:past_end]
        future_xy = self.traj_xy[future_start:future_end]

        past_z = np.zeros((past_xy.shape[0], 1))
        future_z = np.zeros((future_xy.shape[0], 1))

        past_traj = np.concatenate([past_xy, past_z], axis=-1)
        future_traj = np.concatenate([future_xy, future_z], axis=-1)

        past_orient = self.traj_quat[past_start:past_end]
        future_orient = self.traj_quat[future_start:future_end]

        return past_traj, future_traj, past_orient, future_orient


class ObjectMotionDataset:
    """Dataset loader for robot-object npz files."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        pattern = os.path.join(data_dir, "*.npz")
        files = glob.glob(pattern)
        self.files = sorted(files, key=lambda p: _natural_sort_key(os.path.basename(p)))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        self.clips = [ObjectMotionClip(f) for f in self.files]
        print(f"Loaded {len(self.clips)} object motion clips from {data_dir}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx]

    def print_summary(self):
        print("\n" + "=" * 60)
        print(f"{'Object Motion Dataset Summary':^60}")
        print("=" * 60)
        print(f"Source dir: {self.data_dir}")
        print(f"Total clips: {len(self.clips)}")
        print(f"\n{'Name':<45} {'Frames':<10} {'FPS':<5}")
        print("-" * 60)
        for clip in self.clips:
            print(f"{clip.name:<45} {clip.num_frames:<10} {clip.fps:<5}")
        print("=" * 60 + "\n")


class ObjectMotionPlayer:
    """Interactive motion player for robot-object data."""

    def __init__(self, model, data, dataset, show_trajectory=True, past_frames=10, future_frames=45):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.show_trajectory = show_trajectory
        self.past_frames = past_frames
        self.future_frames = future_frames

        # Playback state
        self.current_motion_idx = 0
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()

        # Frame rate
        self.fps = 30
        self.frame_dt = 1.0 / self.fps

        # Load first motion
        self.load_motion(0)

    def load_motion(self, motion_idx):
        """Load a specific motion clip."""
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion = self.dataset[self.current_motion_idx]
        self.current_frame = 0
        self.fps = self.current_motion.fps
        self.frame_dt = 1.0 / self.fps

        grasp_count = sum(self.current_motion.grasp_states)
        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Name: {self.current_motion.name}")
        print(f"Frames: {self.current_motion.num_frames}")
        print(f"Duration: {self.current_motion.num_frames / self.fps:.2f}s")
        print(f"Grasp frames: {grasp_count}/{self.current_motion.num_frames}")
        print(f"{'='*60}\n")

        self.update_pose()

    def update_pose(self):
        """Update MuJoCo model with current frame pose."""
        qpos = self.current_motion.get_qpos(self.current_frame)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Update box color based on grasp state
        in_grasp = self.current_motion.grasp_states[self.current_frame]
        update_box_color(self.model, in_grasp)

        # Update trajectory visualization
        if self.show_trajectory:
            self._update_trajectory()

    def step(self):
        """Step the animation forward."""
        if not self.playing:
            return

        current_time = time.time()
        dt = current_time - self.last_update_time

        if dt >= self.frame_dt / self.playback_speed:
            self.current_frame += 1
            if self.current_frame >= self.current_motion.num_frames:
                self.current_frame = 0
            self.update_pose()
            self.last_update_time = current_time

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
        print(f"{'Playing' if self.playing else 'Paused'}")

    def reset(self):
        self.current_frame = 0
        self.update_pose()
        print("Reset to first frame")

    def set_speed(self, speed):
        self.playback_speed = speed
        print(f"Playback speed: {speed}x")

    def _update_trajectory(self):
        """Update trajectory visualization data."""
        past_traj, future_traj, past_orient, future_orient = \
            self.current_motion.get_trajectory(
                self.current_frame,
                self.past_frames,
                self.future_frames,
            )
        self.past_traj = past_traj
        self.future_traj = future_traj
        self.past_orient = past_orient
        self.future_orient = future_orient

    def render_trajectory(self, scene):
        if not self.show_trajectory:
            return
        if not hasattr(self, 'past_traj') or not hasattr(self, 'future_traj'):
            return
        # --- DRAW PAST (Blue) ---
        if len(self.past_traj) > 0:
            draw_trajectory(scene, self.past_traj, self.past_orient, color=[0.2, 0.5, 1.0, 1.0])
        # --- DRAW FUTURE (Red) ---
        if len(self.future_traj) > 0:
            draw_trajectory(scene, self.future_traj, self.future_orient, color=[1.0, 0.2, 0.2, 1.0])

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")

    def print_status(self):
        in_grasp = self.current_motion.grasp_states[self.current_frame]
        status = (
            f"Motion: {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"Frame: {self.current_frame}/{self.current_motion.num_frames} | "
            f"Name: {self.current_motion.name} | "
            f"Grasp: {'YES' if in_grasp else 'no'} | "
            f"{'Playing' if self.playing else 'Paused'} "
            f"({self.playback_speed}x) | "
            f"Traj: {'ON' if self.show_trajectory else 'OFF'}"
        )
        print(status)


def get_args():
    parser = argparse.ArgumentParser(description="Visualize robot-object training data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/robot-object-mujoco",
        help="Directory containing robot-object .npz files"
    )
    parser.add_argument(
        "--motion",
        type=int,
        default=0,
        help="Starting motion index"
    )
    parser.add_argument(
        "--past-frames",
        type=int,
        default=10,
        help="Number of past trajectory frames to visualize (default: 10)"
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=45,
        help="Number of future trajectory frames to visualize (default: 45)"
    )
    return parser.parse_args()


def print_instruction():
    print("\n" + "=" * 60)
    print("Controls:")
    print("-" * 60)
    print("  SPACE       : Pause/Resume")
    print("  LEFT/RIGHT  : Previous/Next frame (when paused)")
    print("  UP/DOWN     : Previous/Next motion clip")
    print("  R           : Reset to first frame")
    print("  T           : Toggle trajectory visualization")
    print("  1-9         : Set playback speed (1=0.25x, 5=1x, 9=2x)")
    print("  S           : Print status")
    print("  ESC         : Exit")
    print("=" * 60 + "\n")


def key_callback(player: ObjectMotionPlayer, keycode):
    if keycode == 32:  # SPACE
        player.toggle_pause()
    elif keycode == 265:  # UP
        player.next_motion()
    elif keycode == 264:  # DOWN
        player.prev_motion()
    elif keycode == 263:  # LEFT
        player.prev_frame()
    elif keycode == 262:  # RIGHT
        player.next_frame()
    elif keycode == ord('r') or keycode == ord('R'):
        player.reset()
    elif keycode == ord('t') or keycode == ord('T'):
        player.toggle_trajectory()
    elif keycode == ord('s') or keycode == ord('S'):
        player.print_status()
    elif ord('1') <= keycode <= ord('9'):
        speed_map = {
            ord('1'): 0.25,
            ord('2'): 0.5,
            ord('3'): 0.75,
            ord('4'): 0.9,
            ord('5'): 1.0,
            ord('6'): 1.25,
            ord('7'): 1.5,
            ord('8'): 1.75,
            ord('9'): 2.0,
        }
        player.set_speed(speed_map[keycode])


def main():
    args = get_args()

    scene_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "scene_object.xml"
    )

    print("=" * 60)
    print("Step 2: Visualizing Robot-Object Training Data")
    print("=" * 60)

    # Load MuJoCo model (robot + largebox)
    print(f"\nLoading MuJoCo scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Load object motion dataset
    print(f"\nLoading object motion data: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return

    dataset = ObjectMotionDataset(args.data_dir)
    dataset.print_summary()

    # Create motion player
    player = ObjectMotionPlayer(
        model, data, dataset,
        show_trajectory=True,
        past_frames=args.past_frames,
        future_frames=args.future_frames
    )

    # Start from specified motion
    if args.motion > 0:
        player.load_motion(args.motion)

    print_instruction()
    with mujoco.viewer.launch_passive(model, data, key_callback=lambda keycode: key_callback(player, keycode)) as viewer:
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
