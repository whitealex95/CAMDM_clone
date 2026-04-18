"""
Step 2 (Env): Visualise Training Data + Environment Sensor
-----------------------------------------------------------
Single entry-point for two tasks that share the same EnvironmentSensor code:

  1. **Interactive visualisation** (default)
       Shows the LAFAN dataset with a 2-D scan-dot sensor overlaid.
       Obstacles are auto-generated for the current trajectory window and
       change every ``--obstacle-interval`` frames.
       All sensor + obstacle logic lives in ``utils/environment_sensor.py``
       so it is identical to what the training dataset and the future demo use.

  2. **Dataset creation** (``--create-dataset``)
       Processes every clip in the source pkl, augments it with time-varying
       random obstacles, computes per-frame sensor readings, and writes a new
       pkl with a ``sensor_readings`` field added to every motion dict.

Controls (visualisation mode)
-----------------------------
  SPACE       : Pause / Resume
  LEFT / RIGHT: Previous / Next frame (when paused)
  UP   / DOWN : Previous / Next motion clip
  R           : Reset to first frame
  T           : Toggle trajectory visualisation
  C           : Toggle command trajectory (green linear)
  E           : Toggle environment-sensor overlay
  L           : Toggle ray lines (dots only ↔ lines+dots)
  O           : Toggle obstacle display
  N           : Force new obstacle set (regenerate now)
  1-9         : Set playback speed (1=0.25×, 5=1×, 9=2×)
  S           : Print status
  ESC         : Exit

Usage
-----
  # Visualise with obstacles
  python visualize/step2_visualize_data_env2d.py

  # Create augmented dataset
  python visualize/step2_visualize_data_env2d.py --create-dataset
  python visualize/step2_visualize_data_env2d.py --create-dataset --output data/pkls/lafan1_g1_env2d.pkl
"""

import os
import sys
import argparse
import time
import pickle

import numpy as np

from typing import List, Union
import mujoco
import mujoco.viewer
import imageio.v2 as imageio
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset
from visualize.utils.detour import compute_detour_obstacles
from visualize.utils.geometry import (
    draw_trajectory,
    draw_sensor_readings,
    draw_obstacle_box,
    draw_obstacle_circle,
)
from utils.environment_sensor import (
    EnvironmentSensor,
    make_generator,
    compute_clip_sensor_readings,
    CircleObstacle,
    BoxObstacle,
    Obstacle2D,
    quat_wxyz_to_yaw,
)


# ---------------------------------------------------------------------------
# Interactive player
# ---------------------------------------------------------------------------

class SensorMotionPlayer:
    """
    Motion player with a live EnvironmentSensor + randomly generated obstacles.

    Obstacles are regenerated every ``obstacle_interval`` frames so the sensor
    produces varied readings as the robot walks.  The generation is seeded per
    (motion_idx, window_idx) for reproducibility.
    """

    def __init__(
        self,
        model,
        data,
        dataset,
        max_range: float = 2.0,
        resolution: int = 9,
        obstacle_interval: int = 30,
        obstacle_mode: Union[str, List[str]] = 'sparse',
        show_trajectory: bool = True,
        show_sensor: bool = True,
        show_obstacles: bool = True,
        past_frames: int = 10,
        future_frames: int = 45,
        min_start_velocity: float = None,
        robot_safe_radius: float = 0.25,
        max_div_obstacles: int = 3,
    ):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.robot_safe_radius = robot_safe_radius
        self.max_div_obstacles = max_div_obstacles

        # sensor
        self.sensor = EnvironmentSensor(max_range=max_range, resolution=resolution)

        # Decompose new-style mode names into (detour_flags, random_modes)
        if isinstance(obstacle_mode, str):
            obstacle_mode = [obstacle_mode]
        self._detour_flags = [m != "none" for m in obstacle_mode]
        random_modes = [_MODE_TO_RANDOM[m] for m in obstacle_mode]
        self.generator = make_generator(random_modes, robot_safe_radius=robot_safe_radius)

        self.obstacles = []
        self.readings = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        self.sphere_centers = np.zeros((self.sensor.feature_dim, 2), dtype=np.float64)

        self.obstacle_interval = obstacle_interval
        self._last_obstacle_window = -1   # window index that was last generated
        self._has_detour = False           # True when detour obstacles exist in current window

        # display flags
        self.show_trajectory = show_trajectory
        self.show_command_traj = True
        self.show_sensor = show_sensor
        self.show_obstacles = show_obstacles
        self.draw_sensor_lines = True

        # playback state
        self.current_motion_idx = 0
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()
        self.fps = 30
        self.frame_dt = 1.0 / self.fps

        self.past_frames = past_frames
        self.future_frames = future_frames
        self.min_start_velocity = min_start_velocity

        self.load_motion(0)

    # ------------------------------------------------------------------
    # Motion loading
    # ------------------------------------------------------------------

    def load_motion(self, motion_idx: int):
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion = self.dataset[self.current_motion_idx]
        self._last_obstacle_window = -1  # force regeneration

        # Skip initial T-pose / low-velocity frames
        if self.min_start_velocity is not None:
            root_xy = self.current_motion.global_root_positions[:, :2].astype(np.float64)
            vel = np.linalg.norm(np.diff(root_xy, axis=0), axis=1)
            smoothed = np.convolve(vel, np.ones(5) / 5, mode='same')
            active = np.where(smoothed > self.min_start_velocity)[0]
            self.current_frame = int(active[0]) if len(active) > 0 else 0
        else:
            self.current_frame = 0

        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Style:  {self.current_motion.style}")
        print(f"Frames: {self.current_motion.num_frames}  "
              f"({self.current_motion.num_frames / self.fps:.2f} s)")
        print(f"{'='*60}\n")

        self.update_pose()

    def next_motion(self): self.load_motion(self.current_motion_idx + 1)
    def prev_motion(self): self.load_motion(self.current_motion_idx - 1)

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.current_motion.num_frames
        self.update_pose()

    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.current_motion.num_frames
        self.update_pose()

    def reset(self):
        self.current_frame = 0
        self._last_obstacle_window = -1
        self.update_pose()
        print("Reset to first frame")

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def set_speed(self, speed: float):
        self.playback_speed = speed
        print(f"Playback speed: {speed}×")

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def _current_window_idx(self) -> int:
        return self.current_frame // self.obstacle_interval

    def _maybe_regenerate_obstacles(self):
        """Regenerate obstacles when entering a new window."""
        win_idx = self._current_window_idx()
        if win_idx == self._last_obstacle_window:
            return  # still in same window
        self._regenerate_obstacles(win_idx)

    def _regenerate_obstacles(self, win_idx: int = None):
        """Generate a fresh set of obstacles for the current window."""
        if win_idx is None:
            win_idx = self._current_window_idx()

        w_start = win_idx * self.obstacle_interval
        w_end = min(w_start + self.obstacle_interval, self.current_motion.num_frames)
        la_end = min(w_end + self.obstacle_interval, self.current_motion.num_frames)

        # Divergence check must cover the full future_traj window (future_frames),
        # which can extend beyond la_end when current_frame is late in the window.
        div_end = min(w_end + self.future_frames, self.current_motion.num_frames)

        # Collect robot XY for current + lookahead window
        all_qpos = self.current_motion.get_all_qpos()
        window_xy  = all_qpos[w_start:w_end, :2]
        lookahead_xy = all_qpos[w_end:la_end, :2] if la_end > w_end else None
        div_xy = all_qpos[w_start:div_end, :2]   # window + future_frames lookahead

        # Deterministic seed: (motion_idx * 1000 + win_idx)
        self.generator.seed(self.current_motion_idx * 1000 + win_idx)
        self.obstacles = self.generator.generate_for_window(window_xy, lookahead_xy)
        use_detour = self._detour_flags[win_idx % len(self._detour_flags)]
        div_obs = self._detour_obstacles(div_xy) if use_detour else []
        self._has_detour = len(div_obs) > 0
        self.obstacles.extend(div_obs)
        self._last_obstacle_window = win_idx

    def force_new_obstacles(self):
        """Manually regenerate obstacles (N key)."""
        # Use a random seed to get a different placement
        self.generator.seed(int(time.time() * 1000) % 1_000_000)
        win_idx = self._current_window_idx()
        w_start = win_idx * self.obstacle_interval
        w_end = min(w_start + self.obstacle_interval, self.current_motion.num_frames)
        la_end = min(w_end + self.obstacle_interval, self.current_motion.num_frames)
        div_end = min(w_end + self.future_frames, self.current_motion.num_frames)
        all_qpos = self.current_motion.get_all_qpos()
        window_xy = all_qpos[w_start:w_end, :2]
        lookahead_xy = all_qpos[w_end:la_end, :2] if la_end > w_end else None
        div_xy = all_qpos[w_start:div_end, :2]
        self.obstacles = self.generator.generate_for_window(window_xy, lookahead_xy)
        use_detour = self._detour_flags[win_idx % len(self._detour_flags)]
        div_obs = self._detour_obstacles(div_xy) if use_detour else []
        self._has_detour = len(div_obs) > 0
        self.obstacles.extend(div_obs)
        print(f"Regenerated {len(self.obstacles)} obstacles "
              f"({len(div_obs)} from trajectory detour)")

    # ------------------------------------------------------------------
    # Divergence obstacle placement
    # ------------------------------------------------------------------

    def _detour_obstacles(self, path_xy: np.ndarray) -> List[Obstacle2D]:
        """Delegate to the shared compute_detour_obstacles utility."""
        obstacles, _ = compute_detour_obstacles(
            path_xy, robot_safe_radius=self.robot_safe_radius
        )
        for k, obs in enumerate(obstacles):
            tag = "off-green" if k == 0 else "on-green"
            print(f"  [div #{k+1} {tag}] r={obs.radius:.2f}m  center={obs.center.round(3)}")
        return obstacles

    def update_pose(self):
        qpos = self.current_motion.get_qpos(self.current_frame)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Update obstacles if entering new window
        self._maybe_regenerate_obstacles()

        # Compute sensor readings
        robot_pos = qpos[:3]
        robot_yaw = quat_wxyz_to_yaw(qpos[3:7])
        self.readings, self.sphere_centers = self.sensor.compute(
            robot_pos, robot_yaw, self.obstacles
        )

        if self.show_trajectory:
            self._cache_trajectory()

    def step(self):
        if not self.playing:
            return
        now = time.time()
        if now - self.last_update_time >= self.frame_dt / self.playback_speed:
            self.current_frame = (self.current_frame + 1) % self.current_motion.num_frames
            self.update_pose()
            self.last_update_time = now

    # ------------------------------------------------------------------
    # Trajectory cache
    # ------------------------------------------------------------------

    def _cache_trajectory(self):
        result = self.current_motion.get_trajectory(
            self.current_frame, self.past_frames, self.future_frames, kernel_idx=0
        )
        self.past_traj, self.future_traj, self.past_orient, self.future_orient = result
        self.command_traj, self.command_orient = self._compute_command_trajectory(
            self.future_traj, self.future_orient
        )

    def _compute_command_trajectory(self, future_traj, future_orient):
        """
        Command trajectory:
          - No detour this window → same as dataset target (future_traj as-is).
          - Detour detected       → linear interpolation start→end (forces robot
                                    to navigate past the detour obstacle).
        Orientation is always taken from the dataset.
        """
        if future_traj is None or len(future_traj) < 2:
            return None, None

        if not self._has_detour:
            return future_traj, future_orient

        N = len(future_traj)
        t = np.linspace(0.0, 1.0, N)
        start_pos    = future_traj[0]
        end_pos      = future_traj[-1]
        command_traj = start_pos[None] + t[:, None] * (end_pos - start_pos)[None]  # (N, 3)
        return command_traj, future_orient

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, scene, clear: bool = True):
        if clear:
            scene.ngeom = 0

        # trajectory
        if self.show_trajectory and hasattr(self, "past_traj"):
            if self.past_traj is not None and len(self.past_traj) > 0:
                draw_trajectory(scene, self.past_traj, self.past_orient,
                                color=[0.2, 0.5, 1.0, 1.0])
            if self.future_traj is not None and len(self.future_traj) > 0:
                draw_trajectory(scene, self.future_traj, self.future_orient,
                                color=[1.0, 0.2, 0.2, 1.0])   # red: target trajectory
            if self.show_command_traj and \
               self.command_traj is not None and len(self.command_traj) > 0:
                draw_trajectory(scene, self.command_traj, self.command_orient,
                                color=[0.1, 0.9, 0.2, 1.0])   # green: command trajectory

        # obstacles
        if self.show_obstacles:
            for obs in self.obstacles:
                if isinstance(obs, CircleObstacle):
                    draw_obstacle_circle(scene, obs.center, obs.radius, height=0.2)
                elif isinstance(obs, BoxObstacle):
                    draw_obstacle_box(scene, obs.center, obs.half_extents, obs.yaw, height=0.2)

        # sensor spheres
        if self.show_sensor:
            robot_pos = self.data.qpos[:3]
            draw_sensor_readings(
                scene, robot_pos, self.readings, self.sphere_centers,
                z_height=0.08, dot_radius=0.04,
                draw_lines=self.draw_sensor_lines,
            )

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

    def toggle_command_trajectory(self):
        self.show_command_traj = not self.show_command_traj
        print(f"Command trajectory: {'ON' if self.show_command_traj else 'OFF'}")

    def toggle_sensor(self):
        self.show_sensor = not self.show_sensor
        print(f"Sensor: {'ON' if self.show_sensor else 'OFF'}")

    def toggle_sensor_lines(self):
        self.draw_sensor_lines = not self.draw_sensor_lines
        print(f"Sensor lines: {'ON' if self.draw_sensor_lines else 'OFF'}")

    def toggle_obstacles(self):
        self.show_obstacles = not self.show_obstacles
        print(f"Obstacles: {'ON' if self.show_obstacles else 'OFF'}")

    def print_status(self):
        n_active = int((self.readings > 0).sum())
        print(
            f"Motion {self.current_motion_idx+1}/{len(self.dataset)} | "
            f"Frame {self.current_frame}/{self.current_motion.num_frames} | "
            f"Style: {self.current_motion.style} | "
            f"{'Playing' if self.playing else 'Paused'} ({self.playback_speed}×) | "
            f"Sensor active: {n_active}/{self.sensor.feature_dim} | "
            f"Obstacles: {len(self.obstacles)}"
        )


# ---------------------------------------------------------------------------
# Keyboard callback
# ---------------------------------------------------------------------------

SPEED_MAP = {
    ord('1'): 0.25, ord('2'): 0.50, ord('3'): 0.75, ord('4'): 0.90,
    ord('5'): 1.00, ord('6'): 1.25, ord('7'): 1.50, ord('8'): 1.75,
    ord('9'): 2.00,
}


def key_callback(player: SensorMotionPlayer, keycode: int):
    if keycode == 32:                          # SPACE
        player.toggle_pause()
    elif keycode == 265:                       # UP
        player.next_motion()
    elif keycode == 264:                       # DOWN
        player.prev_motion()
    elif keycode == 263:                       # LEFT
        player.prev_frame()
    elif keycode == 262:                       # RIGHT
        player.next_frame()
    elif keycode in (ord('r'), ord('R')):
        player.reset()
    elif keycode in (ord('t'), ord('T')):
        player.toggle_trajectory()
    elif keycode in (ord('c'), ord('C')):
        player.toggle_command_trajectory()
    elif keycode in (ord('e'), ord('E')):
        player.toggle_sensor()
    elif keycode in (ord('l'), ord('L')):
        player.toggle_sensor_lines()
    elif keycode in (ord('o'), ord('O')):
        player.toggle_obstacles()
    elif keycode in (ord('n'), ord('N')):
        player.force_new_obstacles()
    elif keycode in (ord('s'), ord('S')):
        player.print_status()
    elif keycode in SPEED_MAP:
        player.set_speed(SPEED_MAP[keycode])


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_env_dataset(
    source_pkl: str,
    output_pkl: str,
    max_range: float = 2.0,
    resolution: int = 9,
    obstacle_interval: int = 30,
    robot_safe_radius: float = 0.5,
    lookahead_frames: int = 45,
    seed: int = 0,
    mode: Union[str, List[str]] = 'sparse',
):
    """
    Build an obstacle-augmented dataset with NSM Cylindrical sensor readings.

    Loads ``source_pkl``, generates time-varying obstacles for each clip,
    computes per-frame continuous occupancy readings, and writes ``output_pkl``
    with a ``sensor_readings`` field added to every motion dict.

    When multiple modes are given (e.g. ``['sparse', 'dense', 'compact', 'none']``),
    each source clip is duplicated once per mode so every motion is represented with
    every obstacle configuration.  Cycling across modes within a single clip is
    intentionally avoided here — each output clip has a single consistent mode.

    The top-level dict gains metadata keys:
        ``env_sensor_dim``, ``sensor_resolution``, ``sensor_max_range``,
        ``sensor_sphere_radius``, ``obstacle_interval``

    Args:
        source_pkl:        Path to source motion pkl (e.g. lafan1_g1.pkl).
        output_pkl:        Where to write the augmented pkl.
        max_range:         Sensor maximum range in metres (= Size/2 in paper).
        resolution:        Number of radial rings (paper default: 10).
        obstacle_interval: Frames between obstacle regeneration.
        robot_safe_radius: Minimum gap between obstacle surface and robot (m).
        lookahead_frames:  Future window also checked for collisions.
        seed:              Base RNG seed for reproducibility.
        mode:              Single mode string or '|'-joined / list of modes.
                           Multiple modes → each clip is duplicated per mode.
    """
    print(f"\nLoading source dataset: {source_pkl}")
    with open(source_pkl, "rb") as f:
        data_dict = pickle.load(f)

    sensor = EnvironmentSensor(max_range=max_range, resolution=resolution)

    # Normalise mode to a list
    if isinstance(mode, str):
        modes = [m.strip() for m in mode.split("|") if m.strip()]
    else:
        modes = list(mode)

    source_motions = data_dict["motions"]
    print(f"Source clips: {len(source_motions)}, modes: {modes}")
    print(f"  → output clips: {len(source_motions) * len(modes)}")
    print(f"  Sensor: max_range={max_range}m, resolution={resolution}, "
          f"coverage={sensor._coverage:.4f}m, sphere_r={sensor.sphere_radius:.4f}m, "
          f"feature_dim={sensor.feature_dim}")

    output_motions = []
    for m_idx, mode_str in enumerate(modes):
        use_detour  = mode_str != "none"
        random_mode = _MODE_TO_RANDOM[mode_str]
        generator   = make_generator(random_mode, robot_safe_radius=robot_safe_radius,
                                     seed=seed)
        detour_fn = (
            (lambda xy: compute_detour_obstacles(xy, robot_safe_radius=robot_safe_radius)[0])
            if use_detour else None
        )
        print(f"\n[{m_idx+1}/{len(modes)}] mode='{mode_str}'  "
              f"(random={random_mode}, detour={'yes' if use_detour else 'no'}) …")
        for clip_idx, motion in enumerate(tqdm(source_motions)):
            local_rot = motion["local_joint_rotations"]   # (T, 30, 4)
            root_pos  = motion["global_root_positions"]   # (T, 3)

            all_qpos = np.concatenate([
                root_pos,
                local_rot[:, 0, :],
                local_rot[:, 1:, 0],
            ], axis=1).astype(np.float64)  # (T, 36)

            generator.seed(seed + clip_idx)

            readings, _, detour_flags = compute_clip_sensor_readings(
                all_qpos,
                sensor,
                generator,
                obstacle_interval=obstacle_interval,
                lookahead_frames=lookahead_frames,
                detour_fn=detour_fn,
            )
            new_motion = dict(motion)
            new_motion["sensor_readings"]      = readings.astype(np.float32)
            new_motion["command_detour_flags"] = detour_flags          # (T,) bool
            new_motion["obstacle_mode"]        = mode_str
            output_motions.append(new_motion)

    data_dict["motions"] = output_motions

    # Store metadata at top level
    data_dict["env_sensor_dim"]        = sensor.feature_dim
    data_dict["sensor_resolution"]    = resolution
    data_dict["sensor_max_range"]     = max_range
    data_dict["sensor_sphere_radius"] = sensor.sphere_radius
    data_dict["obstacle_interval"]    = obstacle_interval
    data_dict["obstacle_mode"]        = modes

    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(data_dict, f)

    total_frames  = sum(len(m["sensor_readings"]) for m in output_motions)
    active_frames = sum(int((m["sensor_readings"].max(axis=1) > 0).sum()) for m in output_motions)
    print(f"\nSaved augmented dataset → {output_pkl}")
    print(f"  Source clips:          {len(source_motions)}")
    print(f"  Output clips:          {len(output_motions)}  ({len(modes)} modes × {len(source_motions)})")
    print(f"  Total frames:          {total_frames}")
    print(f"  Frames with occupancy: {active_frames} ({100*active_frames/total_frames:.1f} %)")
    print(f"  Sensor: max_range={max_range}m, resolution={resolution} "
          f"→ {sensor.feature_dim} spheres")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Visualise LAFAN data with Environment Sensor / create augmented dataset"
    )
    # general
    p.add_argument("--dataset",  default="lafan1_g1",
                   help="Source dataset name (default: lafan1_g1)")
    p.add_argument("--resolution", type=int, default=9,
                   help="Number of radial rings (paper: 9)")
    p.add_argument("--max-range", type=float, default=2.0,
                   help="Sensor max range in metres = Size/2 (paper: Size=4 → 2.0m)")
    p.add_argument("--obstacle-interval", type=int, default=30,
                   help="Frames between obstacle regeneration (default: 30 = 1 s)")
    p.add_argument("--robot-safe-radius", type=float, default=0.25,
                   help="Minimum clear gap around robot path in metres (default: 0.5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="detour_only|detour_sparse|detour_dense|detour_packed|none",
                   help="Obstacle mode(s), separated by '|'. "
                        "Choices: detour_only, detour_sparse, detour_dense, detour_packed, none. "
                        "Modes prefixed with 'detour_' add trajectory-detour obstacles; "
                        "the suffix selects additional random obstacles "
                        "(detour_only=none, detour_sparse, detour_dense, detour_packed). "
                        "'none' places no obstacles at all. "
                        "Visualiser: cycles modes across obstacle windows. "
                        "Dataset creation: each clip is duplicated once per mode. "
                        "Default: 'detour_only|detour_sparse|detour_dense|detour_packed|none'")

    p.add_argument("--min-start-velocity", type=float, default=0.008,
                   help="Skip initial T-pose frames: first frame whose 5-frame "
                        "smoothed root-XY speed (m/frame) exceeds this threshold "
                        "is used as the start frame. Set 0 to disable. "
                        "(default: 0.008 ≈ 0.24 m/s @ 30 fps)")

    # visualisation-only
    p.add_argument("--motion",   type=int, default=0)
    p.add_argument("--past-frames",   type=int, default=10)
    p.add_argument("--future-frames", type=int, default=45)
    p.add_argument("--no-sensor",     action="store_true")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--no-obstacles",  action="store_true",
                   help="Start with obstacle display OFF (sensor still active)")

    # dataset creation
    p.add_argument("--create-dataset", action="store_true",
                   help="Create augmented dataset pkl instead of launching viewer")
    p.add_argument("--input", default=None,
                   help="Source pkl path for dataset creation "
                        "(overrides --dataset; default: data/pkls/<dataset>.pkl)")
    p.add_argument("--output", default=None,
                   help="Output pkl path (default: data/pkls/<dataset>_env2d_<modes>.pkl)")
    p.add_argument("--lookahead-frames", type=int, default=45)
    p.add_argument("--no-video", action="store_true",
                   help="Disable MP4 recording (visualisation mode only)")
    p.add_argument("--video-width",  type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    return p.parse_args()


def print_instructions():
    print("\n" + "=" * 60)
    print("  Environment-Sensor Visualiser  –  Controls")
    print("-" * 60)
    print("  SPACE       : Pause / Resume")
    print("  LEFT/RIGHT  : Prev / Next frame (when paused)")
    print("  UP/DOWN     : Prev / Next motion clip")
    print("  R           : Reset to first frame")
    print("  T           : Toggle trajectory")
    print("  C           : Toggle command trajectory (green linear interp)")
    print("  E           : Toggle sensor overlay")
    print("  L           : Toggle ray lines (dots only ↔ lines+dots)")
    print("  O           : Toggle obstacle display")
    print("  N           : Regenerate obstacles now")
    print("  1-9         : Playback speed (1=0.25×, 5=1×, 9=2×)")
    print("  S           : Print status")
    print("  ESC         : Exit")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# User-facing mode names
_VALID_MODES = {"detour_only", "detour_sparse", "detour_dense", "detour_packed", "none"}

# Maps each user-facing mode to the underlying ObstacleGenerator (random) mode
_MODE_TO_RANDOM = {
    "detour_only":   "none",
    "detour_sparse": "sparse",
    "detour_dense":  "dense",
    "detour_packed": "compact",
    "none":          "none",
}


def _parse_modes(raw: str) -> List[str]:
    """Parse a '|'-delimited mode string and validate each token."""
    modes = [m.strip() for m in raw.split("|") if m.strip()]
    invalid = [m for m in modes if m not in _VALID_MODES]
    if invalid:
        raise ValueError(f"Unknown mode(s): {invalid}. Valid: {sorted(_VALID_MODES)}")
    return modes


def main():
    args = get_args()

    # Parse '|'-delimited mode string → list
    modes = _parse_modes(args.mode)

    dataset_path = f"data/pkls/{args.dataset}.pkl"

    # ------------------------------------------------------------------ #
    #  Dataset creation mode                                              #
    # ------------------------------------------------------------------ #
    if args.create_dataset:
        source_pkl = args.input or dataset_path
        mode_tag = args.mode.replace("|", "_")
        stem = os.path.splitext(os.path.basename(source_pkl))[0]
        output = args.output or f"data/pkls/{stem}_env2d_{mode_tag}.pkl"
        if not os.path.exists(source_pkl):
            print(f"Source dataset not found: {source_pkl}")
            return
        create_env_dataset(
            source_pkl=source_pkl,
            output_pkl=output,
            max_range=args.max_range,
            resolution=args.resolution,
            obstacle_interval=args.obstacle_interval,
            robot_safe_radius=args.robot_safe_radius,
            lookahead_frames=args.lookahead_frames,
            seed=args.seed,
            mode=modes,
        )
        return

    # ------------------------------------------------------------------ #
    #  Interactive visualisation mode                                     #
    # ------------------------------------------------------------------ #
    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")

    print("=" * 60)
    print("  Step 2 (Env): Data + Environment Sensor Visualiser")
    print("=" * 60)

    print(f"\nLoading MuJoCo scene: {scene_path}")
    model    = mujoco.MjModel.from_xml_path(scene_path)
    mj_data  = mujoco.MjData(model)

    print(f"Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"  Dataset not found: {dataset_path}")
        return
    dataset = MotionDataset(dataset_path)
    dataset.print_summary()

    tmp_sensor = EnvironmentSensor(max_range=args.max_range, resolution=args.resolution)
    mode_str = " | ".join(modes)
    print(f"\nNSM Cylindrical Sensor: max_range={args.max_range}m  resolution={args.resolution}  "
          f"coverage={tmp_sensor._coverage:.4f}m  sphere_r={tmp_sensor.sphere_radius:.4f}m  "
          f"→ {tmp_sensor.feature_dim} spheres")
    print(f"Obstacle mode:      {mode_str}{' (cycling)' if len(modes) > 1 else ''}")
    print(f"Obstacle interval:  {args.obstacle_interval} frames  "
          f"({args.obstacle_interval/30:.1f} s)")
    print(f"Robot safe radius:  {args.robot_safe_radius} m\n")

    player = SensorMotionPlayer(
        model, mj_data, dataset,
        max_range=args.max_range,
        resolution=args.resolution,
        obstacle_interval=args.obstacle_interval,
        obstacle_mode=modes,
        show_trajectory=not args.no_trajectory,
        show_sensor=not args.no_sensor,
        show_obstacles=not args.no_obstacles,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        min_start_velocity=args.min_start_velocity if args.min_start_velocity > 0 else None,
        robot_safe_radius=args.robot_safe_radius,
    )

    if args.motion > 0:
        player.load_motion(args.motion)

    # ── Video recording setup ──────────────────────────────────────────────
    writer   = None
    renderer = None
    frame_last_time = -np.inf
    if not args.no_video:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/step2_{time.strftime('%m%d_%H%M')}.mp4"
        W, H = args.video_width, args.video_height
        writer   = imageio.get_writer(video_path, fps=player.fps,
                                      codec="libx264", pixelformat="yuv420p")
        renderer = mujoco.Renderer(model, height=H, width=W)
        print(f"Recording → {video_path}  ({W}×{H} @ {player.fps}fps)")

    print_instructions()

    with mujoco.viewer.launch_passive(
        model, mj_data,
        key_callback=lambda kc: key_callback(player, kc),
    ) as viewer:
        viewer.sync()
        try:
            while viewer.is_running():
                player.step()
                viewer.user_scn.ngeom = 0
                player.render(viewer.user_scn)
                viewer.cam.lookat[:] = mj_data.qpos[:3]
                viewer.sync()

                if writer is not None and time.time() - frame_last_time > 1.0 / player.fps:
                    renderer.update_scene(mj_data, camera=viewer.cam)
                    player.render(renderer.scene, clear=False)
                    writer.append_data(renderer.render())
                    frame_last_time = time.time()

                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            if writer is not None:
                writer.close()
                print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()
