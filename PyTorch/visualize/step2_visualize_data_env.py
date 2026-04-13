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
  python visualize/step2_visualize_data_env.py

  # Create augmented dataset
  python visualize/step2_visualize_data_env.py --create-dataset
  python visualize/step2_visualize_data_env.py --create-dataset --output data/pkls/lafan1_g1_env.pkl
"""

import os
import sys
import argparse
import time
import pickle

import numpy as np
import mujoco
import mujoco.viewer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import (
    draw_trajectory,
    draw_sensor_readings,
    draw_obstacle_box,
    draw_obstacle_circle,
)
from utils.environment_sensor import (
    EnvironmentSensor,
    ObstacleGenerator,
    compute_clip_sensor_readings,
    CircleObstacle,
    BoxObstacle,
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
        n_rays: int = 36,
        max_range: float = 3.0,
        obstacle_interval: int = 30,
        show_trajectory: bool = True,
        show_sensor: bool = True,
        show_obstacles: bool = True,
        past_frames: int = 10,
        future_frames: int = 45,
    ):
        self.model = model
        self.data = data
        self.dataset = dataset

        # sensor
        self.sensor = EnvironmentSensor(n_rays=n_rays, max_range=max_range)
        self.generator = ObstacleGenerator()
        self.obstacles = []
        self.readings = np.zeros(n_rays, dtype=np.float32)
        self.hit_points = np.zeros((n_rays, 2), dtype=np.float64)

        self.obstacle_interval = obstacle_interval
        self._last_obstacle_window = -1   # window index that was last generated

        # display flags
        self.show_trajectory = show_trajectory
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

        self.load_motion(0)

    # ------------------------------------------------------------------
    # Motion loading
    # ------------------------------------------------------------------

    def load_motion(self, motion_idx: int):
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion = self.dataset[self.current_motion_idx]
        self.current_frame = 0
        self._last_obstacle_window = -1  # force regeneration

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

        # Collect robot XY for current + lookahead window
        all_qpos = self.current_motion.get_all_qpos()
        window_xy = all_qpos[w_start:w_end, :2]
        lookahead_xy = all_qpos[w_end:la_end, :2] if la_end > w_end else None

        # Deterministic seed: (motion_idx * 1000 + win_idx)
        self.generator.seed(self.current_motion_idx * 1000 + win_idx)
        self.obstacles = self.generator.generate_for_window(window_xy, lookahead_xy)
        self._last_obstacle_window = win_idx

    def force_new_obstacles(self):
        """Manually regenerate obstacles (N key)."""
        # Use a random seed to get a different placement
        self.generator.seed(int(time.time() * 1000) % 1_000_000)
        win_idx = self._current_window_idx()
        w_start = win_idx * self.obstacle_interval
        w_end = min(w_start + self.obstacle_interval, self.current_motion.num_frames)
        la_end = min(w_end + self.obstacle_interval, self.current_motion.num_frames)
        all_qpos = self.current_motion.get_all_qpos()
        window_xy = all_qpos[w_start:w_end, :2]
        lookahead_xy = all_qpos[w_end:la_end, :2] if la_end > w_end else None
        self.obstacles = self.generator.generate_for_window(window_xy, lookahead_xy)
        print(f"Regenerated {len(self.obstacles)} obstacles")

    def update_pose(self):
        qpos = self.current_motion.get_qpos(self.current_frame)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Update obstacles if entering new window
        self._maybe_regenerate_obstacles()

        # Compute sensor readings
        robot_pos = qpos[:3]
        robot_yaw = quat_wxyz_to_yaw(qpos[3:7])
        self.readings, self.hit_points = self.sensor.compute(
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, scene):
        scene.ngeom = 0

        # trajectory
        if self.show_trajectory and hasattr(self, "past_traj"):
            if self.past_traj is not None and len(self.past_traj) > 0:
                draw_trajectory(scene, self.past_traj, self.past_orient,
                                color=[0.2, 0.5, 1.0, 1.0])
            if self.future_traj is not None and len(self.future_traj) > 0:
                draw_trajectory(scene, self.future_traj, self.future_orient,
                                color=[1.0, 0.2, 0.2, 1.0])

        # obstacles
        if self.show_obstacles:
            for obs in self.obstacles:
                if isinstance(obs, CircleObstacle):
                    draw_obstacle_circle(scene, obs.center, obs.radius, height=0.2)
                elif isinstance(obs, BoxObstacle):
                    draw_obstacle_box(scene, obs.center, obs.half_extents, obs.yaw, height=0.2)

        # sensor rays + dots
        if self.show_sensor:
            robot_pos = self.data.qpos[:3]
            draw_sensor_readings(
                scene, robot_pos, self.readings, self.hit_points,
                z_height=0.08, dot_radius=0.04,
                draw_lines=self.draw_sensor_lines,
            )

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

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
        n_hits = int(self.readings.sum())
        print(
            f"Motion {self.current_motion_idx+1}/{len(self.dataset)} | "
            f"Frame {self.current_frame}/{self.current_motion.num_frames} | "
            f"Style: {self.current_motion.style} | "
            f"{'Playing' if self.playing else 'Paused'} ({self.playback_speed}×) | "
            f"Sensor hits: {n_hits}/{self.sensor.n_rays} | "
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
    n_rays: int = 36,
    max_range: float = 3.0,
    obstacle_interval: int = 30,
    robot_safe_radius: float = 0.5,
    lookahead_frames: int = 30,
    seed: int = 0,
):
    """
    Build an obstacle-augmented dataset.

    Loads ``source_pkl``, generates time-varying obstacles for each clip,
    computes per-frame binary sensor readings, and writes ``output_pkl`` with
    a ``sensor_readings`` field added to every motion dict.

    The top-level dict gains metadata keys:
        ``sensor_n_rays``, ``sensor_max_range``, ``obstacle_interval``

    Args:
        source_pkl:        Path to source motion pkl (e.g. lafan1_g1.pkl).
        output_pkl:        Where to write the augmented pkl.
        n_rays:            Number of sensor rays.
        max_range:         Sensor maximum range (m).
        obstacle_interval: Frames between obstacle regeneration.
        robot_safe_radius: Minimum gap between obstacle surface and robot (m).
        lookahead_frames:  Future window also checked for collisions.
        seed:              Base RNG seed for reproducibility.
    """
    print(f"\nLoading source dataset: {source_pkl}")
    with open(source_pkl, "rb") as f:
        data_dict = pickle.load(f)

    sensor = EnvironmentSensor(n_rays=n_rays, max_range=max_range)
    generator = ObstacleGenerator(robot_safe_radius=robot_safe_radius, seed=seed)

    motions = data_dict["motions"]
    print(f"Processing {len(motions)} motion clips …")

    for clip_idx, motion in enumerate(tqdm(motions)):
        # Build full qpos array for this clip
        local_rot = motion["local_joint_rotations"]   # (T, 30, 4)
        root_pos  = motion["global_root_positions"]   # (T, 3)
        T = root_pos.shape[0]

        # qpos[t] = [xyz(3), wxyz(4), joint_angles(29)]
        all_qpos = np.concatenate([
            root_pos,
            local_rot[:, 0, :],        # root quaternion wxyz
            local_rot[:, 1:, 0],       # 29 joint angles
        ], axis=1).astype(np.float64)  # (T, 36)

        # Seed per clip for reproducibility
        generator.seed(seed * 10_000 + clip_idx)

        readings, _ = compute_clip_sensor_readings(
            all_qpos,
            sensor,
            generator,
            obstacle_interval=obstacle_interval,
            lookahead_frames=lookahead_frames,
        )
        motion["sensor_readings"] = readings.astype(np.float32)  # (T, n_rays)

    # Store metadata at top level
    data_dict["sensor_n_rays"] = n_rays
    data_dict["sensor_max_range"] = max_range
    data_dict["obstacle_interval"] = obstacle_interval

    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(data_dict, f)

    # Summary
    total_frames = sum(len(m["sensor_readings"]) for m in motions)
    hit_frames = sum(int((m["sensor_readings"].sum(axis=1) > 0).sum()) for m in motions)
    print(f"\nSaved augmented dataset → {output_pkl}")
    print(f"  Clips:              {len(motions)}")
    print(f"  Total frames:       {total_frames}")
    print(f"  Frames with ≥1 hit: {hit_frames} ({100*hit_frames/total_frames:.1f} %)")
    print(f"  Sensor:             {n_rays} rays, {max_range} m range")


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
    p.add_argument("--n-rays",   type=int,   default=36)
    p.add_argument("--max-range",type=float, default=3.0,
                   help="Sensor max range in metres (default: 3.0)")
    p.add_argument("--obstacle-interval", type=int, default=30,
                   help="Frames between obstacle regeneration (default: 30 = 1 s)")
    p.add_argument("--robot-safe-radius", type=float, default=0.5,
                   help="Minimum clear gap around robot path in metres (default: 0.5)")
    p.add_argument("--seed", type=int, default=42)

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
    p.add_argument("--output", default=None,
                   help="Output pkl path (default: data/pkls/<dataset>_env.pkl)")
    p.add_argument("--lookahead-frames", type=int, default=30)
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

def main():
    args = get_args()

    dataset_path = f"data/pkls/{args.dataset}.pkl"

    # ------------------------------------------------------------------ #
    #  Dataset creation mode                                              #
    # ------------------------------------------------------------------ #
    if args.create_dataset:
        output = args.output or f"data/pkls/{args.dataset}_env.pkl"
        if not os.path.exists(dataset_path):
            print(f"Source dataset not found: {dataset_path}")
            return
        create_env_dataset(
            source_pkl=dataset_path,
            output_pkl=output,
            n_rays=args.n_rays,
            max_range=args.max_range,
            obstacle_interval=args.obstacle_interval,
            robot_safe_radius=args.robot_safe_radius,
            lookahead_frames=args.lookahead_frames,
            seed=args.seed,
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

    print(f"\nEnvironment Sensor: {args.n_rays} rays, {args.max_range} m range")
    print(f"Obstacle interval:  {args.obstacle_interval} frames  "
          f"({args.obstacle_interval/30:.1f} s)")
    print(f"Robot safe radius:  {args.robot_safe_radius} m\n")

    player = SensorMotionPlayer(
        model, mj_data, dataset,
        n_rays=args.n_rays,
        max_range=args.max_range,
        obstacle_interval=args.obstacle_interval,
        show_trajectory=not args.no_trajectory,
        show_sensor=not args.no_sensor,
        show_obstacles=not args.no_obstacles,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
    )

    if args.motion > 0:
        player.load_motion(args.motion)

    print_instructions()

    with mujoco.viewer.launch_passive(
        model, mj_data,
        key_callback=lambda kc: key_callback(player, kc),
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            player.step()
            viewer.user_scn.ngeom = 0
            player.render(viewer.user_scn)
            viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
