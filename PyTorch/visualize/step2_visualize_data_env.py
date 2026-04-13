"""
Step 2 (Env): Visualise Training Data + Environment Sensor
-----------------------------------------------------------
Extends step2_visualize_data.py with a 2-D circular scan-dot sensor
drawn around the robot at every frame.

In this first version (Step 1) there are NO obstacles – all sensor
readings are 0 (free) and the scan dots form a uniform green ring at
the sensor's maximum range.  Obstacles will be added in Step 2.

Controls
--------
  SPACE       : Pause / Resume
  LEFT / RIGHT: Previous / Next frame (when paused)
  UP   / DOWN : Previous / Next motion clip
  R           : Reset to first frame
  T           : Toggle trajectory visualisation
  E           : Toggle environment-sensor visualisation
  L           : Toggle ray lines (dots only / lines + dots)
  1-9         : Set playback speed (1 = 0.25×, 5 = 1×, 9 = 2×)
  S           : Print status
  ESC         : Exit

Usage
-----
    python visualize/step2_visualize_data_env.py [--dataset lafan1_g1]
    python visualize/step2_visualize_data_env.py --n-rays 36 --max-range 3.0
"""

import os
import sys
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import (
    draw_trajectory,
    draw_sensor_readings,
    draw_obstacle_box,
    draw_obstacle_circle,
)
from utils.environment_sensor import EnvironmentSensor, quat_wxyz_to_yaw


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class SensorMotionPlayer:
    """
    Interactive motion player that overlays an EnvironmentSensor visualisation.

    Obstacles are stored in ``self.obstacles`` (empty list by default).
    Subclasses or scripts can populate this list to test obstacle detection.
    """

    def __init__(
        self,
        model,
        data,
        dataset,
        n_rays: int = 36,
        max_range: float = 3.0,
        show_trajectory: bool = True,
        show_sensor: bool = True,
        past_frames: int = 10,
        future_frames: int = 45,
    ):
        self.model = model
        self.data = data
        self.dataset = dataset

        # ---- sensor ----
        self.sensor = EnvironmentSensor(n_rays=n_rays, max_range=max_range)
        self.obstacles = []          # Empty for Step 1 (all readings = 0)
        self.readings = np.zeros(n_rays, dtype=np.float32)
        self.hit_points = np.zeros((n_rays, 2), dtype=np.float64)

        # ---- display flags ----
        self.show_trajectory = show_trajectory
        self.show_sensor = show_sensor
        self.draw_sensor_lines = True    # toggle with L key

        # ---- playback state ----
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
    # Motion loading / navigation
    # ------------------------------------------------------------------

    def load_motion(self, motion_idx: int):
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion = self.dataset[self.current_motion_idx]
        self.current_frame = 0

        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Style:  {self.current_motion.style}")
        print(f"Frames: {self.current_motion.num_frames}  "
              f"({self.current_motion.num_frames / self.fps:.2f} s)")
        print(f"{'='*60}\n")

        self.update_pose()

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

    def reset(self):
        self.current_frame = 0
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

    def update_pose(self):
        """Apply current-frame qpos to MuJoCo and recompute sensor."""
        qpos = self.current_motion.get_qpos(self.current_frame)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # -- environment sensor --
        robot_pos = qpos[:3]            # XYZ
        robot_yaw = quat_wxyz_to_yaw(qpos[3:7])  # heading from root quaternion (wxyz)
        self.readings, self.hit_points = self.sensor.compute(
            robot_pos, robot_yaw, self.obstacles
        )

        # -- trajectory cache --
        if self.show_trajectory:
            self._cache_trajectory()

    def step(self):
        """Advance one frame if enough wall-clock time has passed."""
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
            self.current_frame,
            self.past_frames,
            self.future_frames,
            kernel_idx=0,
        )
        self.past_traj, self.future_traj, self.past_orient, self.future_orient = result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, scene):
        """Draw all overlays into the MuJoCo user scene."""
        scene.ngeom = 0

        if self.show_trajectory and hasattr(self, "past_traj"):
            if self.past_traj is not None and len(self.past_traj) > 0:
                draw_trajectory(scene, self.past_traj, self.past_orient,
                                color=[0.2, 0.5, 1.0, 1.0])
            if self.future_traj is not None and len(self.future_traj) > 0:
                draw_trajectory(scene, self.future_traj, self.future_orient,
                                color=[1.0, 0.2, 0.2, 1.0])

        if self.show_sensor:
            robot_pos = self.data.qpos[:3]
            draw_sensor_readings(
                scene,
                robot_pos,
                self.readings,
                self.hit_points,
                z_height=0.08,
                dot_radius=0.04,
                draw_lines=self.draw_sensor_lines,
            )
            # Draw obstacles (none in Step 1, shown here for future extension)
            for obs in self.obstacles:
                from utils.environment_sensor import CircleObstacle, BoxObstacle
                if isinstance(obs, CircleObstacle):
                    draw_obstacle_circle(scene, obs.center, obs.radius)
                elif isinstance(obs, BoxObstacle):
                    draw_obstacle_box(scene, obs.center, obs.half_extents, obs.yaw)

    # ------------------------------------------------------------------
    # Toggle helpers
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
    if keycode == 32:           # SPACE
        player.toggle_pause()
    elif keycode == 265:        # UP
        player.next_motion()
    elif keycode == 264:        # DOWN
        player.prev_motion()
    elif keycode == 263:        # LEFT
        player.prev_frame()
    elif keycode == 262:        # RIGHT
        player.next_frame()
    elif keycode in (ord('r'), ord('R')):
        player.reset()
    elif keycode in (ord('t'), ord('T')):
        player.toggle_trajectory()
    elif keycode in (ord('e'), ord('E')):
        player.toggle_sensor()
    elif keycode in (ord('l'), ord('L')):
        player.toggle_sensor_lines()
    elif keycode in (ord('s'), ord('S')):
        player.print_status()
    elif keycode in SPEED_MAP:
        player.set_speed(SPEED_MAP[keycode])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Visualise LAFAN motion data with Environment Sensor overlay"
    )
    parser.add_argument("--dataset", default="lafan1_g1",
                        help="Dataset name (default: lafan1_g1)")
    parser.add_argument("--motion", type=int, default=0,
                        help="Starting motion clip index (default: 0)")
    parser.add_argument("--n-rays", type=int, default=36,
                        help="Number of sensor rays (default: 36)")
    parser.add_argument("--max-range", type=float, default=3.0,
                        help="Sensor maximum range in metres (default: 3.0)")
    parser.add_argument("--past-frames", type=int, default=10)
    parser.add_argument("--future-frames", type=int, default=45)
    parser.add_argument("--no-sensor", action="store_true",
                        help="Start with sensor display OFF")
    parser.add_argument("--no-trajectory", action="store_true",
                        help="Start with trajectory display OFF")
    return parser.parse_args()


def print_instructions():
    print("\n" + "=" * 60)
    print("  Environment-Sensor Visualiser  –  Controls")
    print("-" * 60)
    print("  SPACE       : Pause / Resume")
    print("  LEFT/RIGHT  : Previous / Next frame (when paused)")
    print("  UP/DOWN     : Previous / Next motion clip")
    print("  R           : Reset to first frame")
    print("  T           : Toggle trajectory visualisation")
    print("  E           : Toggle environment-sensor overlay")
    print("  L           : Toggle ray lines (dots only ↔ lines+dots)")
    print("  1-9         : Set playback speed (1=0.25×, 5=1×, 9=2×)")
    print("  S           : Print status")
    print("  ESC         : Exit")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    scene_path = os.path.join(
        os.path.dirname(__file__), "assets", "scene.xml"
    )
    dataset_path = f"data/pkls/{args.dataset}.pkl"

    print("=" * 60)
    print("  Step 2 (Env): Visualise Data + Environment Sensor")
    print("=" * 60)

    # -- MuJoCo model --
    print(f"\nLoading MuJoCo scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(model)

    # -- dataset --
    print(f"Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"  Dataset not found: {dataset_path}")
        print("  Available pkl files:")
        pkl_dir = "data/pkls"
        for f in sorted(os.listdir(pkl_dir)):
            if f.endswith(".pkl"):
                print(f"    {f[:-4]}")
        return

    dataset = MotionDataset(dataset_path)
    dataset.print_summary()

    print(f"\nEnvironment Sensor: {args.n_rays} rays, {args.max_range} m range")
    print("  (No obstacles in Step 1 – all readings are 0 / free)")

    # -- player --
    player = SensorMotionPlayer(
        model, mj_data, dataset,
        n_rays=args.n_rays,
        max_range=args.max_range,
        show_trajectory=not args.no_trajectory,
        show_sensor=not args.no_sensor,
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
