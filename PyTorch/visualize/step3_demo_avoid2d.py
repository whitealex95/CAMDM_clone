"""
Step 3: Avoid2D Demo – Environment-Sensor-Conditioned Motion Generation
------------------------------------------------------------------------
Like step3_demo.py but uses a model trained with 2-D scan-dot sensor
conditioning (MotionDiffusionEnv / train_g1_env2d.py).

Obstacles are auto-generated around the robot's near-future trajectory and
shown as semi-transparent boxes/cylinders.  Per-frame sensor readings are
computed by casting rays toward the current obstacles and then fed to the
model as an additional local condition.

If the model was trained with sensor conditioning and the obstacles lie close
to the planned path, it may produce avoidance micro-adjustments compared to a
sensor-free baseline.

Controls
--------
  SPACE      : Pause / Resume
  UP/DOWN    : Next / Previous motion clip
  R          : Reset
  T          : Toggle trajectory visualisation
  E          : Toggle sensor overlay
  O          : Toggle obstacle display
  N          : Regenerate obstacles now
  C          : Toggle camera follow
  S          : Print status
  1-9        : Playback speed
  ESC        : Exit

Usage
-----
    python visualize/step3_demo_avoid2d.py \\
        --checkpoint save/<run>/best.pt \\
        --dataset lafan1_g1
"""

import os
import sys
import argparse
import time
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models_env2d import MotionDiffusionEnv
from diffusion.create_diffusion import create_gaussian_diffusion

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import (
    draw_trajectory,
    draw_sensor_readings,
    draw_obstacle_box,
    draw_obstacle_circle,
)
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import (
    blend_trajectory,
    extend_future_traj_heusristic,
    align_trajectory_to_pose,
    match_future_horizon,
)
from utils.environment_sensor import (
    EnvironmentSensor,
    ObstacleGenerator,
    CircleObstacle,
    BoxObstacle,
    quat_wxyz_to_yaw,
)


# ---------------------------------------------------------------------------
# qpos ↔ model-format converters (identical to step3_demo.py)
# ---------------------------------------------------------------------------

def qpos_to_model_format(qpos_seq: np.ndarray) -> np.ndarray:
    """(T, 36) → (T, 31, 6)"""
    T = qpos_seq.shape[0]
    out = np.zeros((T, 31, 6), dtype=np.float32)
    for t in range(T):
        root_quat = qpos_seq[t, 3:7]
        out[t, 0]    = nn_transforms.quat2repr6d(
            torch.from_numpy(root_quat).float().unsqueeze(0)
        ).numpy()[0]
        out[t, 1:30, 0] = qpos_seq[t, 7:]
        out[t, 30, :3]  = qpos_seq[t, :3]
    return out


def model_format_to_qpos(model_out: np.ndarray) -> np.ndarray:
    """(T, 31, 6) → (T, 36)"""
    T = model_out.shape[0]
    qpos = np.zeros((T, 36), dtype=np.float32)
    for t in range(T):
        qpos[t, :3] = model_out[t, 30, :3]
        qpos[t, 3:7] = nn_transforms.repr6d2quat(
            torch.from_numpy(model_out[t, 0]).float().unsqueeze(0)
        ).numpy()[0]
        qpos[t, 7:] = model_out[t, 1:30, 0]
    return qpos


# ---------------------------------------------------------------------------
# Sensor-conditioned motion generator
# ---------------------------------------------------------------------------

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, **kwargs):
        return self.model.forward(
            x, timesteps,
            kwargs.get('past_motion'),
            kwargs.get('traj_pose'),
            kwargs.get('traj_trans'),
            kwargs.get('style_idx'),
            kwargs.get('sensor'),
        )



class SensorMotionGenerator:
    """
    Autoregressive motion generator with sensor conditioning.

    Extends MotionGenerator from step3_demo.py to pass sensor readings.
    """

    def __init__(self, model, diffusion, config, sensor: EnvironmentSensor,
                 device="cuda", sampler="ddpm", cfg_scale=1.0):
        self.model = ModelWrapper(model)
        self.diffusion = diffusion
        self.sensor = sensor
        self.device = device
        self.sampler = sampler.lower()
        self.cfg_scale = float(cfg_scale)

        self.future_frames = config.arch.future_frame
        self.rot_req = config.arch.rot_req
        self.per_rot_feat = 6

    def generate_motion(
        self,
        past_qpos: np.ndarray,
        traj_trans: np.ndarray,
        traj_pose: np.ndarray,
        style_idx: int,
        obstacles,
        cfg_scale=None,
    ) -> np.ndarray:
        """
        Args:
            past_qpos:  (past_frames, 36)
            traj_trans: (future_frames, 2) global XY
            traj_pose:  (future_frames, 4) global wxyz
            obstacles:  list of Obstacle2D
        Returns:
            (future_frames, 36)
        """
        curr_root_XY = past_qpos[-1, :2].copy()

        # ── Past motion ──────────────────────────────────────────────────
        pq = past_qpos.copy()
        pq[:, :2] -= curr_root_XY
        past_motion_t = torch.from_numpy(qpos_to_model_format(pq)).float() \
            .unsqueeze(0).permute(0, 2, 3, 1).to(self.device)  # (1, 31, 6, past)

        # ── Trajectory ───────────────────────────────────────────────────
        traj_t = traj_trans - curr_root_XY
        traj_trans_t = torch.from_numpy(traj_t).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)     # (1, 2, future)

        traj_pose_repr = nn_transforms.get_rotation(
            torch.from_numpy(traj_pose).float(), self.rot_req
        ).numpy()
        traj_pose_t = torch.from_numpy(traj_pose_repr).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)     # (1, 6, future)

        # ── Current-frame sensor reading ─────────────────────────────────
        curr_yaw = quat_wxyz_to_yaw(past_qpos[-1, 3:7])
        sensor_readings, _ = self.sensor.compute(
            past_qpos[-1, :3], curr_yaw, obstacles
        )  # (env_sensor_dim,)
        sensor_t = torch.from_numpy(sensor_readings).float() \
            .unsqueeze(0).to(self.device)                       # (1, env_sensor_dim)

        style_idx_t = torch.tensor([style_idx]).to(self.device)

        # ── Build kwargs ──────────────────────────────────────────────────
        model_kwargs = dict(
            past_motion=past_motion_t,
            traj_trans=traj_trans_t,
            traj_pose=traj_pose_t,
            sensor=sensor_t,
            style_idx=style_idx_t,
            y={},
        )
        uncond_kwargs = dict(
            past_motion=torch.zeros_like(past_motion_t),
            traj_trans=traj_trans_t,
            traj_pose=traj_pose_t,
            sensor=torch.zeros_like(sensor_t),
            style_idx=style_idx_t,
            y={},
        )

        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)

        if scale == 1.0:
            sampling_model = self.model
            sampling_kwargs = model_kwargs
        else:
            cond_m, uncond_kw, s = self.model, uncond_kwargs, scale

            class _CFGWrap(torch.nn.Module):
                def forward(self_, x, timesteps, **kw):  # noqa: N805
                    return cond_m(x, timesteps, **kw) + \
                           s * (cond_m(x, timesteps, **kw) -
                                cond_m(x, timesteps, **uncond_kw))

            sampling_model = _CFGWrap()
            sampling_kwargs = model_kwargs

        shape = (1, 31, self.per_rot_feat, self.future_frames)
        with torch.no_grad():
            if self.sampler == "ddim":
                sample = self.diffusion.ddim_sample_loop(
                    sampling_model, shape, clip_denoised=False,
                    model_kwargs=sampling_kwargs, progress=False, eta=0.0,
                    device=self.device,
                )
            else:
                sample = self.diffusion.p_sample_loop(
                    sampling_model, shape, clip_denoised=False,
                    model_kwargs=sampling_kwargs, progress=False,
                    device=self.device,
                )

        sample = sample.squeeze(0).permute(2, 0, 1).cpu().numpy()
        qpos_out = model_format_to_qpos(sample)
        qpos_out[:, :2] += curr_root_XY
        return qpos_out


# ---------------------------------------------------------------------------
# Demo player with sensor + obstacle overlay
# ---------------------------------------------------------------------------

class DemoPlayerEnv:
    """Step-3 demo player extended with live sensor + obstacle rendering."""

    def __init__(
        self,
        mj_model, mj_data, dataset, generator: SensorMotionGenerator,
        show_trajectory=True,
        past_frames=10, future_frames=45,
        traj_bias_pos=0.4, traj_bias_rot=2.2,
        cfg_count=2, applyframes=15,
        inertialize=True, inertialization_mode="camdm",
        blendtime_rotation=0.2, blendtime_position=0.2,
        spring_halflife_position=0.12, spring_halflife_rotation=0.12,
        inertial_quat_start=3, inertial_quat_end=7,
        obstacle_interval=30, obstacle_mode='sparse',
    ):
        self.model   = mj_model
        self.data    = mj_data
        self.dataset = dataset
        self.motion_generator = generator

        self.sensor    = generator.sensor
        self.generator_obs = ObstacleGenerator(mode=obstacle_mode)
        self.obstacles = []
        self._last_obs_window = -1

        self.show_trajectory = show_trajectory
        self.show_sensor     = True
        self.show_obstacles  = True
        self.draw_lines      = True
        self.camera_follow   = True

        self.past_frames   = past_frames
        self.future_frames = future_frames
        self.fps      = 30
        self.frame_dt = 1.0 / self.fps

        self.apply_generated_frames = int(applyframes)
        self.generated_frame_idx    = 0
        self.generated_qpos         = None
        self.generated_future_traj  = None
        self.generated_future_orient = None

        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)
        self.cfg_count_cache = int(cfg_count)
        self.cfg_count       = int(cfg_count)
        self.prev_style_idx  = None

        self.inertialize           = bool(inertialize)
        self.inertialization_mode  = str(inertialization_mode).lower()
        self.blendtime_rotation    = float(blendtime_rotation)
        self.blendtime_position    = float(blendtime_position)
        self.spring_halflife_position = float(spring_halflife_position)
        self.spring_halflife_rotation = float(spring_halflife_rotation)
        self.quat_slice = slice(int(inertial_quat_start), int(inertial_quat_end))
        self.transition_manager = None

        self.obstacle_interval = int(obstacle_interval)
        self.obstacle_mode     = obstacle_mode

        self.playing           = True
        self.playback_speed    = 1.0
        self.last_update_time  = time.time()
        self.current_frame     = 0
        self.current_motion_idx = 0

        self.qpos_history = deque(maxlen=past_frames)

        # sensor state (current frame)
        self.readings       = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        self.sphere_centers = np.zeros((self.sensor.feature_dim, 2), dtype=np.float64)

        self.load_motion(0)

    # ------------------------------------------------------------------
    # Motion management
    # ------------------------------------------------------------------

    def load_motion(self, motion_idx: int):
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion_data = self.dataset[self.current_motion_idx]
        self.current_frame  = 0
        self._last_obs_window = -1
        if self.prev_style_idx is None or \
                self.current_motion_data.style_idx != self.prev_style_idx:
            self.cfg_count = self.cfg_count_cache
        self.prev_style_idx = self.current_motion_data.style_idx

        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx+1}/{len(self.dataset)}  "
              f"style={self.current_motion_data.style}  "
              f"frames={self.current_motion_data.num_frames}")
        print(f"{'='*60}\n")
        self._init_pose()
        self._update_past_trajectory()
        self._update_future_trajectory()

    def next_motion(self): self.load_motion(self.current_motion_idx + 1)
    def prev_motion(self): self.load_motion(self.current_motion_idx - 1)

    def reset(self):
        self.current_frame = 0
        self.load_motion(self.current_motion_idx)

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def set_speed(self, s):
        self.playback_speed = s
        print(f"Speed: {s}×")

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------

    def _window_idx(self) -> int:
        return self.current_frame // self.obstacle_interval

    def _maybe_regenerate(self):
        w = self._window_idx()
        if w == self._last_obs_window:
            return
        w_start = w * self.obstacle_interval
        w_end   = min(w_start + self.obstacle_interval, self.current_motion_data.num_frames)
        la_end  = min(w_end + self.obstacle_interval, self.current_motion_data.num_frames)
        all_q   = self.current_motion_data.get_all_qpos()
        win_xy  = all_q[w_start:w_end, :2]
        la_xy   = all_q[w_end:la_end, :2] if la_end > w_end else None

        self.generator_obs.seed(self.current_motion_idx * 1000 + w)
        self.obstacles = self.generator_obs.generate_for_window(win_xy, la_xy)
        self._last_obs_window = w

    def force_new_obstacles(self):
        self.generator_obs.seed(int(time.time() * 1000) % 1_000_000)
        w_start = self._window_idx() * self.obstacle_interval
        w_end   = min(w_start + self.obstacle_interval, self.current_motion_data.num_frames)
        la_end  = min(w_end + self.obstacle_interval, self.current_motion_data.num_frames)
        all_q   = self.current_motion_data.get_all_qpos()
        win_xy  = all_q[w_start:w_end, :2]
        la_xy   = all_q[w_end:la_end, :2] if la_end > w_end else None
        self.obstacles = self.generator_obs.generate_for_window(win_xy, la_xy)
        print(f"Regenerated {len(self.obstacles)} obstacles  (mode={self.obstacle_mode})")

    # ------------------------------------------------------------------
    # Pose update (mirrors step3_demo.py)
    # ------------------------------------------------------------------

    def _init_pose(self):
        qpos = self.current_motion_data.get_qpos(self.current_frame)
        for _ in range(self.past_frames):
            self.qpos_history.append(qpos.copy())
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def _generate(self) -> np.ndarray:
        past_qpos = (
            np.array(self.qpos_history)
            if self.current_frame > 0
            else self.current_motion_data.get_past_qpos(self.current_frame)
        )
        eff_scale = self.motion_generator.cfg_scale if self.cfg_count > 0 else 1.0
        gen = self.motion_generator.generate_motion(
            past_qpos,
            self.future_traj,
            self.future_orient,
            self.current_motion_data.style_idx,
            self.obstacles,
            cfg_scale=eff_scale,
        )
        if self.cfg_count > 0:
            self.cfg_count -= 1
        return gen

    def update_pose(self):
        if self.inertialize:
            self._update_inertialized()
        else:
            self._update_raw()

    def _update_raw(self):
        if self.generated_frame_idx == 0:
            self.generated_qpos = self._generate()
        qpos = self.generated_qpos[self.generated_frame_idx]
        self.qpos_history.append(qpos.copy())
        self._update_past_trajectory()
        self._update_future_trajectory()
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.generated_frame_idx = (self.generated_frame_idx + 1) % self.apply_generated_frames

    def _update_inertialized(self):
        if self.transition_manager is None:
            self.transition_manager = create_transition_manager(
                mode=self.inertialization_mode,
                frame_dt=self.frame_dt,
                quat_slice=self.quat_slice,
                blend_time_rotation=self.blendtime_rotation,
                blend_time_position=self.blendtime_position,
                halflife_position=self.spring_halflife_position,
                halflife_rotation=self.spring_halflife_rotation,
            )
        if self.generated_frame_idx == 0:
            self.generated_qpos = self._generate()
            self.transition_manager.start_transition(
                self.qpos_history, self.data.qpos.copy(), self.generated_qpos
            )
        raw = self.generated_qpos[self.generated_frame_idx]
        final = self.transition_manager.apply(raw)
        self.qpos_history.append(final.copy())
        self._update_past_trajectory()
        self._update_future_trajectory()
        self.data.qpos[:] = final
        mujoco.mj_forward(self.model, self.data)
        self.generated_frame_idx = (self.generated_frame_idx + 1) % self.apply_generated_frames

    def step(self):
        if not self.playing:
            return
        now = time.time()
        if now - self.last_update_time >= self.frame_dt / self.playback_speed:
            self.current_frame += 1
            if self.current_frame >= self.current_motion_data.num_frames:
                self.current_frame = 0
            # Obstacle regeneration before generating motion
            self._maybe_regenerate()
            # Update sensor at current position
            qpos = self.data.qpos
            self.readings, self.sphere_centers = self.sensor.compute(
                qpos[:3], quat_wxyz_to_yaw(qpos[3:7]), self.obstacles
            )
            self.update_pose()
            self.last_update_time = now

    # ------------------------------------------------------------------
    # Trajectory helpers (mirrors step3_demo.py)
    # ------------------------------------------------------------------

    def _update_past_trajectory(self):
        qh = np.array(self.qpos_history)
        self.past_traj   = qh[-self.past_frames:, :3]
        self.past_orient = qh[-self.past_frames:, 3:7]

    def _load_traj_from_dataset(self):
        pt, ft, po, fo = self.current_motion_data.get_trajectory(
            self.current_frame, self.past_frames, self.future_frames, kernel_idx=0
        )
        return pt[:, :2], ft[:, :2], po, fo

    def _update_future_trajectory(self):
        _, ft_d, _, fo_d = self._load_traj_from_dataset()
        ref_q  = self.current_motion_data.get_qpos(self.current_frame)
        curr_q = self.data.qpos.copy()
        aligned_t, aligned_o = align_trajectory_to_pose(ft_d, fo_d, ref_q, curr_q)
        self.future_traj_dataset, self.future_orient_dataset = match_future_horizon(
            aligned_t, aligned_o, self.future_frames
        )

        if self.generated_qpos is not None:
            gft  = self.generated_qpos[:, :3]
            gfo  = self.generated_qpos[:, 3:7]
            t_c  = self.generated_frame_idx
            pred_xy  = gft[t_c + 1:, :2]
            pred_ori = gfo[t_c + 1:]
            ext_t, ext_o = extend_future_traj_heusristic(pred_xy, pred_ori, self.future_frames)
            self.future_traj, self.future_orient = blend_trajectory(
                ext_t, ext_o,
                self.future_traj_dataset, self.future_orient_dataset,
                blend=self.traj_bias_pos, blend_rot=self.traj_bias_rot,
            )
        else:
            self.future_traj   = self.future_traj_dataset
            self.future_orient = self.future_orient_dataset

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, scene):
        scene.ngeom = 0

        if self.show_trajectory and hasattr(self, 'past_traj'):
            draw_trajectory(scene, self.past_traj, self.past_orient,
                            color=[0.2, 0.5, 1.0, 1.0])
            if hasattr(self, 'future_traj_dataset'):
                draw_trajectory(scene, self.future_traj_dataset,
                                self.future_orient_dataset, color=[1.0, 0.2, 0.2, 1.0])
            draw_trajectory(scene, self.future_traj, self.future_orient,
                            color=[0.2, 1.0, 0.2, 1.0])
            if self.generated_future_traj is not None:
                draw_trajectory(scene, self.generated_future_traj,
                                self.generated_future_orient, color=[0.2, 0.2, 0.2, 0.5])

        if self.show_obstacles:
            for obs in self.obstacles:
                if isinstance(obs, CircleObstacle):
                    draw_obstacle_circle(scene, obs.center, obs.radius, height=0.2)
                elif isinstance(obs, BoxObstacle):
                    draw_obstacle_box(scene, obs.center, obs.half_extents, obs.yaw, height=0.2)

        if self.show_sensor:
            draw_sensor_readings(
                scene, self.data.qpos[:3], self.readings, self.sphere_centers,
                z_height=0.08, dot_radius=0.04, draw_lines=self.draw_lines,
            )

    # ------------------------------------------------------------------
    # Toggles / status
    # ------------------------------------------------------------------

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

    def toggle_sensor(self):
        self.show_sensor = not self.show_sensor
        print(f"Sensor: {'ON' if self.show_sensor else 'OFF'}")

    def toggle_obstacles(self):
        self.show_obstacles = not self.show_obstacles
        print(f"Obstacles: {'ON' if self.show_obstacles else 'OFF'}")

    def toggle_sensor_lines(self):
        self.draw_lines = not self.draw_lines

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow
        print(f"Camera follow: {'ON' if self.camera_follow else 'OFF'}")

    def print_status(self):
        n_active = int((self.readings > 0).sum())
        print(
            f"Motion {self.current_motion_idx+1}/{len(self.dataset)} | "
            f"Frame {self.current_frame} | "
            f"style={self.current_motion_data.style} | "
            f"{'Playing' if self.playing else 'Paused'} ({self.playback_speed}×) | "
            f"sensor active {n_active}/{self.sensor.feature_dim} | "
            f"obstacles {len(self.obstacles)} ({self.obstacle_mode})"
        )


# ---------------------------------------------------------------------------
# Keyboard callback
# ---------------------------------------------------------------------------

SPEED_MAP = {
    ord('1'): 0.25, ord('2'): 0.50, ord('3'): 0.75, ord('4'): 0.90,
    ord('5'): 1.00, ord('6'): 1.25, ord('7'): 1.50, ord('8'): 1.75,
    ord('9'): 2.00,
}


def key_callback(player: DemoPlayerEnv, keycode: int):
    if keycode == 32:
        player.toggle_pause()
    elif keycode == 265:
        player.next_motion()
    elif keycode == 264:
        player.prev_motion()
    elif keycode in (ord('r'), ord('R')):
        player.reset()
    elif keycode in (ord('t'), ord('T')):
        player.toggle_trajectory()
    elif keycode in (ord('e'), ord('E')):
        player.toggle_sensor()
    elif keycode in (ord('o'), ord('O')):
        player.toggle_obstacles()
    elif keycode in (ord('l'), ord('L')):
        player.toggle_sensor_lines()
    elif keycode in (ord('n'), ord('N')):
        player.force_new_obstacles()
    elif keycode in (ord('c'), ord('C')):
        player.toggle_camera_follow()
    elif keycode in (ord('s'), ord('S')):
        player.print_status()
    elif keycode in SPEED_MAP:
        player.set_speed(SPEED_MAP[keycode])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="2-D obstacle-avoidance demo (sensor-conditioned)")
    p.add_argument("--dataset",    default="lafan1_g1")
    p.add_argument("--checkpoint", default="save/camdm_g1_env_lafan1_g1_env/best.pt",
                   help="Path to MotionDiffusionEnv checkpoint")
    p.add_argument("--mode",  default="sparse", choices=["sparse", "dense", "packed"],
                   help="Obstacle density mode (default: sparse)")
    p.add_argument("--obstacle-interval", type=int, default=30)
    p.add_argument("--resolution", type=int,   default=9)
    p.add_argument("--max-range",  type=float, default=2.0)
    p.add_argument("--traj-bias-pos",  type=float, default=0.4)
    p.add_argument("--traj-bias-rot",  type=float, default=2.2)
    p.add_argument("--past-frames",    type=int,   default=10)
    p.add_argument("--future-frames",  type=int,   default=45)
    p.add_argument("--sampler",        default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--cfg-scale",      type=float, default=0.5)
    p.add_argument("--cfg-count",      type=int,   default=2)
    p.add_argument("--applyframes",    type=int,   default=15)
    p.add_argument("--inertialize",    default="on",    choices=["on", "off"])
    p.add_argument("--inertialization-mode", default="camdm", choices=["camdm", "spring"])
    p.add_argument("--blendtime-rotation",   type=float, default=0.2)
    p.add_argument("--blendtime-position",   type=float, default=0.2)
    p.add_argument("--spring-halflife-position", type=float, default=0.12)
    p.add_argument("--spring-halflife-rotation",  type=float, default=0.12)
    p.add_argument("--inertial-quat-start", type=int, default=3)
    p.add_argument("--inertial-quat-end",   type=int, default=7)
    p.add_argument("--motion", type=int, default=0)
    return p.parse_args()


def print_instructions():
    print("\n" + "=" * 60)
    print("  Avoid2D Demo  –  Controls")
    print("-" * 60)
    print("  SPACE      : Pause / Resume")
    print("  UP / DOWN  : Next / Prev motion clip")
    print("  R          : Reset")
    print("  T          : Toggle trajectory")
    print("  E          : Toggle sensor overlay")
    print("  O          : Toggle obstacles")
    print("  L          : Toggle sensor ray lines")
    print("  N          : Regenerate obstacles")
    print("  C          : Toggle camera follow")
    print("  S          : Print status")
    print("  1-9        : Playback speed")
    print("  ESC        : Exit")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    scene_path   = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")
    dataset_path = f"data/pkls/{args.dataset}.pkl"

    print("=" * 60)
    print("  Step 3: Avoid2D – Sensor-Conditioned Motion Demo")
    print("=" * 60)

    print(f"\nLoading MuJoCo scene: {scene_path}")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data  = mujoco.MjData(mj_model)

    print(f"Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    dataset = MotionDataset(dataset_path)
    dataset.print_summary()

    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first with train_g1_env2d.py")
        return
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    env_sensor_dim = config.arch.env_sensor_dim

    diffusion       = create_gaussian_diffusion(config)
    input_feats     = 31 * 6
    style_set       = dataset.styles

    diffusion_model = MotionDiffusionEnv(
        input_feats, len(style_set), 31, 6,
        config.arch.rot_req, config.arch.clip_len,
        env_sensor_dim=env_sensor_dim,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=device,
    ).to(device)
    diffusion_model.load_state_dict(checkpoint["state_dict"])
    diffusion_model.eval()

    sensor = EnvironmentSensor(max_range=args.max_range, resolution=args.resolution)

    generator = SensorMotionGenerator(
        diffusion_model, diffusion, config, sensor,
        device=device, sampler=args.sampler, cfg_scale=args.cfg_scale,
    )

    player = DemoPlayerEnv(
        mj_model, mj_data, dataset, generator,
        show_trajectory=True,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        traj_bias_pos=args.traj_bias_pos,
        traj_bias_rot=args.traj_bias_rot,
        cfg_count=args.cfg_count,
        applyframes=args.applyframes,
        inertialize=(args.inertialize == "on"),
        inertialization_mode=args.inertialization_mode,
        blendtime_rotation=args.blendtime_rotation,
        blendtime_position=args.blendtime_position,
        spring_halflife_position=args.spring_halflife_position,
        spring_halflife_rotation=args.spring_halflife_rotation,
        inertial_quat_start=args.inertial_quat_start,
        inertial_quat_end=args.inertial_quat_end,
        obstacle_interval=args.obstacle_interval,
        obstacle_mode=args.mode,
    )

    if args.motion > 0:
        player.load_motion(args.motion)

    print_instructions()

    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=lambda kc: key_callback(player, kc),
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            player.step()
            viewer.user_scn.ngeom = 0
            player.render(viewer.user_scn)
            if player.camera_follow:
                viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
