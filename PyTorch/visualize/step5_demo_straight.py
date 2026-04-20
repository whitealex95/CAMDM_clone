"""
Step 5: Straight-line cylinder avoidance test
---------------------------------------------
Minimal controlled experiment for sensor-conditioning evaluation:

  • Robot walks (0,0) → (goal_x, 0) along a straight target trajectory.
  • A cylinder obstacle sits at (obstacle_x, 0) blocking the direct path.
  • Press N to toggle sensor conditioning ON/OFF for live comparison.
  • Press --compare to run both headlessly and save a side-by-side video.

Controls
--------
  SPACE  : Pause / Resume
  R      : Reset
  N      : Toggle sensor ON/OFF  (ablation)
  T      : Toggle trajectory
  E      : Toggle sensor overlay
  O      : Toggle obstacle
  L      : Toggle sensor ray lines
  C      : Toggle camera follow
  S      : Print status
  1-9    : Playback speed
  ESC    : Exit

Usage
-----
    python visualize/step5_demo_straight.py \\
        --checkpoint save/<run>/best.pt \\
        [--dataset lafan1_g1_motion30]
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
import imageio.v2 as imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models_env2d import MotionDiffusionEnv
from diffusion.create_diffusion import create_gaussian_diffusion

from visualize.utils.geometry import (
    draw_trajectory,
    draw_trajectory_lines,
    draw_sensor_readings,
    draw_obstacle_circle,
    draw_label,
)
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic
from utils.environment_sensor import (
    EnvironmentSensor,
    CircleObstacle,
    quat_wxyz_to_yaw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)], dtype=np.float32)


def qpos_to_model_format(qpos_seq: np.ndarray) -> np.ndarray:
    """(T, 36) → (T, 31, 6)"""
    T = qpos_seq.shape[0]
    out = np.zeros((T, 31, 6), dtype=np.float32)
    for t in range(T):
        out[t, 0] = nn_transforms.quat2repr6d(
            torch.from_numpy(qpos_seq[t, 3:7]).float().unsqueeze(0)
        ).numpy()[0]
        out[t, 1:30, 0] = qpos_seq[t, 7:]
        out[t, 30, :3]  = qpos_seq[t, :3]
    return out


def model_format_to_qpos(model_out: np.ndarray) -> np.ndarray:
    """(T, 31, 6) → (T, 36)"""
    T = model_out.shape[0]
    qpos = np.zeros((T, 36), dtype=np.float32)
    for t in range(T):
        qpos[t, :3]  = model_out[t, 30, :3]
        qpos[t, 3:7] = nn_transforms.repr6d2quat(
            torch.from_numpy(model_out[t, 0]).float().unsqueeze(0)
        ).numpy()[0]
        qpos[t, 7:]  = model_out[t, 1:30, 0]
    return qpos


def _make_straight_waypoints(
    start_x: float, goal_x: float, y: float = 0.0, spacing: float = 0.05
) -> np.ndarray:
    n = max(2, int(abs(goal_x - start_x) / spacing))
    xs = np.linspace(start_x, goal_x, n)
    ys = np.full(n, y)
    return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------
# Path controller (dense waypoint follower, same as step4)
# ---------------------------------------------------------------------------

class PathController:
    def __init__(self, waypoints: np.ndarray, speed=0.5, fps=30., future_frames=45):
        self.waypoints     = np.asarray(waypoints, dtype=np.float64)
        self.speed         = float(speed)
        self.fps           = float(fps)
        self.future_frames = int(future_frames)
        diffs      = np.diff(self.waypoints, axis=0)
        self._arc  = np.concatenate([[0.], np.cumsum(np.linalg.norm(diffs, axis=1))])
        self.total_length = float(self._arc[-1])
        self._progress    = 0.

    def reset(self): self._progress = 0.

    @property
    def is_done(self): return self._progress >= self.total_length - 0.3

    def update(self, robot_xy: np.ndarray):
        robot_xy = np.asarray(robot_xy[:2], dtype=np.float64)
        lo    = max(0, int(np.searchsorted(self._arc, self._progress)) - 3)
        dists = np.linalg.norm(self.waypoints[lo:] - robot_xy, axis=1)
        best  = self._arc[lo + int(np.argmin(dists))]
        if best > self._progress:
            self._progress = best

    def get_future_trajectory(self):
        step = self.speed / self.fps
        traj_xy, traj_quat = [], []
        for i in range(1, self.future_frames + 1):
            arc = min(self._progress + i * step, self.total_length)
            traj_xy.append(self._interp_xy(arc))
            traj_quat.append(_yaw_to_quat(self._interp_yaw(arc)))
        return np.array(traj_xy, dtype=np.float32), np.array(traj_quat, dtype=np.float32)

    def _interp_xy(self, arc):
        idx  = int(np.clip(np.searchsorted(self._arc, arc) - 1, 0, len(self.waypoints) - 2))
        span = max(self._arc[idx + 1] - self._arc[idx], 1e-9)
        t    = float(np.clip((arc - self._arc[idx]) / span, 0., 1.))
        return self.waypoints[idx] + t * (self.waypoints[idx + 1] - self.waypoints[idx])

    def _interp_yaw(self, arc):
        idx = int(np.clip(np.searchsorted(self._arc, arc) - 1, 0, len(self.waypoints) - 2))
        d   = self.waypoints[idx + 1] - self.waypoints[idx]
        return float(np.arctan2(d[1], d[0]))


# ---------------------------------------------------------------------------
# Model wrapper & sensor-conditioned generator
# ---------------------------------------------------------------------------

class _ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, **kw):
        return self.model.forward(
            x, timesteps,
            kw.get('past_motion'), kw.get('traj_pose'),
            kw.get('traj_trans'),  kw.get('style_idx'), kw.get('sensor'),
        )


class SensorMotionGenerator:
    def __init__(self, model, diffusion, config, sensor: EnvironmentSensor,
                 device='cuda', sampler='ddpm', cfg_scale=1.0):
        self.model         = _ModelWrapper(model)
        self.diffusion     = diffusion
        self.sensor        = sensor
        self.device        = device
        self.sampler       = sampler.lower()
        self.cfg_scale     = float(cfg_scale)
        self.future_frames = config.arch.future_frame
        self.rot_req       = config.arch.rot_req
        self.per_rot_feat  = 6

    def generate_motion(
        self,
        past_qpos: np.ndarray,
        traj_trans: np.ndarray,
        traj_pose: np.ndarray,
        style_idx: int,
        obstacles: list,
        zero_sensor: bool = False,
        cfg_scale: float = None,
    ) -> np.ndarray:
        curr_xy = past_qpos[-1, :2].copy()

        pq = past_qpos.copy()
        pq[:, :2] -= curr_xy
        past_t = torch.from_numpy(qpos_to_model_format(pq)).float() \
            .unsqueeze(0).permute(0, 2, 3, 1).to(self.device)

        traj_tr_t = torch.from_numpy((traj_trans - curr_xy).astype(np.float32)).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)
        traj_repr = nn_transforms.get_rotation(
            torch.from_numpy(traj_pose).float(), self.rot_req
        ).numpy()
        traj_po_t = torch.from_numpy(traj_repr).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)

        if zero_sensor:
            readings = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        else:
            yaw      = quat_wxyz_to_yaw(past_qpos[-1, 3:7])
            readings, _ = self.sensor.compute(past_qpos[-1, :3], yaw, obstacles)
        sensor_t = torch.from_numpy(readings).float().unsqueeze(0).to(self.device)
        style_t  = torch.tensor([style_idx]).to(self.device)

        model_kwargs = dict(
            past_motion=past_t, traj_trans=traj_tr_t, traj_pose=traj_po_t,
            sensor=sensor_t, style_idx=style_t, y={},
        )
        uncond_kwargs = dict(
            past_motion=torch.zeros_like(past_t),
            traj_trans=traj_tr_t, traj_pose=traj_po_t,
            sensor=torch.zeros_like(sensor_t), style_idx=style_t, y={},
        )

        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if scale == 1.0:
            sampling_model  = self.model
            sampling_kwargs = model_kwargs
        else:
            cond_m, uncond_kw, s = self.model, uncond_kwargs, scale

            class _CFGWrap(torch.nn.Module):
                def forward(self_, x, timesteps, **kw):  # noqa: N805
                    return cond_m(x, timesteps, **kw) + \
                           s * (cond_m(x, timesteps, **kw) -
                                cond_m(x, timesteps, **uncond_kw))

            sampling_model  = _CFGWrap()
            sampling_kwargs = model_kwargs

        shape = (1, 31, self.per_rot_feat, self.future_frames)
        with torch.no_grad():
            if self.sampler == 'ddim':
                out = self.diffusion.ddim_sample_loop(
                    sampling_model, shape, clip_denoised=False,
                    model_kwargs=sampling_kwargs, progress=False, eta=0., device=self.device,
                )
            else:
                out = self.diffusion.p_sample_loop(
                    sampling_model, shape, clip_denoised=False,
                    model_kwargs=sampling_kwargs, progress=False, device=self.device,
                )
        out     = out.squeeze(0).permute(2, 0, 1).cpu().numpy()
        qpos_out = model_format_to_qpos(out)
        qpos_out[:, :2] += curr_xy
        return qpos_out


# ---------------------------------------------------------------------------
# Demo player
# ---------------------------------------------------------------------------

class StraightLinePlayer:
    """
    Autoregressive player for the straight-line cylinder avoidance test.
    Trajectory: straight line from (start_x, 0) to (goal_x, 0).
    Obstacle  : single CircleObstacle at (obstacle_x, 0).
    """

    def __init__(
        self,
        mj_model, mj_data,
        generator: SensorMotionGenerator,
        path: PathController,
        obstacle: CircleObstacle,
        init_qpos: np.ndarray,
        style_idx: int = 0,
        past_frames: int = 10,
        future_frames: int = 45,
        traj_bias_pos: float = 0.4,
        traj_bias_rot: float = 2.2,
        cfg_count: int = 2,
        applyframes: int = 15,
        inertialize: bool = True,
        inertialization_mode: str = 'camdm',
        blendtime_rotation: float = 0.2,
        blendtime_position: float = 0.2,
        spring_halflife_position: float = 0.12,
        spring_halflife_rotation: float = 0.12,
        inertial_quat_start: int = 3,
        inertial_quat_end: int = 7,
    ):
        self.mj_model  = mj_model
        self.mj_data   = mj_data
        self.generator = generator
        self.path      = path
        self.obstacle  = obstacle
        self.obstacles = [obstacle]
        self.init_qpos = init_qpos.copy()
        self.style_idx = style_idx

        self.show_trajectory = True
        self.show_sensor     = True
        self.show_obstacle   = True
        self.draw_lines      = False
        self.camera_follow   = True
        self.zero_sensor     = False

        self.fps           = 30
        self.frame_dt      = 1.0 / self.fps
        self.playing       = True
        self.playback_speed = 1.0
        self.last_update   = time.time()

        self.past_frames   = past_frames
        self.future_frames = future_frames
        self.apply_frames  = int(applyframes)
        self.gen_idx       = 0
        self.gen_qpos      = None

        self.cfg_count_cache = int(cfg_count)
        self.cfg_count       = int(cfg_count)

        self.inertialize          = bool(inertialize)
        self.inertialization_mode = str(inertialization_mode).lower()
        self.blendtime_rotation   = float(blendtime_rotation)
        self.blendtime_position   = float(blendtime_position)
        self.spring_halflife_pos  = float(spring_halflife_position)
        self.spring_halflife_rot  = float(spring_halflife_rotation)
        self.quat_slice           = slice(int(inertial_quat_start), int(inertial_quat_end))
        self.transition_mgr       = None

        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)

        self.sensor         = generator.sensor
        self.readings       = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        self.sphere_centers = np.zeros((self.sensor.feature_dim, 2), dtype=np.float64)

        self.future_traj_wp    = None
        self.future_orient_wp  = None
        self.future_traj       = None
        self.future_orient     = None
        self.past_traj         = None
        self.past_orient       = None
        self.pred_traj         = None
        self.pred_orient       = None

        self.qpos_history = deque(maxlen=past_frames)
        self._init_pose()
        self._update_traj()

    # ------------------------------------------------------------------

    def _init_pose(self):
        for _ in range(self.past_frames):
            self.qpos_history.append(self.init_qpos.copy())
        self.mj_data.qpos[:] = self.init_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self):
        self.path.reset()
        self.gen_idx  = 0
        self.gen_qpos = None
        self.pred_traj = self.pred_orient = None
        self.cfg_count = self.cfg_count_cache
        self.transition_mgr = None
        self.qpos_history.clear()
        self._init_pose()
        self._update_traj()
        print("Reset.")

    # ------------------------------------------------------------------

    def _update_traj(self):
        qh = np.array(self.qpos_history)
        self.past_traj   = qh[-self.past_frames:, :3]
        self.past_orient = qh[-self.past_frames:, 3:7]

        curr_xy = self.mj_data.qpos[:2]
        self.path.update(curr_xy)
        wp_t, wp_o = self.path.get_future_trajectory()
        self.future_traj_wp   = wp_t
        self.future_orient_wp = wp_o

        if self.gen_qpos is not None:
            pred_xy  = self.gen_qpos[self.gen_idx:, :2]
            pred_ori = self.gen_qpos[self.gen_idx:, 3:7]
            ext_t, ext_o = extend_future_traj_heusristic(pred_xy, pred_ori, self.future_frames)
            self.future_traj, self.future_orient = blend_trajectory(
                ext_t, ext_o, wp_t, wp_o,
                blend=self.traj_bias_pos, blend_rot=self.traj_bias_rot,
            )
        else:
            self.future_traj   = wp_t
            self.future_orient = wp_o

    def _generate(self) -> np.ndarray:
        eff_scale = self.generator.cfg_scale if self.cfg_count > 0 else 1.0
        gen = self.generator.generate_motion(
            np.array(self.qpos_history),
            self.future_traj, self.future_orient,
            self.style_idx, self.obstacles,
            zero_sensor=self.zero_sensor,
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
        if self.gen_idx == 0:
            self.gen_qpos = self._generate()
        self.pred_traj   = self.gen_qpos[self.gen_idx:, :3]
        self.pred_orient = self.gen_qpos[self.gen_idx:, 3:7]
        qpos = self.gen_qpos[self.gen_idx]
        self.qpos_history.append(qpos.copy())
        self._update_traj()
        self.mj_data.qpos[:] = qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.gen_idx = (self.gen_idx + 1) % self.apply_frames

    def _update_inertialized(self):
        if self.transition_mgr is None:
            self.transition_mgr = create_transition_manager(
                mode=self.inertialization_mode,
                frame_dt=self.frame_dt,
                quat_slice=self.quat_slice,
                blend_time_rotation=self.blendtime_rotation,
                blend_time_position=self.blendtime_position,
                halflife_position=self.spring_halflife_pos,
                halflife_rotation=self.spring_halflife_rot,
            )
        if self.gen_idx == 0:
            self.gen_qpos = self._generate()
            self.transition_mgr.start_transition(
                self.qpos_history, self.mj_data.qpos.copy(), self.gen_qpos
            )
        self.pred_traj   = self.gen_qpos[self.gen_idx:, :3]
        self.pred_orient = self.gen_qpos[self.gen_idx:, 3:7]
        raw   = self.gen_qpos[self.gen_idx]
        final = self.transition_mgr.apply(raw)
        self.qpos_history.append(final.copy())
        self._update_traj()
        self.mj_data.qpos[:] = final
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.gen_idx = (self.gen_idx + 1) % self.apply_frames

    def step(self):
        if not self.playing:
            return
        now = time.time()
        if now - self.last_update < self.frame_dt / self.playback_speed:
            return
        qpos = self.mj_data.qpos
        self.readings, self.sphere_centers = self.sensor.compute(
            qpos[:3], quat_wxyz_to_yaw(qpos[3:7]), self.obstacles
        )
        self.update_pose()
        self.last_update = now
        if self.path.is_done:
            print("Reached goal – resetting.")
            self.reset()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, scene, clear: bool = True):
        if clear:
            scene.ngeom = 0

        if self.show_trajectory and self.past_traj is not None:
            draw_trajectory(scene, self.past_traj, self.past_orient,
                            color=[0.2, 0.5, 1.0, 1.0])           # blue: past
            if self.future_traj_wp is not None:
                wt3 = np.hstack([self.future_traj_wp,
                                  np.zeros((len(self.future_traj_wp), 1))])
                draw_trajectory(scene, wt3, self.future_orient_wp,
                                color=[1.0, 0.2, 0.2, 1.0])       # red: waypoint target
            if self.future_traj is not None:
                ft3 = np.hstack([self.future_traj,
                                  np.zeros((len(self.future_traj), 1))])
                draw_trajectory(scene, ft3, self.future_orient,
                                color=[0.2, 1.0, 0.2, 1.0])       # green: blended (fed to model)

        # Full reference line: (start, 0) → (goal, 0) as thin yellow line
        if self.show_trajectory:
            wps = self.path.waypoints
            wps3 = np.hstack([wps, np.full((len(wps), 1), 0.02)])
            draw_trajectory_lines(scene, wps3, color=[1.0, 0.85, 0.0, 0.6])  # yellow

        if self.show_obstacle:
            draw_obstacle_circle(scene, self.obstacle.center, self.obstacle.radius, height=1.2)

        if self.show_sensor:
            draw_sensor_readings(
                scene, self.mj_data.qpos[:3], self.readings, self.sphere_centers,
                z_height=0.08, dot_radius=0.04, draw_lines=self.draw_lines,
            )

        # Sensor ON/OFF indicator
        if scene.ngeom < scene.maxgeom - 1:
            robot_pos    = self.mj_data.qpos[:3].copy()
            indicator_pos = robot_pos + np.array([0., 0., 1.6])
            if self.zero_sensor:
                ind_color  = np.array([1.0, 0.15, 0.15, 0.95], dtype=np.float32)
                label_text = "SENSOR: OFF"
            else:
                ind_color  = np.array([0.15, 1.0, 0.15, 0.95], dtype=np.float32)
                label_text = "SENSOR: ON"
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.12, 0.12, 0.12], dtype=np.float64),
                pos=indicator_pos,
                mat=np.eye(3).flatten(),
                rgba=ind_color,
            )
            scene.ngeom += 1
            draw_label(scene, indicator_pos + np.array([0., 0., 0.25]), label_text)

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def toggle_trajectory(self):   self.show_trajectory = not self.show_trajectory
    def toggle_sensor_overlay(self): self.show_sensor   = not self.show_sensor
    def toggle_obstacle(self):     self.show_obstacle   = not self.show_obstacle
    def toggle_sensor_lines(self): self.draw_lines      = not self.draw_lines
    def toggle_camera_follow(self): self.camera_follow  = not self.camera_follow

    def toggle_zero_sensor(self):
        self.zero_sensor = not self.zero_sensor
        print(f"Sensor: {'OFF (zero input)' if self.zero_sensor else 'ON'}")

    def set_speed(self, s: float):
        self.playback_speed = s
        print(f"Speed: {s}×")

    def print_status(self):
        n_act = int((self.readings > 0).sum())
        print(
            f"Sensor {'OFF' if self.zero_sensor else 'ON '} | "
            f"Progress {self.path._progress:.1f}/{self.path.total_length:.1f} m | "
            f"sensor active {n_act}/{self.sensor.feature_dim}"
        )


# ---------------------------------------------------------------------------
# Keyboard callback
# ---------------------------------------------------------------------------

_SPEED_MAP = {
    ord('1'): 0.25, ord('2'): 0.50, ord('3'): 0.75, ord('4'): 0.90,
    ord('5'): 1.00, ord('6'): 1.25, ord('7'): 1.50, ord('8'): 1.75,
    ord('9'): 2.00,
}


def key_callback(player: StraightLinePlayer, keycode: int):
    if keycode == 32:
        player.toggle_pause()
    elif keycode in (ord('r'), ord('R')):
        player.reset()
    elif keycode in (ord('n'), ord('N')):
        player.toggle_zero_sensor()
    elif keycode in (ord('t'), ord('T')):
        player.toggle_trajectory()
    elif keycode in (ord('e'), ord('E')):
        player.toggle_sensor_overlay()
    elif keycode in (ord('o'), ord('O')):
        player.toggle_obstacle()
    elif keycode in (ord('l'), ord('L')):
        player.toggle_sensor_lines()
    elif keycode in (ord('c'), ord('C')):
        player.toggle_camera_follow()
    elif keycode in (ord('s'), ord('S')):
        player.print_status()
    elif keycode in _SPEED_MAP:
        player.set_speed(_SPEED_MAP[keycode])


# ---------------------------------------------------------------------------
# Headless comparison
# ---------------------------------------------------------------------------

def _collect_pass(player: StraightLinePlayer, max_steps: int = 6000) -> np.ndarray:
    player.reset()
    records = [player.mj_data.qpos.copy()]
    for _ in range(max_steps):
        qpos = player.mj_data.qpos
        player.readings, player.sphere_centers = player.sensor.compute(
            qpos[:3], quat_wxyz_to_yaw(qpos[3:7]), player.obstacles
        )
        player.update_pose()
        records.append(player.mj_data.qpos.copy())
        if player.path.is_done:
            break
    return np.array(records)


def run_comparison(args, mj_model, generator, obstacle, init_qpos, style_idx, waypoints):
    fps = 30

    def _make_player(mj_data, zero_sensor):
        path = PathController(waypoints, speed=args.speed, fps=fps,
                              future_frames=args.future_frames)
        p = StraightLinePlayer(
            mj_model, mj_data, generator, path, obstacle, init_qpos,
            style_idx=style_idx,
            past_frames=args.past_frames,
            future_frames=args.future_frames,
            traj_bias_pos=args.traj_bias_pos,
            traj_bias_rot=args.traj_bias_rot,
            cfg_count=args.cfg_count,
            applyframes=args.applyframes,
            inertialize=(args.inertialize == 'on'),
            inertialization_mode=args.inertialization_mode,
            blendtime_rotation=args.blendtime_rotation,
            blendtime_position=args.blendtime_position,
            spring_halflife_position=args.spring_halflife_position,
            spring_halflife_rotation=args.spring_halflife_rotation,
        )
        p.zero_sensor = zero_sensor
        return p

    print("\n[Compare] Pass 1 – SENSOR ON …")
    mj_data_on  = mujoco.MjData(mj_model)
    player_on   = _make_player(mj_data_on, zero_sensor=False)
    qpos_on     = _collect_pass(player_on)
    print(f"          {len(qpos_on)} frames")

    print("[Compare] Pass 2 – SENSOR OFF …")
    mj_data_off = mujoco.MjData(mj_model)
    player_off  = _make_player(mj_data_off, zero_sensor=True)
    qpos_off    = _collect_pass(player_off)
    print(f"          {len(qpos_off)} frames")

    os.makedirs("videos", exist_ok=True)
    tag = time.strftime('%m%d_%H%M')
    _save_comparison_figure(qpos_on, qpos_off, obstacle, waypoints,
                            f"videos/straight_compare_{tag}.png")
    _save_comparison_video(qpos_on, qpos_off, mj_model, obstacle, waypoints,
                           generator.sensor, fps, f"videos/straight_compare_{tag}.mp4")


def _save_comparison_figure(qpos_on, qpos_off, obstacle, waypoints, output_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 6))

    # Reference line
    ax.plot(waypoints[:, 0], waypoints[:, 1], '--',
            color='#ccaa00', linewidth=1.5, alpha=0.7, label='Target path', zorder=2)

    # Obstacle circle
    circ = plt.Circle(obstacle.center[:2], obstacle.radius,
                      color='#888888', alpha=0.5, zorder=3)
    ax.add_patch(circ)

    # Trajectories
    ax.plot(qpos_on[:, 0],  qpos_on[:, 1],  color='#00bb44', linewidth=2.5,
            label='Sensor ON', zorder=4)
    ax.plot(qpos_off[:, 0], qpos_off[:, 1], color='#dd2222', linewidth=2.5,
            alpha=0.85, label='Sensor OFF', zorder=4)

    ax.plot(*waypoints[0],  'o', color='royalblue',  markersize=12, zorder=5, label='Start')
    ax.plot(*waypoints[-1], '*', color='darkorange', markersize=16, zorder=5, label='Goal')

    ax.text(0.02, 0.97,
            f"Sensor ON:  {len(qpos_on)} frames\nSensor OFF: {len(qpos_off)} frames",
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Straight-line Cylinder Avoidance – Sensor ON vs OFF',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.25, linestyle=':')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Compare] Figure → {output_path}")


def _save_comparison_video(qpos_on, qpos_off, mj_model, obstacle, waypoints,
                           sensor, fps, output_path, W=640, H=720):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    mid_x = (waypoints[0, 0] + waypoints[-1, 0]) / 2
    mid_y = (waypoints[0, 1] + waypoints[-1, 1]) / 2
    cam.lookat[:]  = [mid_x, mid_y, 0.5]
    cam.distance   = 10.0
    cam.elevation  = -55.0
    cam.azimuth    = 0.0

    mj_data_on  = mujoco.MjData(mj_model)
    mj_data_off = mujoco.MjData(mj_model)
    ren_on  = mujoco.Renderer(mj_model, height=H, width=W)
    ren_off = mujoco.Renderer(mj_model, height=H, width=W)

    def _draw_overlay(scene, past_xy, traj_color, label, qpos, zero_sensor):
        draw_obstacle_circle(scene, obstacle.center, obstacle.radius, height=1.2)
        wps3 = np.hstack([waypoints, np.full((len(waypoints), 1), 0.02)])
        draw_trajectory_lines(scene, wps3, color=[1.0, 0.85, 0.0, 0.6])
        if len(past_xy) >= 2:
            past3 = np.hstack([past_xy, np.full((len(past_xy), 1), 0.05)])
            draw_trajectory_lines(scene, past3.astype(np.float64), color=traj_color)
        # Sensor dots: actual readings when ON, zeros when OFF
        yaw = quat_wxyz_to_yaw(qpos[3:7])
        readings, sphere_centers = sensor.compute(qpos[:3], yaw, [obstacle])
        if zero_sensor:
            readings = np.zeros_like(readings)
        draw_sensor_readings(scene, qpos[:3], readings, sphere_centers,
                             z_height=0.08, dot_radius=0.04, draw_lines=False)
        center_pos = np.array([mid_x, mid_y, 2.5])
        draw_label(scene, center_pos, label)

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    n = max(len(qpos_on), len(qpos_off))
    print(f"[Compare] Rendering {n} frames …")
    for t in range(n):
        t_on  = min(t, len(qpos_on)  - 1)
        t_off = min(t, len(qpos_off) - 1)

        mj_data_on.qpos[:] = qpos_on[t_on]
        mujoco.mj_forward(mj_model, mj_data_on)
        ren_on.update_scene(mj_data_on, camera=cam)
        _draw_overlay(ren_on.scene, qpos_on[:t_on + 1, :2],
                      [0.1, 1.0, 0.1, 0.95], "SENSOR: ON",
                      qpos=qpos_on[t_on], zero_sensor=False)
        frame_on = ren_on.render().copy()

        mj_data_off.qpos[:] = qpos_off[t_off]
        mujoco.mj_forward(mj_model, mj_data_off)
        ren_off.update_scene(mj_data_off, camera=cam)
        _draw_overlay(ren_off.scene, qpos_off[:t_off + 1, :2],
                      [1.0, 0.15, 0.15, 0.95], "SENSOR: OFF",
                      qpos=qpos_off[t_off], zero_sensor=True)
        frame_off = ren_off.render().copy()

        writer.append_data(np.hstack([frame_on, frame_off]))

    writer.close()
    print(f"[Compare] Video → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Straight-line cylinder avoidance test")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to MotionDiffusionEnv checkpoint (.pt)")
    p.add_argument("--dataset",       default=None,
                   help="Optional: pkl for initial pose & style (e.g. lafan1_g1_motion30)")
    p.add_argument("--motion",        type=int,   default=0,
                   help="Motion clip index (used if --dataset provided)")
    p.add_argument("--goal-x",        type=float, default=6.0,
                   help="Goal X position (robot walks from 0 to this)")
    p.add_argument("--obstacle-x",    type=float, default=3.0,
                   help="Cylinder X position")
    p.add_argument("--obstacle-y",    type=float, default=0.0,
                   help="Cylinder Y position")
    p.add_argument("--obstacle-radius", type=float, default=1.0,
                   help="Cylinder radius in metres")
    p.add_argument("--speed",         type=float, default=0.7,
                   help="Path-following speed in m/s")
    p.add_argument("--resolution",    type=int,   default=9)
    p.add_argument("--max-range",     type=float, default=2.0)
    p.add_argument("--past-frames",   type=int,   default=10)
    p.add_argument("--future-frames", type=int,   default=45)
    p.add_argument("--traj-bias-pos", type=float, default=0.4)
    p.add_argument("--traj-bias-rot", type=float, default=2.2)
    p.add_argument("--sampler",       default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--cfg-scale",     type=float, default=1.0)
    p.add_argument("--cfg-count",     type=int,   default=2)
    p.add_argument("--applyframes",   type=int,   default=15)
    p.add_argument("--inertialize",   default="on", choices=["on", "off"])
    p.add_argument("--inertialization-mode", default="camdm", choices=["camdm", "spring"])
    p.add_argument("--blendtime-rotation",   type=float, default=0.2)
    p.add_argument("--blendtime-position",   type=float, default=0.2)
    p.add_argument("--spring-halflife-position", type=float, default=0.12)
    p.add_argument("--spring-halflife-rotation",  type=float, default=0.12)
    p.add_argument("--compare", action="store_true",
                   help="Run sensor-ON/OFF headlessly and save side-by-side video")
    return p.parse_args()


def print_instructions():
    print("\n" + "=" * 55)
    print("  Step 5: Straight-line Cylinder Avoidance  –  Controls")
    print("-" * 55)
    print("  SPACE  : Pause / Resume")
    print("  R      : Reset")
    print("  N      : Toggle sensor ON/OFF  (ablation)")
    print("  T      : Toggle trajectory")
    print("  E      : Toggle sensor overlay")
    print("  O      : Toggle obstacle")
    print("  L      : Toggle sensor ray lines")
    print("  C      : Toggle camera follow")
    print("  S      : Print status")
    print("  1-9    : Playback speed")
    print("  ESC    : Exit")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    print("=" * 55)
    print("  Step 5: Straight-line Cylinder Avoidance")
    print("=" * 55)

    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")
    mj_model   = mujoco.MjModel.from_xml_path(scene_path)
    mj_data    = mujoco.MjData(mj_model)

    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config     = checkpoint["config"]

    diffusion_model = MotionDiffusionEnv(
        31 * 6, 1, 31, 6,                    # style_num=1 (overridden below if dataset)
        config.arch.rot_req, config.arch.clip_len,
        env_sensor_dim=config.arch.env_sensor_dim,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=device,
    ).to(device)

    # ── Dataset (optional) ───────────────────────────────────────────────
    style_idx = 0
    if args.dataset is not None:
        from visualize.motion_loader import MotionDataset
        dataset_path = f"data/pkls/{args.dataset}.pkl"
        dataset      = MotionDataset(dataset_path)
        motion_idx   = args.motion % len(dataset)
        style_idx    = dataset[motion_idx].style_idx

        # Reload model with correct number of styles
        diffusion_model = MotionDiffusionEnv(
            31 * 6, len(dataset.styles), 31, 6,
            config.arch.rot_req, config.arch.clip_len,
            env_sensor_dim=config.arch.env_sensor_dim,
            latent_dim=config.arch.latent_dim,
            ff_size=config.arch.ff_size,
            num_layers=config.arch.num_layers,
            num_heads=config.arch.num_heads,
            arch=config.arch.decoder,
            cond_mask_prob=config.trainer.cond_mask_prob,
            device=device,
        ).to(device)

        raw_qpos   = dataset[motion_idx].get_qpos(0)
        init_qpos  = raw_qpos.copy()
        init_qpos[0] = 0.0       # X = 0
        init_qpos[1] = 0.0       # Y = 0
        # Keep Z and joint angles from dataset; override yaw to face +X
        init_qpos[3:7] = _yaw_to_quat(0.0)
        print(f"Dataset: {args.dataset}  clip={motion_idx}  style={style_idx}")
    else:
        # Minimal standing pose (pelvis at ~0.78 m, identity rotation)
        init_qpos      = np.zeros(36, dtype=np.float32)
        init_qpos[2]   = 0.78
        init_qpos[3]   = 1.0    # w of quaternion

    diffusion_model.load_state_dict(checkpoint["state_dict"])
    diffusion_model.eval()

    diffusion = create_gaussian_diffusion(config)
    sensor    = EnvironmentSensor(max_range=args.max_range, resolution=args.resolution)
    generator = SensorMotionGenerator(
        diffusion_model, diffusion, config, sensor,
        device=device, sampler=args.sampler, cfg_scale=args.cfg_scale,
    )

    # ── Scene setup ──────────────────────────────────────────────────────
    obstacle  = CircleObstacle(
        center=np.array([args.obstacle_x, args.obstacle_y], dtype=np.float64),
        radius=float(args.obstacle_radius),
    )
    waypoints = _make_straight_waypoints(
        start_x=init_qpos[0], goal_x=args.goal_x, y=init_qpos[1],
    )
    print(f"Obstacle: ({args.obstacle_x}, {args.obstacle_y})  r={args.obstacle_radius} m")
    print(f"Path:     ({init_qpos[0]:.1f}, {init_qpos[1]:.1f}) → ({args.goal_x:.1f}, {init_qpos[1]:.1f})")

    # ── Comparison mode ──────────────────────────────────────────────────
    if args.compare:
        run_comparison(args, mj_model, generator, obstacle, init_qpos, style_idx, waypoints)
        return

    # ── Interactive mode ─────────────────────────────────────────────────
    path   = PathController(waypoints, speed=args.speed, fps=30.,
                            future_frames=args.future_frames)
    player = StraightLinePlayer(
        mj_model, mj_data, generator, path, obstacle, init_qpos,
        style_idx=style_idx,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        traj_bias_pos=args.traj_bias_pos,
        traj_bias_rot=args.traj_bias_rot,
        cfg_count=args.cfg_count,
        applyframes=args.applyframes,
        inertialize=(args.inertialize == 'on'),
        inertialization_mode=args.inertialization_mode,
        blendtime_rotation=args.blendtime_rotation,
        blendtime_position=args.blendtime_position,
        spring_halflife_position=args.spring_halflife_position,
        spring_halflife_rotation=args.spring_halflife_rotation,
    )

    os.makedirs("videos", exist_ok=True)
    video_path = f"videos/straight_{time.strftime('%m%d_%H%M')}.mp4"
    W, H = 1280, 720
    writer   = imageio.get_writer(video_path, fps=player.fps, codec="libx264",
                                  pixelformat="yuv420p")
    renderer = mujoco.Renderer(mj_model, height=H, width=W)
    frame_last_time = -np.inf
    print(f"\nRecording → {video_path}")

    print_instructions()

    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=lambda kc: key_callback(player, kc),
    ) as viewer:
        viewer.cam.distance  = 8.0
        viewer.cam.elevation = -40.0
        viewer.sync()
        try:
            while viewer.is_running():
                player.step()
                viewer.user_scn.ngeom = 0
                player.render(viewer.user_scn)
                if player.camera_follow:
                    viewer.cam.lookat[:] = mj_data.qpos[:3]
                viewer.sync()

                if time.time() - frame_last_time > 1.0 / player.fps:
                    renderer.update_scene(mj_data, camera=viewer.cam)
                    player.render(renderer.scene, clear=False)
                    writer.append_data(renderer.render())
                    frame_last_time = time.time()

                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()
            print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()
