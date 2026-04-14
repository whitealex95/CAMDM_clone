"""
Step 4: Maze Demo – Controlled Sensor-Conditioned Avoidance Test
----------------------------------------------------------------
Creates a static L-shaped corridor (walls = BoxObstacle objects) and drives
the robot through it using a waypoint trajectory controller.

This provides a controlled test of whether the environment-sensor conditioning
helps the model generate collision-free motion near walls.

Press N to toggle sensor on/off live for direct comparison:
  • Sensor ON  → actual obstacle occupancy fed to the model
  • Sensor OFF → zero sensor input (as if no obstacles exist)

Controls
--------
  SPACE      : Pause / Resume
  R          : Reset robot to start of maze
  N          : Toggle sensor conditioning (ON ↔ OFF)
  T          : Toggle trajectory visualisation
  E          : Toggle sensor overlay
  O          : Toggle obstacle / wall display
  L          : Toggle sensor ray lines
  C          : Toggle camera follow
  S          : Print status
  1-9        : Playback speed
  ESC        : Exit

Usage
-----
    python visualize/step4_demo_maze.py \\
        --checkpoint save/<run>/best.pt \\
        --dataset lafan1_g1_motion28
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

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import (
    draw_trajectory,
    draw_sensor_readings,
    draw_obstacle_box,
)
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic
from utils.environment_sensor import (
    EnvironmentSensor,
    BoxObstacle,
    quat_wxyz_to_yaw,
)


# ---------------------------------------------------------------------------
# qpos ↔ model-format converters  (identical to step3_demo_avoid2d.py)
# ---------------------------------------------------------------------------

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


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Yaw angle → wxyz quaternion (rotation around the Z-up axis)."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Maze layout
# ---------------------------------------------------------------------------

class MazeLayout:
    """
    L-shaped corridor maze centred on the robot's initial position.

    Local frame:
        x = forward  (aligned with ``initial_yaw`` in world)
        y = left      (CCW perpendicular)

    Corridor A: x ∈ [0, la],     y ∈ [-hw, hw]  → robot walks in +x
    At the end the robot turns left (into +y) and enters:
    Corridor B: x ∈ [la-hw, la+hw], y ∈ [0, lb]  → robot walks in +y

    Parameters
    ----------
    origin      : (2,) world XY of the maze start (robot's initial XY).
    initial_yaw : Robot's initial heading in radians (CCW from world +X).
    hw          : Corridor half-width (metres, default 0.5 → 1.0 m total).
    wt          : Wall half-thickness (metres, default 0.15 → 0.3 m total).
    la          : Length of Corridor A in metres.
    lb          : Length of Corridor B in metres.
    spacing     : Waypoint spacing in metres.
    """

    def __init__(
        self,
        origin: np.ndarray,
        initial_yaw: float,
        hw: float = 0.7,
        wt: float = 0.15,
        la: float = 5.0,
        lb: float = 5.0,
        spacing: float = 0.05,
    ):
        self.origin      = np.asarray(origin[:2], dtype=np.float64)
        self.initial_yaw = float(initial_yaw)
        self.hw = hw
        self.wt = wt
        self.la = la
        self.lb = lb

        c, s = np.cos(initial_yaw), np.sin(initial_yaw)
        self._R = np.array([[c, -s], [s, c]], dtype=np.float64)

        self.walls     = self._build_walls()
        self.waypoints = self._build_waypoints(spacing)
        self.goal_xy   = self.waypoints[-1].copy()

        print(f"Maze:  {len(self.walls)} walls, {len(self.waypoints)} waypoints")
        print(f"  Start: {self.origin},  Yaw: {np.degrees(initial_yaw):.1f}°")
        print(f"  Goal:  {self.goal_xy}")

    # ------------------------------------------------------------------

    def _w(self, local_xy: np.ndarray) -> np.ndarray:
        """Local → world transform (vectorised)."""
        a = np.asarray(local_xy, dtype=np.float64)
        if a.ndim == 1:
            return self._R @ a + self.origin
        return (self._R @ a.T).T + self.origin

    def _build_walls(self) -> list:
        hw, wt, la, lb = self.hw, self.wt, self.la, self.lb
        y0 = self.initial_yaw

        # Each entry: (cx_local, cy_local, hx_local, hy_local)
        walls_local = [
            # ── Corridor A ───────────────────────────────────────────────
            # South wall  (inner edge y = -hw, runs along full A + east side)
            ((la + hw) / 2,
             -(hw + wt),
             (la + hw) / 2 + wt,
             wt),
            # North wall of A  (inner edge y = +hw, only until the turn at x = la-hw)
            ((la - hw) / 2 - wt / 2,
             hw + wt,
             (la - hw) / 2 + wt / 2,
             wt),
            # ── Corridor B ───────────────────────────────────────────────
            # West wall of B  (inner edge x = la-hw, y ∈ [hw, lb])
            (la - hw - wt,
             (hw + lb) / 2 + wt / 2,
             wt,
             (lb - hw) / 2 + wt / 2),
            # East wall of B  (inner edge x = la+hw, also closes east end of A)
            (la + hw + wt,
             (lb - hw) / 2 - wt / 2,
             wt,
             (lb + hw) / 2 + wt),
            # Top wall of B  (inner edge y = lb)
            (la,
             lb + wt,
             hw + wt,
             wt),
            # ── Caps ─────────────────────────────────────────────────────
            # Start cap  (closes corridor A at x ≈ 0)
            (-wt,
             0.0,
             wt,
             hw + wt),
        ]

        obstacles = []
        for cx_l, cy_l, hx_l, hy_l in walls_local:
            cw = self._w(np.array([cx_l, cy_l]))
            obstacles.append(BoxObstacle(cw, np.array([hx_l, hy_l]), y0))
        return obstacles

    def _build_waypoints(self, spacing: float) -> np.ndarray:
        la, lb = self.la, self.lb
        n_a = max(2, int(la / spacing) + 1)
        n_b = max(2, int(lb / spacing) + 1)

        wp_a = np.column_stack([np.linspace(0, la, n_a), np.zeros(n_a)])
        wp_b = np.column_stack([np.full(n_b, la), np.linspace(0, lb, n_b)])
        local = np.vstack([wp_a, wp_b[1:]])   # skip duplicate junction point
        return self._w(local)


# ---------------------------------------------------------------------------
# Waypoint path controller
# ---------------------------------------------------------------------------

class PathController:
    """
    Tracks progress along a dense waypoint path and supplies future-trajectory
    inputs to the motion diffusion model.

    Progress advances only forward: it is set to the closest path point
    that is ≥ the current arc-length position.

    Parameters
    ----------
    waypoints    : (N, 2) world XY waypoints (dense, e.g. every 0.05 m).
    speed        : Estimated walking speed in m/s (determines lookahead spacing).
    fps          : Simulation frame rate.
    future_frames: Number of future frames to generate.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        speed: float = 0.5,
        fps: float = 30.0,
        future_frames: int = 45,
    ):
        self.waypoints     = np.asarray(waypoints, dtype=np.float64)
        self.speed         = float(speed)
        self.fps           = float(fps)
        self.future_frames = int(future_frames)

        diffs = np.diff(self.waypoints, axis=0)
        self._arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        self.total_length = float(self._arc[-1])
        self._progress    = 0.0

    def reset(self):
        self._progress = 0.0

    @property
    def is_done(self) -> bool:
        return self._progress >= self.total_length - 0.3

    def update(self, robot_xy: np.ndarray):
        """Advance progress to the nearest path point ≥ current progress."""
        robot_xy = np.asarray(robot_xy[:2], dtype=np.float64)
        lo   = max(0, int(np.searchsorted(self._arc, self._progress)) - 3)
        dists = np.linalg.norm(self.waypoints[lo:] - robot_xy, axis=1)
        best_arc = self._arc[lo + int(np.argmin(dists))]
        if best_arc > self._progress:
            self._progress = best_arc

    def get_future_trajectory(self):
        """
        Returns
        -------
        future_traj   : (future_frames, 2)  world XY
        future_orient : (future_frames, 4)  wxyz quaternions
        """
        step = self.speed / self.fps
        traj_xy, traj_quat = [], []
        for i in range(1, self.future_frames + 1):
            arc = min(self._progress + i * step, self.total_length)
            traj_xy.append(self._interp_xy(arc))
            traj_quat.append(_yaw_to_quat(self._interp_yaw(arc)))
        return np.array(traj_xy, dtype=np.float32), np.array(traj_quat, dtype=np.float32)

    # ------------------------------------------------------------------ helpers

    def _interp_xy(self, arc: float) -> np.ndarray:
        idx  = int(np.clip(np.searchsorted(self._arc, arc) - 1, 0, len(self.waypoints) - 2))
        span = max(self._arc[idx + 1] - self._arc[idx], 1e-9)
        t    = float(np.clip((arc - self._arc[idx]) / span, 0.0, 1.0))
        return self.waypoints[idx] + t * (self.waypoints[idx + 1] - self.waypoints[idx])

    def _interp_yaw(self, arc: float) -> float:
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
            kw.get('past_motion'),
            kw.get('traj_pose'),
            kw.get('traj_trans'),
            kw.get('style_idx'),
            kw.get('sensor'),
        )


class SensorMotionGenerator:
    """Autoregressive motion generator with optional sensor conditioning."""

    def __init__(self, model, diffusion, config, sensor: EnvironmentSensor,
                 device='cuda', sampler='ddpm'):
        self.model     = _ModelWrapper(model)
        self.diffusion = diffusion
        self.sensor    = sensor
        self.device    = device
        self.sampler   = sampler.lower()
        self.future_frames  = config.arch.future_frame
        self.rot_req        = config.arch.rot_req
        self.per_rot_feat   = 6

    def generate_motion(
        self,
        past_qpos: np.ndarray,
        traj_trans: np.ndarray,
        traj_pose: np.ndarray,
        style_idx: int,
        obstacles: list,
        zero_sensor: bool = False,
    ) -> np.ndarray:
        """
        Args
        ----
        past_qpos   : (past_frames, 36)
        traj_trans  : (future_frames, 2)  world XY
        traj_pose   : (future_frames, 4)  wxyz quaternions
        obstacles   : list of Obstacle2D
        zero_sensor : if True, pass all-zero sensor input (ablation)

        Returns
        -------
        (future_frames, 36) qpos
        """
        curr_xy = past_qpos[-1, :2].copy()

        # Past motion (relative to current position)
        pq = past_qpos.copy()
        pq[:, :2] -= curr_xy
        past_t = torch.from_numpy(qpos_to_model_format(pq)).float() \
            .unsqueeze(0).permute(0, 2, 3, 1).to(self.device)   # (1, 31, 6, past)

        # Trajectory
        traj_rel  = (traj_trans - curr_xy).astype(np.float32)
        traj_tr_t = torch.from_numpy(traj_rel).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)       # (1, 2, future)
        traj_repr = nn_transforms.get_rotation(
            torch.from_numpy(traj_pose).float(), self.rot_req
        ).numpy()
        traj_po_t = torch.from_numpy(traj_repr).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)       # (1, 6, future)

        # Sensor reading
        if zero_sensor:
            readings = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        else:
            yaw = quat_wxyz_to_yaw(past_qpos[-1, 3:7])
            readings, _ = self.sensor.compute(past_qpos[-1, :3], yaw, obstacles)
        sensor_t = torch.from_numpy(readings).float().unsqueeze(0).to(self.device)

        style_t = torch.tensor([style_idx]).to(self.device)

        model_kwargs = dict(
            past_motion=past_t,
            traj_trans=traj_tr_t,
            traj_pose=traj_po_t,
            sensor=sensor_t,
            style_idx=style_t,
            y={},
        )

        shape = (1, 31, self.per_rot_feat, self.future_frames)
        with torch.no_grad():
            if self.sampler == 'ddim':
                out = self.diffusion.ddim_sample_loop(
                    self.model, shape, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, eta=0.0,
                    device=self.device,
                )
            else:
                out = self.diffusion.p_sample_loop(
                    self.model, shape, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False,
                    device=self.device,
                )

        out = out.squeeze(0).permute(2, 0, 1).cpu().numpy()
        qpos_out = model_format_to_qpos(out)
        qpos_out[:, :2] += curr_xy
        return qpos_out


# ---------------------------------------------------------------------------
# Maze demo player
# ---------------------------------------------------------------------------

class DemoPlayerMaze:
    """
    Autoregressive demo player for the maze scenario.

    Unlike DemoPlayerEnv, the trajectory comes from PathController (not from
    the dataset), and the obstacles are the static maze walls.
    """

    def __init__(
        self,
        mj_model, mj_data, dataset,
        generator: SensorMotionGenerator,
        maze: MazeLayout,
        path: PathController,
        motion_idx: int = 0,
        show_trajectory: bool = True,
        past_frames: int = 10,
        future_frames: int = 45,
        blend: bool = True,            # False = skip blending, use raw waypoint
        traj_bias_pos: float = 0.4,   # position blend exponent (higher = more waypoint-biased)
        traj_bias_rot: float = 2.2,   # rotation blend exponent
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
        self.dataset   = dataset
        self.generator = generator
        self.maze      = maze
        self.path      = path
        self.obstacles = maze.walls       # static – never changes

        # Display toggles
        self.show_trajectory = show_trajectory
        self.show_sensor     = True
        self.show_obstacles  = True
        self.draw_lines      = False
        self.camera_follow   = True
        self.zero_sensor     = False      # ablation: N toggles this

        # Timing
        self.fps          = 30
        self.frame_dt     = 1.0 / self.fps
        self.playback_speed = 1.0
        self.playing      = True
        self.last_update  = time.time()

        # Generation state
        self.past_frames  = past_frames
        self.future_frames = future_frames
        self.apply_frames = int(applyframes)
        self.gen_idx      = 0
        self.gen_qpos     = None

        self.cfg_count_cache = int(cfg_count)
        self.cfg_count       = int(cfg_count)

        # Inertialization
        self.inertialize          = bool(inertialize)
        self.inertialization_mode = str(inertialization_mode).lower()
        self.blendtime_rotation   = float(blendtime_rotation)
        self.blendtime_position   = float(blendtime_position)
        self.spring_halflife_pos  = float(spring_halflife_position)
        self.spring_halflife_rot  = float(spring_halflife_rotation)
        self.quat_slice           = slice(int(inertial_quat_start), int(inertial_quat_end))
        self.transition_mgr       = None

        # Sensor state
        self.sensor  = generator.sensor
        self.readings = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        self.sphere_centers = np.zeros((self.sensor.feature_dim, 2), dtype=np.float64)

        # Blend parameters
        self.blend         = bool(blend)
        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)

        # Trajectory state (filled by _update_traj)
        self.future_traj_waypoint = None   # raw waypoint path  (red)
        self.future_orient_waypoint = None
        self.future_traj   = None          # blended = fed to model (green)
        self.future_orient = None
        self.past_traj     = None
        self.past_orient   = None

        # Model-predicted future trajectory (gray overlay)
        self.pred_traj   = None
        self.pred_orient = None

        # Style from dataset
        self.motion_idx = motion_idx % len(dataset)
        self.style_idx  = dataset[self.motion_idx].style_idx

        self.qpos_history = deque(maxlen=past_frames)
        self._init_pose()
        self._update_traj()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_pose(self):
        qpos = self.dataset[self.motion_idx].get_qpos(0)
        for _ in range(self.past_frames):
            self.qpos_history.append(qpos.copy())
        self.mj_data.qpos[:] = qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self):
        self.path.reset()
        self.gen_idx   = 0
        self.gen_qpos  = None
        self.pred_traj   = None
        self.pred_orient = None
        self.cfg_count = self.cfg_count_cache
        self.transition_mgr = None
        self.qpos_history.clear()
        self._init_pose()
        self._update_traj()
        print("Reset.")

    # ------------------------------------------------------------------
    # Trajectory update (from PathController)
    # ------------------------------------------------------------------

    def _update_traj(self):
        qh = np.array(self.qpos_history)
        self.past_traj   = qh[-self.past_frames:, :3]
        self.past_orient = qh[-self.past_frames:, 3:7]

        curr_xy = self.mj_data.qpos[:2]
        self.path.update(curr_xy)
        wp_t, wp_o = self.path.get_future_trajectory()
        self.future_traj_waypoint   = wp_t
        self.future_orient_waypoint = wp_o

        # Blend model-predicted path with waypoint path (like step3)
        if self.blend and self.gen_qpos is not None:
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

    # ------------------------------------------------------------------
    # Motion generation helpers
    # ------------------------------------------------------------------

    def _generate(self) -> np.ndarray:
        past_qpos = np.array(self.qpos_history)
        return self.generator.generate_motion(
            past_qpos,
            self.future_traj,
            self.future_orient,
            self.style_idx,
            self.obstacles,
            zero_sensor=self.zero_sensor,
        )

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
        # Update sensor readings at current pose
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
            # Blue  – past robot path
            draw_trajectory(scene, self.past_traj, self.past_orient,
                            color=[0.2, 0.5, 1.0, 1.0])
            # Red   – raw waypoint path (command)
            if self.future_traj_waypoint is not None:
                wt3 = np.hstack([self.future_traj_waypoint,
                                  np.zeros((len(self.future_traj_waypoint), 1))])
                draw_trajectory(scene, wt3, self.future_orient_waypoint,
                                color=[1.0, 0.2, 0.2, 1.0])
            # Green – blended trajectory fed to the model
            if self.future_traj is not None:
                ft3 = np.hstack([self.future_traj, np.zeros((len(self.future_traj), 1))])
                draw_trajectory(scene, ft3, self.future_orient,
                                color=[0.2, 1.0, 0.2, 1.0])
            # Gray  – model's raw predicted output path
            if self.pred_traj is not None and len(self.pred_traj) >= 2:
                draw_trajectory(scene, self.pred_traj, self.pred_orient,
                                color=[0.3, 0.3, 0.3, 0.5])

        if self.show_obstacles:
            for obs in self.obstacles:
                if isinstance(obs, BoxObstacle):
                    draw_obstacle_box(scene, obs.center, obs.half_extents, obs.yaw,
                                      height=1.8, color=[0.55, 0.55, 0.55, 0.7])

        if self.show_sensor:
            draw_sensor_readings(
                scene, self.mj_data.qpos[:3], self.readings, self.sphere_centers,
                z_height=0.08, dot_radius=0.04, draw_lines=self.draw_lines,
            )

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory

    def toggle_sensor(self):
        self.show_sensor = not self.show_sensor

    def toggle_obstacles(self):
        self.show_obstacles = not self.show_obstacles

    def toggle_sensor_lines(self):
        self.draw_lines = not self.draw_lines

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow

    def toggle_zero_sensor(self):
        self.zero_sensor = not self.zero_sensor
        label = "OFF (zero input)" if self.zero_sensor else "ON"
        print(f"Sensor conditioning: {label}")

    def set_speed(self, s: float):
        self.playback_speed = s
        print(f"Speed: {s}×")

    def print_status(self):
        n_act = int((self.readings > 0).sum())
        prog  = f"{self.path._progress:.1f} / {self.path.total_length:.1f} m"
        print(
            f"Sensor {'OFF' if self.zero_sensor else 'ON'} | "
            f"Path {prog} | "
            f"sensor active {n_act}/{self.sensor.feature_dim} | "
            f"walls {len(self.obstacles)}"
        )


# ---------------------------------------------------------------------------
# Keyboard callback
# ---------------------------------------------------------------------------

_SPEED_MAP = {
    ord('1'): 0.25, ord('2'): 0.50, ord('3'): 0.75, ord('4'): 0.90,
    ord('5'): 1.00, ord('6'): 1.25, ord('7'): 1.50, ord('8'): 1.75,
    ord('9'): 2.00,
}


def key_callback(player: DemoPlayerMaze, keycode: int):
    if keycode == 32:                             # SPACE
        player.toggle_pause()
    elif keycode in (ord('r'), ord('R')):
        player.reset()
    elif keycode in (ord('n'), ord('N')):
        player.toggle_zero_sensor()
    elif keycode in (ord('t'), ord('T')):
        player.toggle_trajectory()
    elif keycode in (ord('e'), ord('E')):
        player.toggle_sensor()
    elif keycode in (ord('o'), ord('O')):
        player.toggle_obstacles()
    elif keycode in (ord('l'), ord('L')):
        player.toggle_sensor_lines()
    elif keycode in (ord('c'), ord('C')):
        player.toggle_camera_follow()
    elif keycode in (ord('s'), ord('S')):
        player.print_status()
    elif keycode in _SPEED_MAP:
        player.set_speed(_SPEED_MAP[keycode])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Maze demo – sensor-conditioned avoidance")
    p.add_argument("--dataset",       default="lafan1_g1_motion28")
    p.add_argument("--checkpoint",    default="save/camdm_g1_env/best.pt")
    p.add_argument("--motion",        type=int,   default=0,
                   help="Motion clip index from dataset (for initial pose & style)")
    p.add_argument("--corridor-hw",   type=float, default=0.5,
                   help="Corridor half-width in metres (default 0.5 → 1.0 m wide)")
    p.add_argument("--corridor-a",    type=float, default=5.0,
                   help="Length of corridor A (forward arm) in metres")
    p.add_argument("--corridor-b",    type=float, default=5.0,
                   help="Length of corridor B (left-turn arm) in metres")
    p.add_argument("--speed",         type=float, default=0.7,
                   help="Path-following speed in m/s: controls how fast the red-arrow waypoints advance. "
                        "Increase (e.g. 1.0–1.5) to pull the robot faster along the corridor.")
    p.add_argument("--resolution",    type=int,   default=9)
    p.add_argument("--max-range",     type=float, default=2.0)
    p.add_argument("--past-frames",   type=int,   default=10)
    p.add_argument("--future-frames", type=int,   default=45)
    p.add_argument("--no-blend",       action="store_true",
                   help="Disable trajectory blending; feed raw waypoint path directly to the model")
    p.add_argument("--traj-bias-pos",  type=float, default=0.4,
                   help="Position blend exponent: higher = trajectory pulled more toward waypoints")
    p.add_argument("--traj-bias-rot",  type=float, default=2.2,
                   help="Rotation blend exponent")
    p.add_argument("--sampler",       default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--cfg-count",     type=int,   default=2)
    p.add_argument("--applyframes",   type=int,   default=15)
    p.add_argument("--inertialize",   default="on", choices=["on", "off"])
    p.add_argument("--inertialization-mode", default="camdm", choices=["camdm", "spring"])
    p.add_argument("--blendtime-rotation",   type=float, default=0.2)
    p.add_argument("--blendtime-position",   type=float, default=0.2)
    p.add_argument("--spring-halflife-position", type=float, default=0.12)
    p.add_argument("--spring-halflife-rotation",  type=float, default=0.12)
    return p.parse_args()


def print_instructions():
    print("\n" + "=" * 60)
    print("  Step 4: Maze Demo  –  Controls")
    print("-" * 60)
    print("  SPACE      : Pause / Resume")
    print("  R          : Reset robot to start")
    print("  N          : Toggle sensor ON/OFF  (ablation)")
    print("  T          : Toggle trajectory")
    print("  E          : Toggle sensor overlay")
    print("  O          : Toggle walls")
    print("  L          : Toggle sensor ray lines")
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
    print("  Step 4: Maze Demo – Sensor-Conditioned Avoidance")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data  = mujoco.MjData(mj_model)

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    dataset = MotionDataset(dataset_path)
    dataset.print_summary()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config     = checkpoint["config"]

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
    diffusion_model.load_state_dict(checkpoint["state_dict"])
    diffusion_model.eval()

    diffusion = create_gaussian_diffusion(config)
    sensor    = EnvironmentSensor(max_range=args.max_range, resolution=args.resolution)
    generator = SensorMotionGenerator(
        diffusion_model, diffusion, config, sensor,
        device=device, sampler=args.sampler,
    )

    # ── Build maze relative to robot's initial position & heading ──────────
    motion_idx  = args.motion % len(dataset)
    init_qpos   = dataset[motion_idx].get_qpos(0)
    init_xy     = init_qpos[:2].copy()
    init_yaw    = quat_wxyz_to_yaw(init_qpos[3:7])

    maze = MazeLayout(
        origin=init_xy,
        initial_yaw=init_yaw,
        hw=args.corridor_hw,
        la=args.corridor_a,
        lb=args.corridor_b,
    )
    path = PathController(
        waypoints=maze.waypoints,
        speed=args.speed,
        fps=30.0,
        future_frames=args.future_frames,
    )

    player = DemoPlayerMaze(
        mj_model, mj_data, dataset, generator, maze, path,
        motion_idx=motion_idx,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        blend=not args.no_blend,
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
    )

    # ── Video recording ────────────────────────────────────────────────────
    os.makedirs("videos", exist_ok=True)
    video_path = f"videos/demo_maze_{time.strftime('%m%d_%H%M')}.mp4"
    W, H = 1280, 720
    FPS  = player.fps
    writer   = imageio.get_writer(video_path, fps=FPS, codec="libx264", pixelformat="yuv420p")
    renderer = mujoco.Renderer(mj_model, height=H, width=W)
    frame_last_time = -np.inf
    print(f"\nRecording → {video_path}  ({W}×{H} @ {FPS}fps)")

    print_instructions()

    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=lambda kc: key_callback(player, kc),
    ) as viewer:
        viewer.cam.distance = 8.0
        viewer.sync()
        try:
            while viewer.is_running():
                player.step()

                viewer.user_scn.ngeom = 0
                player.render(viewer.user_scn)
                if player.camera_follow:
                    viewer.cam.lookat[:] = mj_data.qpos[:3]
                viewer.sync()

                if time.time() - frame_last_time > 1.0 / FPS:
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
