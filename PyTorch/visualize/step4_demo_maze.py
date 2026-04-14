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

import heapq

import numpy as np
import mujoco
import mujoco.viewer
import torch
import imageio.v2 as imageio
from scipy.interpolate import splprep, splev

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models_env2d import MotionDiffusionEnv
from diffusion.create_diffusion import create_gaussian_diffusion

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import (
    draw_trajectory,
    draw_trajectory_lines,
    draw_sensor_readings,
    draw_obstacle_box,
    draw_label,
)
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic
from visualize.utils.sdf_guidance import make_obstacle_cond_fn
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
# Maze layout  –  two rooms + narrow corridor
# ---------------------------------------------------------------------------

def _astar_path(
    open_rects: list,
    start: np.ndarray,
    goal: np.ndarray,
    resolution: float = 0.1,
    robot_radius: float = 0.15,
) -> np.ndarray:
    """
    8-directional A* path planner in the local frame.

    Cells inside ``open_rects`` (eroded by ``robot_radius``) are passable;
    everything else is a wall.

    Returns an (N, 2) array of path points in local frame.
    Falls back to a direct straight line if no path is found.
    """
    margin = 0.5
    x_lo = min(r[0] for r in open_rects) - margin
    x_hi = max(r[1] for r in open_rects) + margin
    y_lo = min(r[2] for r in open_rects) - margin
    y_hi = max(r[3] for r in open_rects) + margin

    res = float(resolution)
    nx  = int(np.ceil((x_hi - x_lo) / res)) + 1
    ny  = int(np.ceil((y_hi - y_lo) / res)) + 1

    # Build passable-cell mask (vectorised)
    gx, gy = np.meshgrid(
        x_lo + np.arange(nx) * res,
        y_lo + np.arange(ny) * res,
        indexing='ij',
    )
    passable = np.zeros((nx, ny), dtype=bool)
    r = robot_radius
    for (rxlo, rxhi, rylo, ryhi) in open_rects:
        passable |= (gx >= rxlo + r) & (gx <= rxhi - r) & \
                    (gy >= rylo + r) & (gy <= ryhi - r)

    def to_idx(p):
        ix = int(round((float(p[0]) - x_lo) / res))
        iy = int(round((float(p[1]) - y_lo) / res))
        return np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)

    def to_xy(ix, iy):
        return np.array([x_lo + ix * res, y_lo + iy * res])

    si, gi = to_idx(start), to_idx(goal)

    dirs  = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    costs = [res]*4 + [res * 1.41421]*4

    g_cost = {si: 0.0}
    parent: dict = {}
    tie   = 0
    heap  = [(np.hypot(si[0]-gi[0], si[1]-gi[1]) * res, tie, si)]

    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur == gi:
            path = []
            while cur in parent:
                path.append(to_xy(*cur))
                cur = parent[cur]
            path.append(to_xy(*si))
            return np.array(path[::-1])
        for (dx, dy), c in zip(dirs, costs):
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if not passable[nb]:
                continue
            ng = g_cost[cur] + c
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                parent[nb] = cur
                tie += 1
                f = ng + np.hypot(nb[0]-gi[0], nb[1]-gi[1]) * res
                heapq.heappush(heap, (f, tie, nb))

    return np.array([start, goal])   # fallback


def _spline_waypoints(
    astar_pts: np.ndarray,
    spacing: float,
    n_ctrl: int = 30,
    s_smooth: float = 0.3,
) -> np.ndarray:
    """
    Smooth an A* path with a parametric cubic spline and resample at ``spacing``.

    1. Coarsen to ``n_ctrl`` control points (uniform re-sampling).
    2. Fit ``splprep`` spline with smoothing ``s_smooth * n_ctrl``.
    3. Evaluate at uniform arc-length steps of ``spacing`` m.
    """
    # Coarsen
    idx  = np.round(np.linspace(0, len(astar_pts) - 1, n_ctrl)).astype(int)
    ctrl = astar_pts[idx]

    # Remove consecutive duplicates
    keep = np.concatenate([[True], np.any(np.diff(ctrl, axis=0) != 0, axis=1)])
    ctrl = ctrl[keep]
    if len(ctrl) < 4:
        return ctrl

    try:
        tck, _ = splprep([ctrl[:, 0], ctrl[:, 1]],
                         s=s_smooth * len(ctrl), k=3)
    except Exception:
        return ctrl

    # Estimate total arc length via dense evaluation, then resample
    u_dense = np.linspace(0, 1, max(500, int(len(astar_pts) * 2)))
    xy_dense = np.column_stack(splev(u_dense, tck))
    arc = np.concatenate([[0.0],
                           np.cumsum(np.linalg.norm(np.diff(xy_dense, axis=0), axis=1))])
    n_pts = max(2, int(arc[-1] / spacing))
    u_uniform = np.interp(np.linspace(0, arc[-1], n_pts), arc, u_dense)
    return np.column_stack(splev(u_uniform, tck))


class MazeLayout:
    """
    Two large rooms connected by a single narrow corridor (top area).

    Local frame  (x = initial heading forward, y = left):

        ┌──────────────┐  ┌──┐  ┌──────────────┐
        │              │  │  │  │              │
        │  Left room   │  │co│  │  Right room  │
        │    (Goal)    │  │rr│  │   (Start)    │
        │              │  │  │  │              │
        └──────────────┘  └──┘  └──────────────┘
          x ∈ [x_l, x_l+rw]  corridor  x ∈ [0, rw]

    Walls are flood-filled with a tight box grid so every sensor sphere
    that lands outside a room or the corridor reads ≥ 1 (fully red).

    Parameters
    ----------
    origin        : (2,) world XY that maps to local (0, 0).
                    Pass ``init_xy - R @ start_local`` so the robot's initial
                    world position coincides with the first waypoint.
    initial_yaw   : Robot heading in radians (CCW from world +X).
    room_w        : Room width in the x direction (metres).
    room_h        : Room height in the y direction (metres).
    corridor_len  : Length of the corridor in the x direction (metres).
    corridor_hw   : Half-width of the corridor in the y direction (metres).
    corridor_y    : Y-centre of the corridor (offset from room centre, metres).
    grid_spacing  : Box grid pitch for flood-fill walls (metres).
    box_half      : Half-extent of each wall box (metres).  Should be
                    ≥ grid_spacing/2 so boxes tile flush with no gaps.
    wp_spacing    : Waypoint spacing (metres).
    """

    # These define where on the path the robot starts and ends (local frame).
    # start_local is (room_w * START_FRAC, 0); goal_local is in the left room.
    START_FRAC = 0.75   # fraction of room_w → start near the right side of right room
    GOAL_FRAC  = 0.25   # fraction of room_w → goal near the right side of left room

    def __init__(
        self,
        origin: np.ndarray,
        initial_yaw: float,
        room_w: float = 4.0,
        room_h: float = 4.0,
        corridor_len: float = 1.5,
        corridor_hw: float = 0.45,
        corridor_y: float = 1.0,
        grid_spacing: float = 0.5,
        box_half: float = 0.26,
        wp_spacing: float = 0.05,
    ):
        self.origin      = np.asarray(origin[:2], dtype=np.float64)
        self.initial_yaw = float(initial_yaw)

        c, s = np.cos(initial_yaw), np.sin(initial_yaw)
        self._R = np.array([[c, -s], [s, c]], dtype=np.float64)

        self.room_w       = float(room_w)
        self.room_h       = float(room_h)
        self.corridor_len = float(corridor_len)
        self.corridor_hw  = float(corridor_hw)
        self.corridor_y   = float(corridor_y)

        # Local-frame positions of the three open areas:
        #   Right room :  x ∈ [0, rw],                      y ∈ [-rh/2, rh/2]
        #   Corridor   :  x ∈ [rw, rw+cl],                  y ∈ [cy-chw, cy+chw]
        #   Left room  :  x ∈ [rw+cl, 2*rw+cl],             y ∈ [-rh/2, rh/2]
        rw, rh, cl  = room_w, room_h, corridor_len
        chw, cy     = corridor_hw, corridor_y
        rhw         = rh / 2.0
        x_left_lo   = rw + cl

        self._open_rects = [           # (xlo, xhi, ylo, yhi) in local frame
            (0.0,       rw,           -rhw,         rhw),          # right room
            (rw,        rw + cl,       cy - chw,    cy + chw),     # corridor
            (x_left_lo, x_left_lo+rw, -rhw,         rhw),          # left room
        ]

        self.walls     = self._flood_fill(grid_spacing, box_half)
        self.waypoints = self._build_waypoints(rw, cl, wp_spacing)
        self.goal_xy   = self.waypoints[-1].copy()

        print(f"Maze:  {len(self.walls)} wall boxes, {len(self.waypoints)} waypoints")
        print(f"  Origin: {self.origin},  Yaw: {np.degrees(initial_yaw):.1f}°")
        print(f"  Goal:   {self.goal_xy}")

    # ------------------------------------------------------------------

    def _w(self, local_xy: np.ndarray) -> np.ndarray:
        """Local → world (vectorised)."""
        a = np.asarray(local_xy, dtype=np.float64)
        if a.ndim == 1:
            return self._R @ a + self.origin
        return (self._R @ a.T).T + self.origin

    def _flood_fill(self, gs: float, h: float) -> list:
        """
        Place axis-aligned box obstacles on a regular grid, omitting any box
        whose footprint overlaps one of the three open rectangles.

        The grid extends ``sensor_buffer`` beyond the open area so the robot's
        sensor is fully occluded in every direction outside the rooms.
        """
        sensor_buffer = 2.5  # slightly larger than max sensor range (2.0 m)

        xs_lo = min(r[0] for r in self._open_rects) - sensor_buffer
        xs_hi = max(r[1] for r in self._open_rects) + sensor_buffer
        ys_lo = min(r[2] for r in self._open_rects) - sensor_buffer
        ys_hi = max(r[3] for r in self._open_rects) + sensor_buffer

        xs = np.arange(xs_lo, xs_hi + gs * 0.5, gs)
        ys = np.arange(ys_lo, ys_hi + gs * 0.5, gs)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2) local

        # Mark points whose box [cx-h, cx+h]×[cy-h, cy+h] overlaps any open rect
        in_open = np.zeros(len(pts), dtype=bool)
        for (xlo, xhi, ylo, yhi) in self._open_rects:
            in_open |= (
                (pts[:, 0] + h > xlo) & (pts[:, 0] - h < xhi) &
                (pts[:, 1] + h > ylo) & (pts[:, 1] - h < yhi)
            )

        wall_pts_local = pts[~in_open]
        half = np.array([h, h], dtype=np.float64)
        return [
            BoxObstacle(self._w(p), half, self.initial_yaw)
            for p in wall_pts_local
        ]

    def _build_waypoints(
        self, rw, cl, spacing
    ) -> np.ndarray:
        """
        Plan a path in local frame using A* through the open rectangles,
        then smooth with a parametric cubic spline.

        Path shape (arch):
          Start (right room, y=0) → arc up through corridor (y≈cy) → Goal (left room, y=0)
        """
        sx = rw * self.START_FRAC           # start x in right room
        gx = rw + cl + rw * self.GOAL_FRAC  # goal x in left room

        start_local = np.array([sx,  0.0])
        goal_local  = np.array([gx,  0.0])

        astar_pts = _astar_path(
            self._open_rects,
            start_local,
            goal_local,
            resolution=0.1,
            robot_radius=0.0,   # path is a reference only; model handles avoidance
        )
        self.astar_pts_world = self._w(astar_pts)   # store for visualization
        local = _spline_waypoints(astar_pts, spacing)
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
        guidance_scale: float = 0.0,
        guidance_margin: float = 0.3,
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

        # SDF guidance: build cond_fn if guidance_scale > 0
        cond_fn = None
        if guidance_scale > 0.0 and obstacles:
            cond_fn = make_obstacle_cond_fn(
                obstacles, curr_xy, device=self.device,
                margin=guidance_margin, scale=guidance_scale,
            )

        shape    = (1, 31, self.per_rot_feat, self.future_frames)
        use_grad = cond_fn is not None
        # When cond_fn is active we need enable_grad (p_sample_with_grad rebuilds
        # the graph inside its own th.enable_grad block).  When guidance is off,
        # use no_grad for speed.
        grad_ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with grad_ctx:
            if self.sampler == 'ddim':
                out = self.diffusion.ddim_sample_loop(
                    self.model, shape, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, eta=0.0,
                    device=self.device,
                    cond_fn=cond_fn, cond_fn_with_grad=use_grad,
                )
            else:
                out = self.diffusion.p_sample_loop(
                    self.model, shape, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False,
                    device=self.device,
                    cond_fn=cond_fn, cond_fn_with_grad=use_grad,
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
        guidance_scale: float = 0.0,  # SDF guidance strength (0 = off)
        guidance_margin: float = 0.3, # minimum clearance in metres for guidance
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
        self.traj_bias_rot    = float(traj_bias_rot)
        self.guidance_scale   = float(guidance_scale)
        self.guidance_margin  = float(guidance_margin)

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
            guidance_scale=self.guidance_scale,
            guidance_margin=self.guidance_margin,
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
                                      height=0.8, color=[0.55, 0.55, 0.55, 0.7])

        # Yellow line – full A* planned path (start → corridor → goal)
        if self.show_trajectory and hasattr(self.maze, 'astar_pts_world'):
            apts = self.maze.astar_pts_world
            apts3 = np.hstack([apts, np.full((len(apts), 1), 0.04)])
            draw_trajectory_lines(scene, apts3, color=[1.0, 0.85, 0.0, 0.9])

        if self.show_sensor:
            draw_sensor_readings(
                scene, self.mj_data.qpos[:3], self.readings, self.sphere_centers,
                z_height=0.08, dot_radius=0.04, draw_lines=self.draw_lines,
            )

        # Sensor ON/OFF indicator: colored sphere + label above robot's head
        if scene.ngeom < scene.maxgeom - 1:
            robot_pos = self.mj_data.qpos[:3].copy()
            indicator_pos = robot_pos + np.array([0.0, 0.0, 1.6])
            if self.zero_sensor:
                ind_color = np.array([1.0, 0.15, 0.15, 0.95], dtype=np.float32)
                label_text = "SENSOR: OFF"
            else:
                ind_color = np.array([0.15, 1.0, 0.15, 0.95], dtype=np.float32)
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
            draw_label(scene, indicator_pos + np.array([0.0, 0.0, 0.25]), label_text)

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
# Headless comparison utilities
# ---------------------------------------------------------------------------

def _collect_pass(player: DemoPlayerMaze, max_steps: int = 8000) -> np.ndarray:
    """
    Run one full maze pass without real-time throttling.

    Returns
    -------
    qpos : (T, 36) array – one row per simulation step (including initial frame).
    """
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


def run_comparison(args, mj_model, dataset, generator, maze: MazeLayout):
    """
    Run sensor-ON and sensor-OFF passes headlessly, then save a side-by-side
    comparison video:  left = SENSOR ON (green),  right = SENSOR OFF (red).
    """
    fps        = 30
    motion_idx = args.motion % len(dataset)

    def _make_player(mj_data, zero_sensor: bool) -> DemoPlayerMaze:
        path = PathController(
            waypoints=maze.waypoints,
            speed=args.speed,
            fps=fps,
            future_frames=args.future_frames,
        )
        p = DemoPlayerMaze(
            mj_model, mj_data, dataset, generator, maze, path,
            motion_idx=motion_idx,
            past_frames=args.past_frames,
            future_frames=args.future_frames,
            blend=not args.no_blend,
            traj_bias_pos=args.traj_bias_pos,
            traj_bias_rot=args.traj_bias_rot,
            guidance_scale=args.guidance_scale,
            guidance_margin=args.guidance_margin,
            cfg_count=args.cfg_count,
            applyframes=args.applyframes,
            inertialize=(args.inertialize == "on"),
            inertialization_mode=args.inertialization_mode,
            blendtime_rotation=args.blendtime_rotation,
            blendtime_position=args.blendtime_position,
            spring_halflife_position=args.spring_halflife_position,
            spring_halflife_rotation=args.spring_halflife_rotation,
        )
        p.zero_sensor = zero_sensor
        return p

    # ── Pass 1: sensor ON ─────────────────────────────────────────────────
    print("\n[Compare] Pass 1 – SENSOR ON …")
    mj_data_on = mujoco.MjData(mj_model)
    player_on  = _make_player(mj_data_on, zero_sensor=False)
    qpos_on    = _collect_pass(player_on)
    print(f"          {len(qpos_on)} frames recorded")

    # ── Pass 2: sensor OFF ────────────────────────────────────────────────
    print("[Compare] Pass 2 – SENSOR OFF …")
    mj_data_off = mujoco.MjData(mj_model)
    player_off  = _make_player(mj_data_off, zero_sensor=True)
    qpos_off    = _collect_pass(player_off)
    print(f"          {len(qpos_off)} frames recorded")

    os.makedirs("videos", exist_ok=True)
    tag = time.strftime('%m%d_%H%M')

    # ── 2D map figure ────────────────────────────────────────────────────
    fig_path = f"videos/demo_maze_compare_{tag}.png"
    _save_comparison_figure(qpos_on, qpos_off, maze, fig_path)

    # ── Side-by-side MP4 ─────────────────────────────────────────────────
    video_path = f"videos/demo_maze_compare_{tag}.mp4"
    _save_comparison_video(qpos_on, qpos_off, mj_model, maze, fps, video_path)


def _save_comparison_video(
    qpos_on: np.ndarray,
    qpos_off: np.ndarray,
    mj_model,
    maze: MazeLayout,
    fps: int,
    output_path: str,
    W: int = 640,
    H: int = 720,
):
    """
    Render a side-by-side MP4:  left = SENSOR ON (green),  right = SENSOR OFF (red).
    Both panels share a fixed overhead camera centred on the maze.
    """
    # Fixed overhead camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    center_local = np.array([maze.room_w + maze.corridor_len * 0.5,
                              maze.corridor_y * 0.4])
    center_world = maze._w(center_local)
    cam.lookat[:] = [center_world[0], center_world[1], 0.4]
    cam.distance  = 14.0
    cam.elevation = -65.0
    cam.azimuth   = float(np.degrees(maze.initial_yaw) - 90.0)

    mj_data_on  = mujoco.MjData(mj_model)
    mj_data_off = mujoco.MjData(mj_model)
    ren_on  = mujoco.Renderer(mj_model, height=H, width=W)
    ren_off = mujoco.Renderer(mj_model, height=H, width=W)

    def _draw_overlay(scene, past_xy: np.ndarray, traj_color: list, label: str):
        for obs in maze.walls:
            draw_obstacle_box(scene, obs.center, obs.half_extents, obs.yaw,
                              height=0.8, color=[0.55, 0.55, 0.55, 0.7])
        if hasattr(maze, 'astar_pts_world'):
            apts = maze.astar_pts_world
            apts3 = np.hstack([apts, np.full((len(apts), 1), 0.04)])
            draw_trajectory_lines(scene, apts3, color=[1.0, 0.85, 0.0, 0.7])
        if len(past_xy) >= 2:
            past3 = np.ascontiguousarray(
                np.hstack([past_xy, np.full((len(past_xy), 1), 0.05)]),
                dtype=np.float64,
            )
            draw_trajectory_lines(scene, past3, color=traj_color)
        draw_label(scene, np.array([center_world[0], center_world[1], 2.8]), label)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    n = max(len(qpos_on), len(qpos_off))
    print(f"[Compare] Rendering {n} frames for MP4 …")

    for t in range(n):
        t_on  = min(t, len(qpos_on)  - 1)
        t_off = min(t, len(qpos_off) - 1)

        mj_data_on.qpos[:] = qpos_on[t_on]
        mujoco.mj_forward(mj_model, mj_data_on)
        ren_on.update_scene(mj_data_on, camera=cam)
        _draw_overlay(ren_on.scene, qpos_on[:t_on + 1, :2],
                      [0.1, 1.0, 0.1, 0.95], "SENSOR: ON")
        frame_on = ren_on.render().copy()

        mj_data_off.qpos[:] = qpos_off[t_off]
        mujoco.mj_forward(mj_model, mj_data_off)
        ren_off.update_scene(mj_data_off, camera=cam)
        _draw_overlay(ren_off.scene, qpos_off[:t_off + 1, :2],
                      [1.0, 0.15, 0.15, 0.95], "SENSOR: OFF")
        frame_off = ren_off.render().copy()

        writer.append_data(np.hstack([frame_on, frame_off]))

    writer.close()
    print(f"[Compare] MP4 saved  → {output_path}")


def _save_comparison_figure(
    qpos_on: np.ndarray,
    qpos_off: np.ndarray,
    maze: MazeLayout,
    output_path: str,
):
    """
    Save a 2-D top-down map comparing sensor-ON vs sensor-OFF trajectories.

    Layout
    ------
    Open rooms / corridor  : light gray fill, dark border
    A* reference path      : dashed gold line
    Sensor-ON trajectory   : solid green line  (with start/end arrows)
    Sensor-OFF trajectory  : solid red line    (with start/end arrows)
    Start marker           : blue circle
    Goal marker            : orange star
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(13, 7))

    # ── Maze geometry ────────────────────────────────────────────────────
    # Draw each open rectangle as a light patch (rotated if initial_yaw ≠ 0)
    for (xlo, xhi, ylo, yhi) in maze._open_rects:
        corners_local = np.array([
            [xlo, ylo], [xhi, ylo], [xhi, yhi], [xlo, yhi],
        ])
        corners_world = maze._w(corners_local)
        poly = MplPolygon(corners_world, closed=True,
                          facecolor='#e8e8e8', edgecolor='#444444',
                          linewidth=1.8, zorder=1)
        ax.add_patch(poly)

    # ── A* reference path ────────────────────────────────────────────────
    if hasattr(maze, 'astar_pts_world'):
        apts = maze.astar_pts_world
        ax.plot(apts[:, 0], apts[:, 1], '--',
                color='#ccaa00', linewidth=1.5, alpha=0.75,
                label='A* reference path', zorder=2)

    # ── Trajectories ─────────────────────────────────────────────────────
    ax.plot(qpos_on[:, 0], qpos_on[:, 1],
            color='#00bb44', linewidth=2.5, label='Sensor ON', zorder=3)
    ax.plot(qpos_off[:, 0], qpos_off[:, 1],
            color='#dd2222', linewidth=2.5, label='Sensor OFF', alpha=0.85, zorder=3)

    # Arrow at the end of each trajectory to show direction of travel
    for xy, color in [(qpos_on, '#00bb44'), (qpos_off, '#dd2222')]:
        if len(xy) >= 2:
            dx = xy[-1, 0] - xy[-2, 0]
            dy = xy[-1, 1] - xy[-2, 1]
            ax.annotate('', xy=xy[-1, :2], xytext=xy[-2, :2],
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                        zorder=4)

    # ── Start / Goal markers ─────────────────────────────────────────────
    start = maze.waypoints[0]
    goal  = maze.waypoints[-1]
    ax.plot(*start, 'o', color='royalblue',  markersize=12, zorder=5, label='Start')
    ax.plot(*goal,  '*', color='darkorange', markersize=16, zorder=5, label='Goal')

    # ── Frame length annotation ───────────────────────────────────────────
    ax.text(0.02, 0.97,
            f"Sensor ON:  {len(qpos_on)} frames\nSensor OFF: {len(qpos_off)} frames",
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # ── Styling ───────────────────────────────────────────────────────────
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Maze Navigation – Sensor ON vs OFF', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.grid(True, alpha=0.25, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Compare] Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Maze demo – sensor-conditioned avoidance")
    p.add_argument("--dataset",       default="lafan1_g1_motion28")
    p.add_argument("--checkpoint",    default="save/camdm_g1_env/best.pt")
    p.add_argument("--motion",        type=int,   default=0,
                   help="Motion clip index from dataset (for initial pose & style)")
    p.add_argument("--room-w",        type=float, default=4.0,
                   help="Room width in the forward (x) direction, metres")
    p.add_argument("--room-h",        type=float, default=4.0,
                   help="Room height in the lateral (y) direction, metres")
    p.add_argument("--corridor-len",  type=float, default=1.5,
                   help="Corridor length (x direction) in metres")
    p.add_argument("--corridor-hw",   type=float, default=0.45,
                   help="Corridor half-width (y direction) in metres (default 0.45 → 0.9 m)")
    p.add_argument("--corridor-y",    type=float, default=1.0,
                   help="Y-offset of corridor centre from room centre, metres")
    p.add_argument("--speed",         type=float, default=0.7,
                   help="Path-following speed in m/s: controls how fast the red-arrow waypoints advance. "
                        "Increase (e.g. 1.0–1.5) to pull the robot faster along the corridor.")
    p.add_argument("--resolution",    type=int,   default=9)
    p.add_argument("--max-range",     type=float, default=2.0)
    p.add_argument("--past-frames",   type=int,   default=10)
    p.add_argument("--future-frames", type=int,   default=45)
    p.add_argument("--no-blend",        action="store_true",
                   help="Disable trajectory blending; feed raw waypoint path directly to the model")
    p.add_argument("--guidance-scale",  type=float, default=0.0,
                   help="SDF obstacle-avoidance guidance strength (0 = off). "
                        "Adds ∇_x(−E) to each denoising step where E is the "
                        "hinge-loss penetration energy. Try 1–10.")
    p.add_argument("--guidance-margin", type=float, default=0.3,
                   help="Minimum clearance in metres used by SDF guidance (default 0.3)")
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
    p.add_argument("--compare", action="store_true",
                   help="Run sensor-ON then sensor-OFF headlessly and save a side-by-side "
                        "comparison video (no interactive window)")
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

    # ── Build maze so that robot's initial position = first waypoint ──────────
    motion_idx = args.motion % len(dataset)
    init_qpos  = dataset[motion_idx].get_qpos(0)
    init_xy    = init_qpos[:2].copy()
    init_yaw   = quat_wxyz_to_yaw(init_qpos[3:7])

    # start_local: where the first waypoint sits in the maze's local frame
    # = right-side centre of the right room (room_w * START_FRAC, 0)
    c_y, s_y = np.cos(init_yaw), np.sin(init_yaw)
    R_mat    = np.array([[c_y, -s_y], [s_y, c_y]])
    start_local = np.array([args.room_w * MazeLayout.START_FRAC, 0.0])
    maze_origin = init_xy - R_mat @ start_local   # local (0,0) in world

    maze = MazeLayout(
        origin=maze_origin,
        initial_yaw=init_yaw,
        room_w=args.room_w,
        room_h=args.room_h,
        corridor_len=args.corridor_len,
        corridor_hw=args.corridor_hw,
        corridor_y=args.corridor_y,
    )

    # ── Comparison mode: run both passes headlessly and exit ──────────────
    if args.compare:
        run_comparison(args, mj_model, dataset, generator, maze)
        return

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
        guidance_scale=args.guidance_scale,
        guidance_margin=args.guidance_margin,
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
