"""
Detour obstacle placement utilities
=====================================
When the actual root trajectory (red) takes a detour from the linear
command trajectory (green), circular obstacles are placed in the gap so
the policy must navigate around them.

Two circles are placed per window:
  Circle 1 – off-path center
      Seed: green point farthest from red arrow points.
      Center moved away from red to maximise radius while the circle
      still overlaps the green (command) line.

  Circle 2 – on-path center
      Center fixed on the green line at the point with maximum
      clearance from red.

Overlap between the two circles is not checked.
All obstacles are discarded when every circle is smaller than
robot_safe_radius (detour too shallow to be useful).
"""

from typing import List, Tuple

import numpy as np
from utils.environment_sensor import CircleObstacle, EnvironmentSensor
from visualize.utils.trajectory import extend_future_traj_heusristic

# Every 5th frame – matches draw_trajectory_arrows in geometry.py
ARROW_STEP = 5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polyline_min_dist(center: np.ndarray,
                      P1: np.ndarray,
                      seg_d: np.ndarray,
                      seg_d_sq: np.ndarray) -> float:
    """Minimum distance from *center* to a polyline (segment-based)."""
    t = np.sum((center - P1) * seg_d, axis=1) / (seg_d_sq + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return float(np.linalg.norm(center - (P1 + t[:, None] * seg_d), axis=1).min())


def batch_polyline_dist(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """
    Vectorised distance from each point in *points* (M, 2) to the nearest
    segment of *polyline* (N, 2). Returns (M,) distances.
    """
    if len(polyline) < 2:
        diffs = points[:, None, :] - polyline[None, :, :]
        return np.linalg.norm(diffs, axis=-1).min(axis=1)

    P0    = polyline[:-1]                                 # (N-1, 2)
    seg_d = polyline[1:] - P0                             # (N-1, 2)
    seg_l = (seg_d * seg_d).sum(axis=1) + 1e-12           # (N-1,)

    rel  = points[:, None, :] - P0[None, :, :]            # (M, N-1, 2)
    t    = (rel * seg_d[None, :, :]).sum(axis=2) / seg_l[None, :]
    t    = np.clip(t, 0.0, 1.0)                           # (M, N-1)
    proj = P0[None, :, :] + t[:, :, None] * seg_d[None, :, :]
    return np.linalg.norm(points[:, None, :] - proj, axis=2).min(axis=1)


def point_to_seg_dist(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point *c* to line segment *a*–*b*."""
    ab = b - a
    t  = np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-12)
    return float(np.linalg.norm(c - (a + np.clip(t, 0.0, 1.0) * ab)))


# ---------------------------------------------------------------------------
# Circle finders
# ---------------------------------------------------------------------------

def find_offpath_circle(seed: np.ndarray,
                        green_start: np.ndarray, green_end: np.ndarray,
                        P1: np.ndarray, seg_d: np.ndarray, seg_d_sq: np.ndarray,
                        robot_safe_radius: float,
                        n_steps: int = 60, max_dist: float = 2.5):
    """
    Circle 1 – off-path center.

    Starts at *seed* (a green-line point) and steps away from the nearest
    red segment to find the center that maximises obstacle radius while the
    circle still overlaps the green (command) line.

    Returns (center, radius) or (None, 0.0) when no valid position is found.
    """
    t = np.sum((seed - P1) * seg_d, axis=1) / (seg_d_sq + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest_on_red = P1 + t[:, None] * seg_d
    nearest_red    = closest_on_red[np.linalg.norm(seed - closest_on_red, axis=1).argmin()]
    away = seed - nearest_red
    norm = np.linalg.norm(away)
    if norm < 1e-6:
        return None, 0.0
    away /= norm

    best_r, best_c = 0.0, None
    for d in np.linspace(0.0, max_dist, n_steps):
        c   = seed + away * d
        eff = polyline_min_dist(c, P1, seg_d, seg_d_sq) - robot_safe_radius
        if eff <= 0:
            continue
        r = eff * 0.90
        if r < 0.05:
            continue
        if point_to_seg_dist(c, green_start, green_end) > r:
            continue
        if r > best_r:
            best_r, best_c = r, c.copy()

    return best_c, best_r


def find_onpath_circle(green_xy: np.ndarray,
                       P1: np.ndarray, seg_d: np.ndarray, seg_d_sq: np.ndarray,
                       robot_safe_radius: float, margin: int):
    """
    Circle 2 – on-path center.

    Scans every non-margin green-line point and returns the one that yields
    the largest safe radius (clearance from red minus robot_safe_radius).

    Returns (center, radius) or (None, 0.0) when no valid point is found.
    """
    N = len(green_xy)
    best_r, best_c = 0.0, None
    for i in range(margin, N - margin):
        c   = green_xy[i]
        eff = polyline_min_dist(c, P1, seg_d, seg_d_sq) - robot_safe_radius
        if eff <= 0:
            continue
        r = eff * 0.90
        if r >= 0.05 and r > best_r:
            best_r, best_c = r, c.copy()
    return best_c, best_r


# ---------------------------------------------------------------------------
# Command trajectory builder
# ---------------------------------------------------------------------------

def make_command_xy(actual_xy: np.ndarray,
                    cmd_aug: str = "linear",
                    past_xy: np.ndarray = None,
                    weight: float = 1.0) -> np.ndarray:
    """
    Build the command (green) trajectory used for detour-obstacle placement.

    Parameters
    ----------
    actual_xy : (N, 2) actual robot positions across the obstacle window.
    cmd_aug   : "linear"      → straight line actual_xy[0] → actual_xy[-1].
                "extrap_pos"  → constant-velocity extrapolation of past trend.
                "extrap_pos_noyaw" → identical XY to extrap_pos (yaw handling
                                differs only in the visualiser).
                "extrap_pos_preserve_len" → straight line in past direction,
                                but per-frame step lengths are taken from
                                the actual trajectory (so total path length
                                matches gt).
                "extrap_hfte" → HFTE (heuristic central-symmetry) extension
                                seeded with past + current frame. See detour.md.
    past_xy   : (P, 2) past positions (oldest first). Required for extrap_pos*,
                extrap_hfte; falls back to "linear" when too few frames.
    weight    : in [0, 1]. Linear blend
                ``w * extrap + (1 - w) * actual``. 1.0 = pure extrap (default),
                0.0 = pure ground truth.

    Returns
    -------
    command_xy : (N, 2) command trajectory starting at actual_xy[0].
    """
    actual_xy = np.asarray(actual_xy, dtype=np.float64)
    N = len(actual_xy)

    if cmd_aug in ("extrap_pos", "extrap_pos_noyaw") \
            and past_xy is not None and len(past_xy) >= 2:
        past_xy = np.asarray(past_xy, dtype=np.float64)
        v = (past_xy[-1] - past_xy[0]) / (len(past_xy) - 1)   # per-frame velocity
        steps = np.arange(N)
        extrap = actual_xy[0][None] + steps[:, None] * v[None]
    elif cmd_aug == "extrap_pos_preserve_len" and past_xy is not None and len(past_xy) >= 2:
        past_xy = np.asarray(past_xy, dtype=np.float64)
        v = (past_xy[-1] - past_xy[0]) / (len(past_xy) - 1)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-9:
            # Stationary past — fall through to linear
            t = np.linspace(0.0, 1.0, N)
            extrap = actual_xy[0][None] + t[:, None] * (actual_xy[-1] - actual_xy[0])[None]
        else:
            direction = v / v_norm                                       # (2,)
            seg_lens  = np.linalg.norm(np.diff(actual_xy, axis=0), axis=1)
            arc       = np.concatenate([[0.0], np.cumsum(seg_lens)])     # (N,)
            extrap    = actual_xy[0][None] + arc[:, None] * direction[None]
    elif cmd_aug == "extrap_hfte" and past_xy is not None and len(past_xy) >= 1:
        past_xy = np.asarray(past_xy, dtype=np.float64)
        seed = np.vstack([past_xy, actual_xy[0][None]])       # (P+1, 2)
        total = len(seed) + (N - 1)
        dummy_orient = np.tile([1.0, 0.0, 0.0, 0.0], (len(seed), 1))
        ext_traj, _ = extend_future_traj_heusristic(seed, dummy_orient, total)
        extrap = ext_traj[-N:]                                # starts at a_0
    else:
        # Default / fallback: linear interpolation
        t = np.linspace(0.0, 1.0, N)
        extrap = actual_xy[0][None] + t[:, None] * (actual_xy[-1] - actual_xy[0])[None]

    if weight == 1.0:
        return extrap
    return weight * extrap + (1.0 - weight) * actual_xy


def make_command_yaws(actual_yaws: np.ndarray,
                     cmd_aug: str = "linear",
                     past_xy: np.ndarray = None,
                     past_yaws: np.ndarray = None,
                     current_yaw: float = None,
                     weight: float = 1.0) -> np.ndarray:
    """
    Build the command (yellow) yaw sequence over the future horizon,
    mirroring the modes in :func:`make_command_xy`.

    Parameters
    ----------
    actual_yaws : (N,) future yaws (gt) in radians; the linear fallback
                  and the gt side of the blend.
    cmd_aug     : ``"linear"`` / ``"extrap_pos"`` / ``"extrap_pos_noyaw"``
                  / ``"extrap_pos_preserve_len"`` / ``"extrap_hfte"``.
    past_xy     : (P, 2) past XY (only used by ``extrap_pos_noyaw`` to
                  recover the past velocity direction).
    past_yaws   : (P,) past yaws (used by ``extrap_pos`` /
                  ``extrap_pos_preserve_len`` for the constant rate, and
                  by ``extrap_hfte`` for the HFTE seed).
    current_yaw : "now" yaw (used by ``extrap_pos_noyaw`` and
                  ``extrap_hfte``). Defaults to ``actual_yaws[0]``.
    weight      : blend ``w * extrap + (1 - w) * actual_yaws`` along the
                  shortest signed yaw path. ``1.0`` = pure extrap
                  (default), ``0.0`` = pure gt.

    Returns
    -------
    yaws : (N,) yaw values in radians. Wrapping is *not* applied — feed
           through ``cos(y/2), sin(y/2)`` for quaternion conversion if
           you need it back in [-π, π].

    Notes
    -----
    Whenever the requested mode lacks the inputs it needs (e.g. fewer
    than two past frames for ``extrap_pos``), the function silently
    falls back to the linear behaviour: it returns ``actual_yaws`` as is.
    """
    actual_yaws = np.asarray(actual_yaws, dtype=np.float64)
    N = len(actual_yaws)

    if current_yaw is None and N >= 1:
        current_yaw = float(actual_yaws[0])

    extrap = None

    if cmd_aug in ("extrap_pos", "extrap_pos_preserve_len") \
            and past_yaws is not None and len(past_yaws) >= 2:
        past_yaws_u = np.unwrap(np.asarray(past_yaws, dtype=np.float64))
        rate = (past_yaws_u[-1] - past_yaws_u[0]) / (len(past_yaws_u) - 1)
        extrap = past_yaws_u[-1] + np.arange(1, N + 1) * rate

    elif cmd_aug == "extrap_pos_noyaw" \
            and past_xy is not None and current_yaw is not None \
            and len(past_xy) >= 2:
        past_xy_arr = np.asarray(past_xy, dtype=np.float64)
        v = (past_xy_arr[-1] - past_xy_arr[0]) / (len(past_xy_arr) - 1)
        yaw_target = float(np.arctan2(v[1], v[0]))
        delta = (yaw_target - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
        ts = np.linspace(0.0, 1.0, N)
        extrap = current_yaw + ts * delta

    elif cmd_aug == "extrap_hfte" \
            and past_yaws is not None and current_yaw is not None \
            and len(past_yaws) >= 1:
        past_yaws_arr = np.asarray(past_yaws, dtype=np.float64)
        seed_yaws = np.unwrap(np.concatenate([past_yaws_arr, [current_yaw]]))
        seed_2d   = np.stack([seed_yaws, np.zeros_like(seed_yaws)], axis=1)
        total     = len(seed_2d) + (N - 1)
        dummy     = np.tile([1.0, 0.0, 0.0, 0.0], (len(seed_2d), 1))
        ext_2d, _ = extend_future_traj_heusristic(seed_2d, dummy, total)
        extrap    = ext_2d[-N:, 0]

    if extrap is None:
        return actual_yaws.copy()    # linear fallback (no augmentation)

    if weight >= 1.0:
        return extrap
    if weight <= 0.0:
        return actual_yaws.copy()

    delta = (extrap - actual_yaws + np.pi) % (2.0 * np.pi) - np.pi
    delta = np.unwrap(delta)
    return actual_yaws + weight * delta


# ---------------------------------------------------------------------------
# Scandot-fill obstacle augmentation
# ---------------------------------------------------------------------------

# Minimum max-pointwise yellow↔red deviation (m) below which we declare the
# window has no detour and return no obstacles.
_DETOUR_DEVIATION_MIN = 0.10


def _make_yellow_circle_obstacles(
    command_xy: np.ndarray,
    actual_xy: np.ndarray,
    safe_distance: float,
    min_radius: float = 0.05,
) -> Tuple[List[CircleObstacle], float]:
    """
    Place one ``CircleObstacle`` at every yellow (command) arrow point.
    Each circle gets its own *largest* radius such that it stays at least
    ``safe_distance`` away from the red (actual) polyline.

    Per-point radius (uses the polyline distance to red so a circle never
    reaches into red anywhere on the future horizon):

        r_i = max( d(yellow_i, red_polyline) − safe_distance, 0 )

    Yellow points whose r_i falls below ``min_radius`` are dropped — this
    naturally drops the boundary points where yellow coincides with red
    (linear / extrap_pos* trajectories share endpoints with red, giving
    polyline distance 0). If *no* point is left, no obstacles are
    produced at all.

    Note: very oscillatory motion (e.g. running with hip swing) brings
    red close to the yellow chord even when the trajectory clearly
    deviates, so the polyline metric will report tiny distances and few
    obstacles will be placed. That's the price of guaranteeing no
    obstacle ever overlaps red.

    Returns ``(obstacles, mean_radius)``. ``mean_radius`` is 0.0 when
    no obstacles can be placed.
    """
    if len(command_xy) < 1:
        return [], 0.0

    d_red_per_yellow = batch_polyline_dist(command_xy, actual_xy)
    radii_per_yellow = d_red_per_yellow - safe_distance

    valid = radii_per_yellow >= min_radius
    if not valid.any():
        return [], 0.0

    obstacles = [CircleObstacle(p.copy(), float(r))
                 for p, r, ok in zip(command_xy, radii_per_yellow, valid) if ok]
    mean_r = float(radii_per_yellow[valid].mean())
    return obstacles, mean_r


def make_scandot_fill_obstacles(
    sensor: EnvironmentSensor,
    robot_pos: np.ndarray,
    robot_yaw: float,
    command_xy: np.ndarray,
    actual_xy: np.ndarray,
    past_xy: np.ndarray = None,
    robot_safe_radius: float = 0.25,
    obstacle_radius: float = None,
    deviation_min: float = _DETOUR_DEVIATION_MIN,
    fill_method: str = "band",
) -> Tuple[List[CircleObstacle], dict]:
    """
    Per-step augmentation: fill sensor scandots between the command (yellow)
    and actual (red) trajectories, while keeping the past (blue) trajectory
    clear. Each filled scandot becomes a CircleObstacle so the sensor reads
    occupancy ≈ 1 there.

    Two fill strategies (selected by ``fill_method``):

      "band" (default) — scandot-based, band around yellow:
          keep scandot ⇔  d(s, actual) ≥ r_safe
          (the historical d(s, command) < r_safe inclusion was disabled
          experimentally; see the comment in the body to re-enable.)

      "yellow_circle" — one CircleObstacle per yellow arrow point:
          obstacles are not placed at scandots; instead, every yellow
          (command) point hosts a CircleObstacle. Each circle gets its
          own largest radius such that it stays at least ``r_safe`` away
          from the red polyline:
              r_i = max(d(yellow_i, red) − r_safe, 0)
          Points where r_i falls below 0.05 m are skipped; this naturally
          drops boundary points where yellow coincides with red (the
          linear / extrap_pos* trajectories share endpoints with red so
          the polyline distance there is 0). If every point fails the
          test, no obstacles are produced.

    Past-trajectory exclusion (centre within r_safe of past_xy) is applied
    to both methods when ``past_xy`` is provided.

    Parameters
    ----------
    sensor            : EnvironmentSensor used to obtain scandot positions
                        (and ``max_adjacent_distance`` for the obstacle radius).
    robot_pos, robot_yaw : current robot pose; scandots are placed relative
                        to this pose (window = 1 frame).
    command_xy, actual_xy : (N, 2) yellow / red trajectories over the
                        future horizon.
    past_xy           : optional (P, 2) past trajectory (blue). When provided,
                        scandots within ``robot_safe_radius`` of any point on
                        the past polyline are also excluded so obstacles never
                        appear where the robot has just been.
    robot_safe_radius : threshold for the "close to yellow" inclusion and the
                        "safe distance from red / past" exclusions.
    obstacle_radius   : per-scandot CircleObstacle radius. Defaults to half
                        the sensor's max nearest-neighbour distance, which
                        is the smallest value that still tiles a fully-filled
                        region with no gaps.
    deviation_min     : min max-pointwise(yellow, red) below which the
                        window is treated as no-detour and no obstacles
                        are produced.
    fill_method       : "band" or "ring_arc" (see above).

    Returns
    -------
    obstacles : list[CircleObstacle]
    info      : dict for visualisation:
        all_centers              : (N_dots, 2) every scandot's world XY
        fill_mask                : (N_dots,) bool, True = filled
        d_yellow, d_red, d_past  : (N_dots,) per-scandot polyline distances
                                   (d_past is None when no past_xy was given)
        obstacle_radius          : the radius used
        max_pointwise            : max |yellow_i − red_i| over the horizon
        has_detour               : bool, True iff at least one obstacle was
                                   actually produced (use this as the
                                   per-frame ``command_detour_flags`` flag).
        fill_method              : echoed back for downstream debugging
    """
    command_xy = np.asarray(command_xy, dtype=np.float64)
    actual_xy  = np.asarray(actual_xy,  dtype=np.float64)

    if obstacle_radius is None:
        obstacle_radius = 0.5 * sensor.max_adjacent_distance

    max_pw = float(np.linalg.norm(command_xy - actual_xy, axis=1).max())

    # Get scandot world positions (occupancy is irrelevant here)
    _, centers = sensor.compute(robot_pos, robot_yaw, [])

    info = {
        "all_centers":     centers,
        "fill_mask":       np.zeros(len(centers), dtype=bool),
        "d_yellow":        None,
        "d_red":           None,
        "d_past":          None,
        "obstacle_radius": obstacle_radius,
        "max_pointwise":   max_pw,
        "has_detour":      False,    # set to len(obstacles) > 0 at the end
        "fill_method":     fill_method,
    }
    # Cheap deviation-based early-out: if yellow barely deviates from red
    # there's nothing to augment.
    if max_pw < deviation_min or len(command_xy) < 2 or len(actual_xy) < 2:
        info["reason"] = ("no detour" if max_pw < deviation_min
                          else "trajectory too short")
        return [], info

    d_yellow = batch_polyline_dist(centers, command_xy)
    d_red    = batch_polyline_dist(centers, actual_xy)
    info["d_yellow"] = d_yellow
    info["d_red"]    = d_red

    if past_xy is not None and len(past_xy) >= 1:
        past_xy_arr = np.asarray(past_xy, dtype=np.float64)
        info["d_past"] = batch_polyline_dist(centers, past_xy_arr)
    else:
        past_xy_arr = None

    if fill_method == "yellow_circle":
        # Place a circle AT each interior yellow point. Each circle's
        # radius is the largest value that stays r_safe away from red.
        # If even one interior point can't host a non-trivial circle the
        # augmentation falls back to the existing min-radius drop logic
        # (point-wise, not whole-augmentation).
        obstacles, mean_r = _make_yellow_circle_obstacles(
            command_xy, actual_xy, robot_safe_radius
        )
        info["obstacle_radius"] = mean_r
        # Past exclusion: drop circles whose extent enters the past
        # safety zone, i.e. d(centre, past) − radius < r_safe.
        if obstacles and past_xy_arr is not None:
            centers_arr = np.array([o.center for o in obstacles])
            radii_arr   = np.array([o.radius for o in obstacles])
            d_past_obs  = batch_polyline_dist(centers_arr, past_xy_arr)
            obstacles = [o for o, d, r in zip(obstacles, d_past_obs, radii_arr)
                         if d - r >= robot_safe_radius]
        info["has_detour"] = len(obstacles) > 0
        return obstacles, info

    # Scandot-based methods: each filled scandot becomes one obstacle
    # of radius `obstacle_radius`.
    if fill_method == "band":
        band_center = robot_pos
        fill_mask = (d_red >= robot_safe_radius)
        fill_mask &= (np.linalg.norm(centers-band_center, axis=1) < np.maximum(
            np.max(np.linalg.norm(command_xy-band_center, axis=1)), 
            np.max(np.linalg.norm(actual_xy-band_center, axis=1))
        ))
    elif fill_method == "band_yellow":
        band_center = command_xy.mean(axis=0)
        fill_mask = (d_red >= robot_safe_radius)
        fill_mask &= (np.linalg.norm(centers-band_center, axis=1) < np.maximum(
            np.max(np.linalg.norm(command_xy-band_center, axis=1)), 
            np.max(np.linalg.norm(actual_xy-band_center, axis=1))
        ))
    else:
        raise ValueError(f"Unknown fill_method: {fill_method!r}")

    if past_xy_arr is not None:
        fill_mask &= (info["d_past"] >= robot_safe_radius)

    info["fill_mask"] = fill_mask
    obstacles = [CircleObstacle(centers[i].copy(), obstacle_radius)
                 for i in np.where(fill_mask)[0]]
    info["has_detour"] = len(obstacles) > 0
    return obstacles, info


# ---------------------------------------------------------------------------
# Legacy 2-circle detour (kept for now; will be removed once scandot-fill
# is plumbed everywhere)
# ---------------------------------------------------------------------------

def compute_detour_obstacles(path_xy: np.ndarray,
                             command_xy: np.ndarray = None,
                             robot_safe_radius: float = 0.25):
    """
    Place up to two detour obstacles for one trajectory window.

    Parameters
    ----------
    path_xy : (N, 2) float array
        Root XY positions for the window (red / actual trajectory).
    command_xy : (N, 2) optional
        Command (green) trajectory the policy is told to follow. Must be the
        same length as path_xy. When None, falls back to linear interpolation
        from path_xy[0] to path_xy[-1] (legacy behaviour).
    robot_safe_radius : float
        Body clearance subtracted from raw segment distance when
        computing obstacle radius.

    Returns
    -------
    obstacles : list[CircleObstacle]
    info      : dict  (debug / visualisation data)
    """
    N = len(path_xy)
    if N < 4:
        return [], {"reason": "N<4"}

    if command_xy is None:
        t_param  = np.linspace(0.0, 1.0, N)
        green_xy = path_xy[0][None] + t_param[:, None] * (path_xy[-1] - path_xy[0])[None]
    else:
        green_xy = np.asarray(command_xy, dtype=np.float64)
        assert len(green_xy) == N, \
            f"command_xy length {len(green_xy)} != path_xy length {N}"
    green_start = green_xy[0].copy()
    green_end   = green_xy[-1].copy()

    pointwise = np.linalg.norm(green_xy - path_xy, axis=1)
    if float(pointwise.max()) < 0.10:
        return [], {"reason": f"max_pointwise={pointwise.max():.4f} < 0.10"}

    P1       = path_xy[:-1]
    seg_d    = path_xy[1:] - P1
    seg_d_sq = (seg_d * seg_d).sum(axis=1)

    red_arrow_xy = path_xy[np.arange(0, N, ARROW_STEP)]
    margin       = max(2, N // 4)

    obstacles    = []
    seed_indices = []   # None entry = no single seed point (circle 2)

    # --- Circle 1: off-path ---
    # da[i] = distance from green_xy[i] to the nearest red arrow point
    da = np.linalg.norm(
        green_xy[:, None, :] - red_arrow_xy[None, :, :], axis=2
    ).min(axis=1)
    da[:margin]   = 0.0
    da[N-margin:] = 0.0

    if float(da.max()) >= 0.05:
        seed_idx       = int(da.argmax())
        seed           = green_xy[seed_idx].copy()
        center, radius = find_offpath_circle(
            seed, green_start, green_end,
            P1, seg_d, seg_d_sq, robot_safe_radius,
        )
        if center is not None and radius >= 0.05:
            obstacles.append(CircleObstacle(center, radius))
            seed_indices.append(seed_idx)

    # --- Circle 2: on-path ---
    center, radius = find_onpath_circle(
        green_xy, P1, seg_d, seg_d_sq, robot_safe_radius, margin
    )
    if center is not None and radius >= 0.05:
        obstacles.append(CircleObstacle(center, radius))
        seed_indices.append(None)

    # Discard all if every circle is smaller than robot_safe_radius
    if obstacles and all(o.radius < robot_safe_radius for o in obstacles):
        obstacles    = []
        seed_indices = []

    info = {
        "N":                 N,
        "max_pointwise":     float(pointwise.max()),
        "green_xy":          green_xy,
        "red_arrow_xy":      red_arrow_xy,
        "path_xy":           path_xy,
        "margin":            margin,
        "robot_safe_radius": robot_safe_radius,
        "seed_indices":      seed_indices,
        "obstacles":         obstacles,
    }
    if not obstacles:
        info["reason"] = "no valid placement found"
    return obstacles, info
