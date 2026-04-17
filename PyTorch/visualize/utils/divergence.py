"""
Divergence obstacle placement utilities
========================================
Shared logic for detecting trajectory divergence and placing circular
obstacles in the gap between the linear command (green) and actual (red)
root trajectories.

Two circles are placed per window:
  Circle 1 – off-green-line center
      Seed: green point farthest from red arrow points.
      Center moved away from red to maximise radius while the circle
      still overlaps the green line.

  Circle 2 – on-green-line center
      Center fixed on the green line at the point with maximum
      clearance from red.

Overlap between the two circles is not checked.
All obstacles are discarded when every circle is smaller than
robot_safe_radius (trajectory too close to green to be useful).
"""

import numpy as np
from utils.environment_sensor import CircleObstacle

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


def point_to_seg_dist(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point *c* to line segment *a*–*b*."""
    ab = b - a
    t  = np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-12)
    return float(np.linalg.norm(c - (a + np.clip(t, 0.0, 1.0) * ab)))


# ---------------------------------------------------------------------------
# Obstacle placement
# ---------------------------------------------------------------------------

def optimize_center(seed: np.ndarray,
                    green_start: np.ndarray, green_end: np.ndarray,
                    P1: np.ndarray, seg_d: np.ndarray, seg_d_sq: np.ndarray,
                    robot_safe_radius: float,
                    n_steps: int = 60, max_dist: float = 2.5):
    """
    Circle 1 – off-green-line center.

    Starts at *seed* (a green-line point) and steps away from the nearest
    red segment to find the center that maximises obstacle radius while the
    circle still overlaps the green line.

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


def green_line_center(green_xy: np.ndarray,
                      P1: np.ndarray, seg_d: np.ndarray, seg_d_sq: np.ndarray,
                      robot_safe_radius: float, margin: int):
    """
    Circle 2 – on-green-line center.

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


def compute_divergence_obstacles(path_xy: np.ndarray, robot_safe_radius: float = 0.25):
    """
    Place up to two divergence obstacles for one trajectory window.

    Parameters
    ----------
    path_xy : (N, 2) float array
        Root XY positions for the window (red trajectory).
    robot_safe_radius : float
        Body clearance; subtracted from raw segment distance when
        computing obstacle radius.

    Returns
    -------
    obstacles : list[CircleObstacle]
    info      : dict  (debug / visualisation data)
    """
    N = len(path_xy)
    if N < 4:
        return [], {"reason": "N<4"}

    t_param     = np.linspace(0.0, 1.0, N)
    green_start = path_xy[0].copy()
    green_end   = path_xy[-1].copy()
    green_xy    = green_start + t_param[:, None] * (green_end - green_start)

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

    # --- Circle 1: off-green-line ---
    da = np.linalg.norm(
        green_xy[:, None, :] - red_arrow_xy[None, :, :], axis=2
    ).min(axis=1)
    da[:margin]   = 0.0
    da[N-margin:] = 0.0

    if float(da.max()) >= 0.05:
        seed_idx       = int(da.argmax())
        seed           = green_xy[seed_idx].copy()
        center, radius = optimize_center(
            seed, green_start, green_end,
            P1, seg_d, seg_d_sq, robot_safe_radius,
        )
        if center is not None and radius >= 0.05:
            obstacles.append(CircleObstacle(center, radius))
            seed_indices.append(seed_idx)

    # --- Circle 2: on-green-line ---
    center, radius = green_line_center(
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
