"""
Step 2 – 2-D Trajectory & Obstacle Debug Visualiser
=====================================================
Renders a top-down (XY) view of every obstacle-generation window for a given
motion clip and saves a PNG per window.

Usage
-----
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23 --motion 2 --all-windows
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23 --out-dir debug_imgs

Legend
------
  Light coral line  : full clip root trajectory (background reference)
  Green line        : linear command trajectory (A → B) for the window
  Green × markers  : arrow sample points (every 5th frame)
  Green shading     : robot safe-radius tube around green trajectory
  Red line          : actual root trajectory for the window
  Red × markers    : red arrow points used for seed selection
  Red shading       : robot safe-radius tube around red trajectory
  Grey dots         : excluded margin at start / end of window
  Orange circle     : obstacle #1  (off-green-line, max-radius)
  Blue circle       : obstacle #2  (on-green-line center)
  ★ markers        : seed points used for circle 1
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset
from utils.environment_sensor import CircleObstacle

# Every 5th frame matches draw_trajectory_arrows in geometry.py
ARROW_STEP = 5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polyline_min_dist(center: np.ndarray,
                      P1: np.ndarray, seg_d: np.ndarray, seg_d_sq: np.ndarray) -> float:
    """Minimum distance from *center* to a polyline given as pre-computed segments."""
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
    # Compute the direction away from the nearest red segment point
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
        # Circle must still reach (overlap) the green line
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

    Circle 1 – off-green-line: seed is the green point farthest from red
               arrow points; center is then optimised away from red.
    Circle 2 – on-green-line: center fixed on the green line at the point
               with maximum clearance from red.

    Overlap between the two circles is not checked.
    All obstacles are discarded when every circle is smaller than
    *robot_safe_radius* (trajectory too close to green to be useful).

    Returns
    -------
    obstacles : list[CircleObstacle]
    info      : dict  (debug data for plotting)
    """
    N = len(path_xy)
    if N < 4:
        return [], {"reason": "N<4"}

    # Build green (linear command) trajectory
    t_param     = np.linspace(0.0, 1.0, N)
    green_start = path_xy[0].copy()
    green_end   = path_xy[-1].copy()
    green_xy    = green_start + t_param[:, None] * (green_end - green_start)

    # Skip windows where red is nearly identical to green
    pointwise = np.linalg.norm(green_xy - path_xy, axis=1)
    if float(pointwise.max()) < 0.10:
        return [], {"reason": f"max_pointwise={pointwise.max():.4f} < 0.10"}

    # Pre-compute red polyline segments (reused across calls)
    P1       = path_xy[:-1]
    seg_d    = path_xy[1:] - P1
    seg_d_sq = (seg_d * seg_d).sum(axis=1)

    red_arrow_xy = path_xy[np.arange(0, N, ARROW_STEP)]
    margin       = max(2, N // 4)   # exclude shared start / end region

    obstacles    = []
    seed_indices = []   # for debug plotting; -1 means "no single seed point"

    # --- Circle 1: off-green-line ---
    # da[i] = how far green_xy[i] is from the nearest red arrow point
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
        seed_indices.append(None)   # no single seed point for on-line search

    # Discard everything when all circles are too small to be meaningful
    if obstacles and all(o.radius < robot_safe_radius for o in obstacles):
        obstacles    = []
        seed_indices = []

    info = {
        "N":              N,
        "max_pointwise":  float(pointwise.max()),
        "green_xy":       green_xy,
        "red_arrow_xy":   red_arrow_xy,
        "path_xy":        path_xy,
        "margin":         margin,
        "robot_safe_radius": robot_safe_radius,
        "seed_indices":   seed_indices,
        "obstacles":      obstacles,
    }
    if not obstacles:
        info["reason"] = "no valid placement found"
    return obstacles, info


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_safe_tube(ax, xy, radius, color, alpha=0.15, label=None, step=2):
    """Shade a robot-safe-radius band around a polyline using overlapping circles."""
    for i, pt in enumerate(xy[::step]):
        circ = plt.Circle(pt, radius, color=color, alpha=alpha, zorder=1,
                          label=label if i == 0 else None)
        ax.add_patch(circ)


def plot_window(motion_idx: int, win_idx: int,
                full_path_xy: np.ndarray, div_xy: np.ndarray,
                obstacles: list, info: dict,
                out_dir: str, w_start: int, w_end: int) -> str:
    """Render one window to a PNG and return the file path."""
    safe_r = info.get("robot_safe_radius", 0.25)
    colors = ["orange", "royalblue", "mediumpurple", "forestgreen"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title(
        f"motion={motion_idx}  window={win_idx}  frames=[{w_start},{w_end})\n"
        f"N_div={info.get('N', len(div_xy))}  "
        f"max_pw={info.get('max_pointwise', 0):.3f}m  "
        f"safe_r={safe_r}m  "
        + (f"obstacles={len(obstacles)}" if obstacles
           else f"NO OBS ({info.get('reason', '')})")
    )

    # Background: full clip
    ax.plot(full_path_xy[:, 0], full_path_xy[:, 1],
            color="lightcoral", lw=0.8, alpha=0.3, label="full clip")

    # Green trajectory + safe tube
    if "green_xy" in info:
        green_xy = info["green_xy"]
        _draw_safe_tube(ax, green_xy, safe_r, color="limegreen", alpha=0.15,
                        label=f"green safe (r={safe_r}m)")
        ax.plot(green_xy[:, 0], green_xy[:, 1],
                color="green", lw=1.8, zorder=3, label="green (linear cmd)")
        arrow_idx = np.arange(0, len(green_xy), ARROW_STEP)
        ax.scatter(green_xy[arrow_idx, 0], green_xy[arrow_idx, 1],
                   marker="x", s=40, color="green", zorder=5)

    # Red trajectory + safe tube
    _draw_safe_tube(ax, div_xy, safe_r, color="red", alpha=0.15,
                    label=f"red safe (r={safe_r}m)")
    ax.plot(div_xy[:, 0], div_xy[:, 1],
            color="red", lw=1.8, zorder=3, label="red (actual)")
    if "red_arrow_xy" in info:
        rax = info["red_arrow_xy"]
        ax.scatter(rax[:, 0], rax[:, 1],
                   marker="x", s=40, color="darkred", zorder=5, label="red arrow pts")

    # Seed points (only for circle 1 which has a real seed index)
    if "seed_indices" in info and "green_xy" in info:
        for k, si in enumerate(info["seed_indices"]):
            if si is None:
                continue
            c = info["green_xy"][si]
            ax.scatter(*c, marker="*", s=150, color=colors[k % len(colors)],
                       zorder=6, label=f"seed #{k+1}")

    # Obstacle circles
    for k, obs in enumerate(obstacles):
        col  = colors[k % len(colors)]
        circ = plt.Circle(obs.center, obs.radius, color=col, alpha=0.35, zorder=4)
        ax.add_patch(circ)
        ax.scatter(*obs.center, s=40, color=col, zorder=7)
        ax.annotate(
            f"#{k+1} r={obs.radius:.2f}m",
            xy=obs.center,
            xytext=(obs.center[0] + obs.radius * 0.7 + 0.05,
                    obs.center[1] + obs.radius * 0.7 + 0.05),
            fontsize=7, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
        )

    # Start / end markers
    ax.scatter(*div_xy[0],  marker="^", s=80, color="blue",   zorder=6, label="start")
    ax.scatter(*div_xy[-1], marker="s", s=80, color="purple", zorder=6, label="end")

    # Excluded margin region
    if "green_xy" in info and "margin" in info:
        m   = info["margin"]
        gxy = info["green_xy"]
        ax.scatter(gxy[:m, 0],  gxy[:m, 1],  s=8, color="gray", alpha=0.5)
        ax.scatter(gxy[-m:, 0], gxy[-m:, 1], s=8, color="gray", alpha=0.5,
                   label=f"excl. margin ±{m}")

    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    tag   = "OBS" if obstacles else "skip"
    fname = os.path.join(out_dir, f"m{motion_idx:03d}_w{win_idx:03d}_{tag}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="2-D debug: root trajectory + divergence obstacle visualiser"
    )
    p.add_argument("--dataset",            default="lafan1_g1_motion23")
    p.add_argument("--motion",             type=int,   default=0,
                   help="Motion clip index (0-based)")
    p.add_argument("--obstacle-interval",  type=int,   default=30)
    p.add_argument("--future-frames",      type=int,   default=45)
    p.add_argument("--all-windows",        action="store_true",
                   help="Save a PNG for every window, not just those with obstacles")
    p.add_argument("--robot-safe-radius",  type=float, default=0.25)
    p.add_argument("--out-dir",            default="debug_2d")
    return p.parse_args()


def main():
    args = get_args()

    dataset_path = f"data/pkls/{args.dataset}.pkl"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    dataset = MotionDataset(dataset_path)
    motion  = dataset[args.motion % len(dataset)]
    print(f"Motion {args.motion}: style={motion.style}  frames={motion.num_frames}")

    all_qpos     = motion.get_all_qpos()   # (T, 36)
    full_path_xy = all_qpos[:, :2]
    T            = motion.num_frames

    os.makedirs(args.out_dir, exist_ok=True)

    saved = skipped = 0
    for win_idx, w_start in enumerate(range(0, T, args.obstacle_interval)):
        w_end   = min(w_start + args.obstacle_interval, T)
        div_end = min(w_end   + args.future_frames,     T)
        div_xy  = all_qpos[w_start:div_end, :2]

        obs_list, info = compute_divergence_obstacles(
            div_xy, robot_safe_radius=args.robot_safe_radius
        )

        status = (
            f"PLACED {len(obs_list)} obs  " +
            "  ".join(f"r={o.radius:.2f}" for o in obs_list)
        ) if obs_list else f"skip ({info.get('reason', '')})"
        print(f"  window {win_idx:3d}  [{w_start:4d},{w_end:4d})  "
              f"N_div={len(div_xy):3d}  {status}")

        if obs_list or args.all_windows:
            fname = plot_window(
                motion_idx=args.motion, win_idx=win_idx,
                full_path_xy=full_path_xy, div_xy=div_xy,
                obstacles=obs_list, info=info,
                out_dir=args.out_dir, w_start=w_start, w_end=w_end,
            )
            print(f"    → saved {fname}")
            saved += 1
        else:
            skipped += 1

    print(f"\nDone. saved={saved}  skipped={skipped}  out_dir={args.out_dir}/")


if __name__ == "__main__":
    main()
