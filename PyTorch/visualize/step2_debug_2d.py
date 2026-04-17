"""
Step 2 – 2-D Trajectory & Obstacle Debug Visualiser
=====================================================
Renders a top-down (XY) view of every obstacle-generation window for a given
motion clip and saves a PNG per window whenever a divergence obstacle is placed.

Usage
-----
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23 --motion 2 --all-windows
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23 --out-dir debug_imgs

What each plot shows
--------------------
  Red   solid line   : actual root trajectory (full clip)
  Green solid line   : linear command (straight A→B for each div window)
  Green ×  markers  : arrow sample points (every 5th frame)
  Orange filled circle: divergence obstacle (if placed)
  Red   ×  markers  : red arrow points used for clearance check
  Light grey dots    : raw path_xy (div_xy) used for final radius check
"""

import os
import sys
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset
from utils.environment_sensor import CircleObstacle, quat_wxyz_to_yaw


# ---------------------------------------------------------------------------
# Core divergence logic (mirrors SensorMotionPlayer._divergence_obstacles)
# ---------------------------------------------------------------------------

ARROW_STEP = 5   # matches draw_trajectory_arrows in geometry.py


def divergence_obstacle(path_xy: np.ndarray, robot_safe_radius: float = 0.5, debug: bool = False):
    """
    Returns (CircleObstacle | None, info_dict).

    path_xy : (N, 2) combined window + future_frames trajectory
    """
    N = len(path_xy)
    if N < 4:
        return None, {}

    t = np.linspace(0.0, 1.0, N)
    green_xy = path_xy[0] + t[:, None] * (path_xy[-1] - path_xy[0])

    pointwise = np.linalg.norm(green_xy - path_xy, axis=1)
    if float(pointwise.max()) < 0.10:
        return None, {"reason": f"max_pointwise={pointwise.max():.4f} < 0.10"}

    # Center selection: arrow points only (sparser → larger clearance candidates)
    arrow_idx = np.arange(0, N, ARROW_STEP)
    red_arrow_xy = path_xy[arrow_idx]
    diffs_arrow = green_xy[:, None, :] - red_arrow_xy[None, :, :]
    dist_arrow = np.linalg.norm(diffs_arrow, axis=2).min(axis=1)

    margin = max(2, N // 4)
    dist_arrow[:margin] = 0.0
    dist_arrow[N - margin:] = 0.0

    if float(dist_arrow.max()) < 0.05:
        return None, {"reason": f"arrow_clearance_max={dist_arrow.max():.4f} < 0.05"}

    best_idx = int(dist_arrow.argmax())
    center = green_xy[best_idx].copy()

    # Radius: segment-based distance so no LINE SEGMENT clips the obstacle
    P1 = path_xy[:-1]
    P2 = path_xy[1:]
    seg_d = P2 - P1
    t = np.sum((center - P1) * seg_d, axis=1) / (np.sum(seg_d * seg_d, axis=1) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = P1 + t[:, None] * seg_d
    seg_clearance = float(np.linalg.norm(center - closest, axis=1).min())
    effective_clearance = seg_clearance - robot_safe_radius
    radius = effective_clearance * 0.90

    info = {
        "best_idx": best_idx,
        "N": N,
        "arrow_clearance": float(dist_arrow[best_idx]),
        "seg_clearance": seg_clearance,
        "effective_clearance": effective_clearance,
        "robot_safe_radius": robot_safe_radius,
        "radius": float(radius),
        "max_pointwise": float(pointwise.max()),
        "green_xy": green_xy,
        "red_arrow_xy": red_arrow_xy,
        "path_xy": path_xy,
        "center": center,
        "margin": margin,
    }

    if radius < 0.05:
        info["reason"] = f"radius={radius:.4f} < 0.05"
        return None, info

    return CircleObstacle(center, radius), info


# ---------------------------------------------------------------------------
# Per-window plot
# ---------------------------------------------------------------------------

def draw_safe_tube(ax, xy, radius, color, alpha=0.15, label=None, step=2):
    """Draw robot_safe_radius tube around a polyline as overlapping circles."""
    for i, pt in enumerate(xy[::step]):
        circ = plt.Circle(pt, radius, color=color, alpha=alpha, zorder=1,
                          label=label if i == 0 else None)
        ax.add_patch(circ)


def plot_window(
    motion_idx: int,
    win_idx: int,
    full_path_xy: np.ndarray,
    div_xy: np.ndarray,
    obs,
    info: dict,
    out_dir: str,
    w_start: int,
    w_end: int,
):
    robot_safe_radius = info.get("robot_safe_radius", 0.25)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title(
        f"motion={motion_idx}  window={win_idx}  frames=[{w_start},{w_end})\n"
        f"N_div={info.get('N', len(div_xy))}  "
        f"max_pw={info.get('max_pointwise', 0):.3f}m  "
        f"safe_r={robot_safe_radius}m\n"
        + (f"r={info.get('radius',0):.3f}m  "
           f"seg_clr={info.get('seg_clearance',0):.3f}m  "
           f"eff={info.get('effective_clearance',0):.3f}m"
           if obs else f"NO OBS ({info.get('reason','')})")
    )

    # Full clip trajectory (faint background)
    ax.plot(full_path_xy[:, 0], full_path_xy[:, 1],
            color="lightcoral", lw=0.8, alpha=0.3, label="full clip")

    if "green_xy" in info:
        green_xy = info["green_xy"]

        # Green safe tube
        draw_safe_tube(ax, green_xy, robot_safe_radius,
                       color="limegreen", alpha=0.15,
                       label=f"green safe (r={robot_safe_radius}m)")
        # Green line
        ax.plot(green_xy[:, 0], green_xy[:, 1],
                color="green", lw=1.8, zorder=3, label="green (linear interp)")
        # Green arrow pts
        arrow_idx = np.arange(0, len(green_xy), ARROW_STEP)
        ax.scatter(green_xy[arrow_idx, 0], green_xy[arrow_idx, 1],
                   marker="x", s=40, color="green", zorder=5)

    # Red safe tube
    draw_safe_tube(ax, div_xy, robot_safe_radius,
                   color="red", alpha=0.15,
                   label=f"red safe (r={robot_safe_radius}m)")
    # Red line
    ax.plot(div_xy[:, 0], div_xy[:, 1],
            color="red", lw=1.8, zorder=3, label="red (actual)")
    # Red arrow pts
    if "red_arrow_xy" in info:
        rax = info["red_arrow_xy"]
        ax.scatter(rax[:, 0], rax[:, 1],
                   marker="x", s=40, color="darkred", zorder=5, label="red arrow pts")

    # Best center
    if "best_idx" in info and "green_xy" in info:
        c = info["green_xy"][info["best_idx"]]
        ax.scatter(*c, marker="*", s=150, color="orange", zorder=6, label="best center")

    # Obstacle circle
    if obs is not None:
        circ = plt.Circle(obs.center, obs.radius,
                          color="orange", alpha=0.45, zorder=4)
        ax.add_patch(circ)
        ax.scatter(*obs.center, s=40, color="darkorange", zorder=7)
        ax.annotate(
            f"r={obs.radius:.3f}m\neff={info['effective_clearance']:.3f}m",
            xy=obs.center,
            xytext=(obs.center[0] + obs.radius + 0.05,
                    obs.center[1] + obs.radius + 0.05),
            fontsize=7, color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8),
        )

    # Start / end
    ax.scatter(*div_xy[0],  marker="^", s=80, color="blue",   zorder=6, label="start")
    ax.scatter(*div_xy[-1], marker="s", s=80, color="purple", zorder=6, label="end")

    # Excluded margin on green
    if "green_xy" in info and "margin" in info:
        m = info["margin"]
        gxy = info["green_xy"]
        ax.scatter(gxy[:m, 0], gxy[:m, 1], s=8, color="gray", alpha=0.5)
        ax.scatter(gxy[-m:, 0], gxy[-m:, 1], s=8, color="gray", alpha=0.5,
                   label=f"excl. margin ±{m}")

    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    tag = "OBS" if obs else "skip"
    fname = os.path.join(out_dir, f"m{motion_idx:03d}_w{win_idx:03d}_{tag}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="2-D debug: root trajectory + divergence obstacle visualiser"
    )
    p.add_argument("--dataset", default="lafan1_g1_motion23")
    p.add_argument("--motion", type=int, default=0,
                   help="Motion clip index (0-based)")
    p.add_argument("--obstacle-interval", type=int, default=30)
    p.add_argument("--future-frames", type=int, default=45)
    p.add_argument("--all-windows", action="store_true",
                   help="Save ALL windows (not just ones with obstacles)")
    p.add_argument("--robot-safe-radius", type=float, default=0.25)
    p.add_argument("--out-dir", default="debug_2d")
    return p.parse_args()


def main():
    args = get_args()

    dataset_path = f"data/pkls/{args.dataset}.pkl"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    dataset = MotionDataset(dataset_path)
    motion = dataset[args.motion % len(dataset)]

    print(f"Motion {args.motion}: style={motion.style}  frames={motion.num_frames}")

    all_qpos = motion.get_all_qpos()   # (T, 36)
    full_path_xy = all_qpos[:, :2]

    os.makedirs(args.out_dir, exist_ok=True)

    T = motion.num_frames
    obs_interval = args.obstacle_interval
    future_frames = args.future_frames

    saved, skipped = 0, 0
    for win_idx, w_start in enumerate(range(0, T, obs_interval)):
        w_end = min(w_start + obs_interval, T)
        div_end = min(w_end + future_frames, T)

        div_xy = all_qpos[w_start:div_end, :2]

        obs, info = divergence_obstacle(div_xy, robot_safe_radius=args.robot_safe_radius)

        status = "PLACED" if obs else f"skip ({info.get('reason','')})"
        print(f"  window {win_idx:3d}  [{w_start:4d},{w_end:4d})  "
              f"N_div={len(div_xy):3d}  {status}")

        if obs or args.all_windows:
            fname = plot_window(
                motion_idx=args.motion,
                win_idx=win_idx,
                full_path_xy=full_path_xy,
                div_xy=div_xy,
                obs=obs,
                info=info,
                out_dir=args.out_dir,
                w_start=w_start,
                w_end=w_end,
            )
            print(f"    → saved {fname}")
            saved += 1
        else:
            skipped += 1

    print(f"\nDone. saved={saved}  skipped={skipped}  out_dir={args.out_dir}/")


if __name__ == "__main__":
    main()
