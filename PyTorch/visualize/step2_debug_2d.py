"""
Step 2 – 2-D Trajectory & Scandot-Fill Debug Visualiser
========================================================
Renders a top-down (XY) view of the scandot-fill obstacle augmentation for a
given motion clip. One PNG is saved every ``--obstacle-interval`` frames
(the obstacle augmentation itself is computed *per frame* with the same
pipeline used by ``step2_visualize_data_env2d.py``).

Usage
-----
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23 --motion 2 --all-windows
  python visualize/step2_debug_2d.py --dataset lafan1_g1_motion23 --out-dir debug_imgs

Legend
------
  Light coral line  : full clip root trajectory (background reference)
  Green line        : command trajectory for the future horizon (--cmd-aug)
  Green tube        : robot safe-radius band around green
  Red line          : actual root trajectory for the future horizon
  Red tube          : robot safe-radius band around red
  Light grey dots   : every sensor scandot at the current robot pose
  Dark filled disks : scandots that became CircleObstacles (scandot-fill)
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
from visualize.utils.detour import (
    make_command_xy,
    make_scandot_fill_obstacles,
)
from utils.environment_sensor import EnvironmentSensor, quat_wxyz_to_yaw


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_safe_tube(ax, xy, radius, color, alpha=0.15, label=None, step=2):
    """Shade a robot-safe-radius band around a polyline using overlapping circles."""
    for i, pt in enumerate(xy[::step]):
        circ = plt.Circle(pt, radius, color=color, alpha=alpha, zorder=1,
                          label=label if i == 0 else None)
        ax.add_patch(circ)


def plot_frame(motion_idx: int, frame: int,
               full_path_xy: np.ndarray,
               red_xy: np.ndarray, green_xy: np.ndarray,
               past_xy: np.ndarray,
               robot_xy: np.ndarray,
               info: dict,
               safe_r: float,
               out_dir: str) -> str:
    """Render one frame's scandot-fill picture to a PNG and return the path."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    n_filled = int(info["fill_mask"].sum())
    has_det  = info["has_detour"]
    obs_r    = info["obstacle_radius"]
    ax.set_title(
        f"motion={motion_idx}  frame={frame}\n"
        f"max_pw={info['max_pointwise']:.3f}m  "
        f"safe_r={safe_r}m  obs_r={obs_r:.3f}m  "
        + (f"filled={n_filled}/{len(info['all_centers'])}"
           if has_det else f"NO DETOUR ({info.get('reason', '')})")
    )

    # Background: full clip
    ax.plot(full_path_xy[:, 0], full_path_xy[:, 1],
            color="lightcoral", lw=0.8, alpha=0.3, label="full clip")

    # Green (command) trajectory + safe tube
    if green_xy is not None and len(green_xy) >= 2:
        _draw_safe_tube(ax, green_xy, safe_r, color="limegreen", alpha=0.10,
                        label=f"green safe (r={safe_r}m)")
        ax.plot(green_xy[:, 0], green_xy[:, 1],
                color="green", lw=1.8, zorder=3, label="green (command)")

    # Red (actual) trajectory + safe tube
    if red_xy is not None and len(red_xy) >= 2:
        _draw_safe_tube(ax, red_xy, safe_r, color="red", alpha=0.12,
                        label=f"red safe (r={safe_r}m)")
        ax.plot(red_xy[:, 0], red_xy[:, 1],
                color="red", lw=1.8, zorder=3, label="red (actual)")

    # Past (blue) trajectory + safe tube — scandots within safe_r of this
    # are also excluded from the fill so obstacles don't appear behind the
    # robot.
    if past_xy is not None and len(past_xy) >= 1:
        _draw_safe_tube(ax, past_xy, safe_r, color="dodgerblue", alpha=0.12,
                        label=f"past safe (r={safe_r}m)")
        if len(past_xy) >= 2:
            ax.plot(past_xy[:, 0], past_xy[:, 1],
                    color="dodgerblue", lw=1.8, zorder=3, label="past (blue)")

    # Scandot grid: all dots in light grey
    centers = info["all_centers"]
    fill_mask = info["fill_mask"]
    ax.scatter(centers[~fill_mask, 0], centers[~fill_mask, 1],
               s=6, color="0.75", zorder=2, label="scandot (free)")

    # Filled scandots: dark disks of obstacle_radius (so the figure matches
    # what the sensor actually sees)
    for c in centers[fill_mask]:
        circ = plt.Circle(c, obs_r, color="midnightblue", alpha=0.55, zorder=4)
        ax.add_patch(circ)
    if n_filled > 0:
        ax.scatter(centers[fill_mask, 0], centers[fill_mask, 1],
                   s=10, color="midnightblue", zorder=5,
                   label=f"scandot (filled, r={obs_r:.2f}m)")

    # Robot position
    ax.scatter(*robot_xy, marker="*", s=180, color="gold",
               edgecolor="black", linewidth=0.6, zorder=7, label="robot")

    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    tag   = "FILL" if has_det else "skip"
    fname = os.path.join(out_dir, f"m{motion_idx:03d}_f{frame:05d}_{tag}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="2-D debug visualiser for scandot-fill obstacle augmentation"
    )
    p.add_argument("--dataset",            default="lafan1_g1_motion23")
    p.add_argument("--motion",             type=int,   default=0,
                   help="Motion clip index (0-based)")
    p.add_argument("--obstacle-interval",  type=int,   default=30,
                   help="PNG sampling stride in frames "
                        "(scandot-fill itself is computed per frame)")
    p.add_argument("--future-frames",      type=int,   default=45)
    p.add_argument("--past-frames",        type=int,   default=10,
                   help="Past frames used by --cmd-aug")
    p.add_argument("--cmd-aug", default="linear",
                   choices=["linear", "extrap_pos", "extrap_pos_noyaw",
                            "extrap_pos_preserve_len", "extrap_hfte"],
                   help="Command trajectory builder (see visualize/utils/detour.md)")
    p.add_argument("--cmd-aug-weight", type=float, default=1.0,
                   help="Blend weight w: command = w*extrap + (1-w)*gt. "
                        "1.0=pure extrap (default), 0.0=pure gt.")
    p.add_argument("--max-range",          type=float, default=2.0,
                   help="Sensor max range in metres (default: 2.0)")
    p.add_argument("--resolution",         type=int,   default=9,
                   help="Sensor radial-ring count (default: 9)")
    p.add_argument("--all-windows",        action="store_true",
                   help="Save a PNG for every sampled frame, not just those "
                        "with at least one filled scandot")
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

    all_qpos     = motion.get_all_qpos()    # (T, 36)
    full_path_xy = all_qpos[:, :2]
    T            = motion.num_frames

    sensor = EnvironmentSensor(
        max_range=args.max_range, resolution=args.resolution
    )
    obs_r = 0.5 * sensor.max_adjacent_distance
    print(f"Sensor: feature_dim={sensor.feature_dim}  "
          f"max_adj_dist={sensor.max_adjacent_distance:.3f}m  "
          f"obstacle_radius={obs_r:.3f}m")

    os.makedirs(args.out_dir, exist_ok=True)

    saved = skipped = 0
    sample_frames = list(range(0, T, args.obstacle_interval))
    for f in sample_frames:
        red_xy  = all_qpos[f : f + args.future_frames, :2]
        past_xy = all_qpos[max(0, f - args.past_frames):f, :2]
        if len(red_xy) < 2:
            skipped += 1
            continue

        green_xy = make_command_xy(
            red_xy, args.cmd_aug, past_xy, weight=args.cmd_aug_weight
        )
        qpos      = all_qpos[f]
        robot_pos = qpos[:2]
        robot_yaw = quat_wxyz_to_yaw(qpos[3:7])

        _, info = make_scandot_fill_obstacles(
            sensor, robot_pos, robot_yaw, green_xy, red_xy,
            past_xy=past_xy,
            robot_safe_radius=args.robot_safe_radius,
        )

        n_filled = int(info["fill_mask"].sum())
        status = (f"FILL {n_filled} dots" if info["has_detour"]
                  else f"skip ({info.get('reason', '')})")
        print(f"  frame {f:5d}  max_pw={info['max_pointwise']:.3f}m  {status}")

        if info["has_detour"] or args.all_windows:
            fname = plot_frame(
                motion_idx=args.motion, frame=f,
                full_path_xy=full_path_xy,
                red_xy=red_xy, green_xy=green_xy,
                past_xy=past_xy,
                robot_xy=robot_pos,
                info=info,
                safe_r=args.robot_safe_radius,
                out_dir=args.out_dir,
            )
            print(f"    → saved {fname}")
            saved += 1
        else:
            skipped += 1

    print(f"\nDone. saved={saved}  skipped={skipped}  out_dir={args.out_dir}/")


if __name__ == "__main__":
    main()
