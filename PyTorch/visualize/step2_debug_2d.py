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
  Green line        : command trajectory for the window (--cmd-aug)
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
from visualize.utils.detour import (
    ARROW_STEP,
    compute_detour_obstacles,
    make_command_xy,
)


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
                color="green", lw=1.8, zorder=3, label="green (command)")
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
    p.add_argument("--past-frames",        type=int,   default=10,
                   help="Past frames used for past_extrap command (default: 10)")
    p.add_argument("--cmd-aug", default="linear",
                   choices=["linear", "extrap_pos", "extrap_pos_noyaw",
                            "extrap_pos_preserve_len", "extrap_hfte"],
                   help="Command trajectory used for detour placement "
                        "(see visualize/utils/detour.md): "
                        "linear=straight start→end (default), "
                        "extrap_pos=constant-velocity extrapolation of past, "
                        "extrap_pos_noyaw=same XY as extrap_pos (visualiser-only yaw differs), "
                        "extrap_pos_preserve_len=past direction with gt step lengths, "
                        "extrap_hfte=HFTE central-symmetry extension of past")
    p.add_argument("--cmd-aug-weight", type=float, default=1.0,
                   help="Linear blend weight w between cmd_aug and ground truth: "
                        "command = w*extrap + (1-w)*gt. "
                        "1.0=pure extrap (default), 0.0=pure gt.")
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
        past_xy = all_qpos[max(0, w_start - args.past_frames):w_start, :2]

        command_xy = make_command_xy(div_xy, args.cmd_aug, past_xy,
                                     weight=args.cmd_aug_weight)
        obs_list, info = compute_detour_obstacles(
            div_xy, command_xy=command_xy, robot_safe_radius=args.robot_safe_radius
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
