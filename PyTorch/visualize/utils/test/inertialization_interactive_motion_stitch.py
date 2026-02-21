"""
Interactive motion-stitch inertialization test.

Visualizes stitching two motion clips and compares:
- Raw stitched sequence
- CAMDM-style inertialized sequence
- Spring-style inertialized sequence

Interactive controls:
- Motion A index
- Motion B index
- Switch frame in motion A
- Start frame in motion B
- Post-switch horizon
- CAMDM blend times (rotation/position)
- Spring halflife (rotation/position)
- Joint DOF index to inspect
"""

import argparse
import os
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required for this interactive test.\n"
        "Install it with: pip install matplotlib"
    ) from exc


# Allow running from repository root or from this folder.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZE_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
PYTORCH_DIR = os.path.dirname(VISUALIZE_DIR)
if PYTORCH_DIR not in sys.path:
    sys.path.append(PYTORCH_DIR)

from visualize.motion_loader import MotionDataset
from visualize.utils.transition_manager import create_transition_manager


def yaw_from_wxyz(quat_wxyz):
    """Return yaw (radians) from quaternion in wxyz order, shape (T,4) or (4,)."""
    q = np.asarray(quat_wxyz)
    if q.ndim == 1:
        w, x, y, z = q
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_stitch_sequences(
    dataset,
    motion_a_idx,
    motion_b_idx,
    switch_a,
    start_b,
    horizon,
    fps,
    blend_time_rotation,
    blend_time_position,
    spring_halflife_rotation,
    spring_halflife_position,
    quat_slice=slice(3, 7),
):
    ma = dataset[motion_a_idx]
    mb = dataset[motion_b_idx]
    qa = ma.get_all_qpos()
    qb = mb.get_all_qpos()

    switch_a = int(np.clip(switch_a, 1, max(1, qa.shape[0] - 1)))
    start_b = int(np.clip(start_b, 0, max(0, qb.shape[0] - 1)))
    horizon = int(max(1, horizon))

    # Pre segment ends at switch_a-1.
    pre = qa[:switch_a].copy()

    # Post targets from motion B.
    post_targets = []
    for i in range(horizon):
        idx = min(start_b + i, qb.shape[0] - 1)
        post_targets.append(qb[idx].copy())
    post_targets = np.asarray(post_targets)

    raw = np.concatenate([pre, post_targets], axis=0)

    dt = 1.0 / fps
    manager_camdm = create_transition_manager(
        mode="camdm",
        frame_dt=dt,
        quat_slice=quat_slice,
        blend_time_rotation=blend_time_rotation,
        blend_time_position=blend_time_position,
    )
    manager_spring = create_transition_manager(
        mode="spring",
        frame_dt=dt,
        quat_slice=quat_slice,
        halflife_position=spring_halflife_position,
        halflife_rotation=spring_halflife_rotation,
    )

    if pre.shape[0] >= 2:
        hist = [pre[-2], pre[-1]]
        curr = pre[-1]
    else:
        hist = [pre[-1], pre[-1]]
        curr = pre[-1]
    manager_camdm.start_transition(hist, curr, post_targets)
    manager_spring.start_transition(hist, curr, post_targets)

    post_camdm = []
    post_spring = []
    for i in range(post_targets.shape[0]):
        post_camdm.append(manager_camdm.apply(post_targets[i]))
        post_spring.append(manager_spring.apply(post_targets[i]))
    post_camdm = np.asarray(post_camdm)
    post_spring = np.asarray(post_spring)

    inert_camdm = np.concatenate([pre, post_camdm], axis=0)
    inert_spring = np.concatenate([pre, post_spring], axis=0)
    switch_global = pre.shape[0]
    t = np.arange(raw.shape[0]) * dt

    return t, raw, inert_camdm, inert_spring, switch_global


def main():
    parser = argparse.ArgumentParser(description="Interactive motion stitch inertialization test")
    parser.add_argument("--dataset", type=str, default="lafan1_g1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--horizon", type=int, default=120)
    args = parser.parse_args()

    dataset_path = os.path.join(PYTORCH_DIR, "data", "pkls", f"{args.dataset}.pkl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = MotionDataset(dataset_path)
    n_motions = len(dataset)

    init = {
        "motion_a_idx": 0,
        "motion_b_idx": min(1, n_motions - 1),
        "switch_a": 60,
        "start_b": 0,
        "horizon": args.horizon,
        "blend_time_rotation": 0.2,
        "blend_time_position": 0.2,
        "spring_halflife_rotation": 0.12,
        "spring_halflife_position": 0.12,
        "joint_dof_idx": 7,  # qpos scalar dof index to inspect
    }

    t, raw, inert_camdm, inert_spring, switch_g = build_stitch_sequences(
        dataset=dataset,
        motion_a_idx=init["motion_a_idx"],
        motion_b_idx=init["motion_b_idx"],
        switch_a=init["switch_a"],
        start_b=init["start_b"],
        horizon=init["horizon"],
        fps=args.fps,
        blend_time_rotation=init["blend_time_rotation"],
        blend_time_position=init["blend_time_position"],
        spring_halflife_rotation=init["spring_halflife_rotation"],
        spring_halflife_position=init["spring_halflife_position"],
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.30, hspace=0.25)

    # Root X
    l_raw_x, = axes[0].plot(t, raw[:, 0], color="red", lw=1.8, label="Raw")
    l_camdm_x, = axes[0].plot(t, inert_camdm[:, 0], color="green", lw=1.8, label="CAMDM")
    l_spring_x, = axes[0].plot(t, inert_spring[:, 0], color="blue", lw=1.5, label="Spring")
    axes[0].set_ylabel("Root X")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    # Root Z
    l_raw_z, = axes[1].plot(t, raw[:, 2], color="red", lw=1.8)
    l_camdm_z, = axes[1].plot(t, inert_camdm[:, 2], color="green", lw=1.8)
    l_spring_z, = axes[1].plot(t, inert_spring[:, 2], color="blue", lw=1.5)
    axes[1].set_ylabel("Root Z")
    axes[1].grid(alpha=0.25)

    # Root yaw
    l_raw_yaw, = axes[2].plot(t, yaw_from_wxyz(raw[:, 3:7]), color="red", lw=1.8)
    l_camdm_yaw, = axes[2].plot(t, yaw_from_wxyz(inert_camdm[:, 3:7]), color="green", lw=1.8)
    l_spring_yaw, = axes[2].plot(t, yaw_from_wxyz(inert_spring[:, 3:7]), color="blue", lw=1.5)
    axes[2].set_ylabel("Root Yaw (rad)")
    axes[2].grid(alpha=0.25)

    # Selected scalar DOF
    l_raw_dof, = axes[3].plot(t, raw[:, init["joint_dof_idx"]], color="red", lw=1.8)
    l_camdm_dof, = axes[3].plot(t, inert_camdm[:, init["joint_dof_idx"]], color="green", lw=1.8)
    l_spring_dof, = axes[3].plot(t, inert_spring[:, init["joint_dof_idx"]], color="blue", lw=1.5)
    axes[3].set_ylabel(f"qpos[{init['joint_dof_idx']}]")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.25)

    vlines = [ax.axvline(t[switch_g], color="black", ls="--", lw=1.0) for ax in axes]
    axes[0].set_title("Motion Stitch Test: Raw vs CAMDM vs Spring")

    def add_slider(y, label, vmin, vmax, vinit, valstep=None):
        sax = plt.axes([0.10, y, 0.36, 0.025])
        return Slider(sax, label, vmin, vmax, valinit=vinit, valstep=valstep)

    s_a = add_slider(0.24, "Motion A", 0, n_motions - 1, init["motion_a_idx"], valstep=1)
    s_b = add_slider(0.20, "Motion B", 0, n_motions - 1, init["motion_b_idx"], valstep=1)
    s_sw = add_slider(0.16, "Switch A frame", 1, 400, init["switch_a"], valstep=1)
    s_sb = add_slider(0.12, "Start B frame", 0, 400, init["start_b"], valstep=1)
    s_h = add_slider(0.08, "Horizon", 10, 300, init["horizon"], valstep=1)
    s_br = add_slider(0.24, "Blend rot (s)", 0.01, 1.0, init["blend_time_rotation"])
    s_bp = add_slider(0.20, "Blend pos (s)", 0.01, 1.0, init["blend_time_position"])
    s_dof = add_slider(0.16, "DOF idx", 7, 35, init["joint_dof_idx"], valstep=1)
    s_hr = add_slider(0.12, "Spring hl rot (s)", 0.01, 1.0, init["spring_halflife_rotation"])
    s_hp = add_slider(0.08, "Spring hl pos (s)", 0.01, 1.0, init["spring_halflife_position"])

    # Put right-side sliders.
    for s in (s_br, s_bp, s_dof, s_hr, s_hp):
        x, y, w, h = s.ax.get_position().bounds
        s.ax.set_position([0.58, y, 0.34, h])

    reset_ax = plt.axes([0.47, 0.02, 0.08, 0.035])
    b_reset = Button(reset_ax, "Reset")

    def update(_):
        ta, ra, ia_camdm, ia_spring, swg = build_stitch_sequences(
            dataset=dataset,
            motion_a_idx=int(s_a.val),
            motion_b_idx=int(s_b.val),
            switch_a=int(s_sw.val),
            start_b=int(s_sb.val),
            horizon=int(s_h.val),
            fps=args.fps,
            blend_time_rotation=float(s_br.val),
            blend_time_position=float(s_bp.val),
            spring_halflife_rotation=float(s_hr.val),
            spring_halflife_position=float(s_hp.val),
        )

        l_raw_x.set_data(ta, ra[:, 0]); l_camdm_x.set_data(ta, ia_camdm[:, 0]); l_spring_x.set_data(ta, ia_spring[:, 0])
        l_raw_z.set_data(ta, ra[:, 2]); l_camdm_z.set_data(ta, ia_camdm[:, 2]); l_spring_z.set_data(ta, ia_spring[:, 2])
        l_raw_yaw.set_data(ta, yaw_from_wxyz(ra[:, 3:7]))
        l_camdm_yaw.set_data(ta, yaw_from_wxyz(ia_camdm[:, 3:7]))
        l_spring_yaw.set_data(ta, yaw_from_wxyz(ia_spring[:, 3:7]))

        dof = int(s_dof.val)
        l_raw_dof.set_data(ta, ra[:, dof]); l_camdm_dof.set_data(ta, ia_camdm[:, dof]); l_spring_dof.set_data(ta, ia_spring[:, dof])
        axes[3].set_ylabel(f"qpos[{dof}]")

        for vl in vlines:
            vl.set_xdata([ta[swg], ta[swg]])

        for ax in axes:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    def on_reset(_):
        for s in (s_a, s_b, s_sw, s_sb, s_h, s_br, s_bp, s_dof, s_hr, s_hp):
            s.reset()

    for s in (s_a, s_b, s_sw, s_sb, s_h, s_br, s_bp, s_dof, s_hr, s_hp):
        s.on_changed(update)
    b_reset.on_clicked(on_reset)

    plt.show()


if __name__ == "__main__":
    main()
