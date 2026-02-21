"""
Interactive trajectory extension/blending test.

This tool mirrors the inertialization interactive tests:
- Slider-driven parameter exploration
- Side-by-side trajectory/time visualizations
- Fast iteration for debugging trajectory conditioning

It visualizes:
1) Predicted trajectory segment
2) HFTE-extended trajectory
3) Target trajectory
4) Blended trajectory used for conditioning
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

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

from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic


def yaw_to_wxyz(yaw):
    """Convert yaw angles (radians) to wxyz quaternions."""
    yaw = np.asarray(yaw, dtype=float)
    h = 0.5 * yaw
    w = np.cos(h)
    z = np.sin(h)
    return np.stack([w, np.zeros_like(w), np.zeros_like(w), z], axis=-1)


def yaw_from_wxyz(q):
    """Extract yaw (radians) from wxyz quaternion array."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        w, x, y, z = q
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def forward_xy_from_wxyz(q):
    """Return XY forward vectors from wxyz quaternions."""
    q = np.asarray(q, dtype=float)
    r = R.from_quat(q[:, [1, 2, 3, 0]])  # to xyzw
    return r.apply(np.array([1.0, 0.0, 0.0]))[:, :2]


def make_trajectory(T, x_end, y_end, yaw_end, curve_xy, curve_yaw):
    """Generate a simple parametric trajectory with optional curvature."""
    T = max(2, int(T))
    u = np.linspace(0.0, 1.0, T)
    x = x_end * u
    y = y_end * u + curve_xy * np.sin(np.pi * u)
    yaw = yaw_end * u + curve_yaw * np.sin(np.pi * u)
    trans = np.stack([x, y], axis=-1)
    pose = yaw_to_wxyz(yaw)
    return trans, pose


def build_sequences(
    total_len,
    pred_len,
    pred_x_end,
    pred_y_end,
    pred_yaw_end,
    pred_curve_xy,
    pred_curve_yaw,
    target_x_end,
    target_y_end,
    target_yaw_end,
    target_curve_xy,
    target_curve_yaw,
    bias_pos,
    bias_rot,
):
    total_len = max(2, int(total_len))
    pred_len = int(np.clip(pred_len, 2, total_len))

    pred_trans, pred_pose = make_trajectory(
        pred_len, pred_x_end, pred_y_end, pred_yaw_end, pred_curve_xy, pred_curve_yaw
    )
    target_trans, target_pose = make_trajectory(
        total_len, target_x_end, target_y_end, target_yaw_end, target_curve_xy, target_curve_yaw
    )

    ext_trans, ext_pose = extend_future_traj_heusristic(
        pred_trans, pred_pose, t_total=total_len, K=1
    )
    blend_trans, blend_pose = blend_trajectory(
        ext_trans, ext_pose, target_trans, target_pose, blend=bias_pos, blend_rot=bias_rot
    )
    return pred_trans, pred_pose, ext_trans, ext_pose, target_trans, target_pose, blend_trans, blend_pose


def main():
    parser = argparse.ArgumentParser(description="Interactive trajectory extension/blending test")
    parser.add_argument("--total-len", type=int, default=45)
    args = parser.parse_args()

    init = {
        "total_len": int(args.total_len),
        "pred_len": 15,
        "pred_x_end": 6.0,
        "pred_y_end": 1.0,
        "pred_yaw_end": 0.6,
        "pred_curve_xy": 1.5,
        "pred_curve_yaw": 0.3,
        "target_x_end": 8.0,
        "target_y_end": 6.0,
        "target_yaw_end": 1.2,
        "target_curve_xy": 0.8,
        "target_curve_yaw": 0.2,
        "bias_pos": 0.4,
        "bias_rot": 2.2,
    }

    seq = build_sequences(**init)
    pred_t, pred_q, ext_t, ext_q, tgt_t, tgt_q, bl_t, bl_q = seq
    idx = np.arange(init["total_len"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    plt.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.34, hspace=0.30)

    # XY
    l_pred_xy, = axes[0].plot(pred_t[:, 0], pred_t[:, 1], color="gray", lw=2.0, label="Predicted (raw)")
    l_ext_xy, = axes[0].plot(ext_t[:, 0], ext_t[:, 1], color="blue", lw=1.8, label="HFTE-extended")
    l_tgt_xy, = axes[0].plot(tgt_t[:, 0], tgt_t[:, 1], color="red", lw=1.8, label="Target")
    l_bl_xy, = axes[0].plot(bl_t[:, 0], bl_t[:, 1], color="green", lw=2.2, label="Blended")

    # Per-frame points
    s_pred_xy = axes[0].scatter(pred_t[:, 0], pred_t[:, 1], color="gray", s=14, alpha=0.8)
    s_ext_xy = axes[0].scatter(ext_t[:, 0], ext_t[:, 1], color="blue", s=14, alpha=0.8)
    s_tgt_xy = axes[0].scatter(tgt_t[:, 0], tgt_t[:, 1], color="red", s=14, alpha=0.8)
    s_bl_xy = axes[0].scatter(bl_t[:, 0], bl_t[:, 1], color="green", s=18, alpha=0.9)

    # Orientation arrows (sampled for readability)
    step = max(1, init["total_len"] // 20)
    scale = 0.5
    pred_fwd = forward_xy_from_wxyz(pred_q)
    ext_fwd = forward_xy_from_wxyz(ext_q)
    tgt_fwd = forward_xy_from_wxyz(tgt_q)
    bl_fwd = forward_xy_from_wxyz(bl_q)
    q_pred = axes[0].quiver(
        pred_t[::step, 0], pred_t[::step, 1], pred_fwd[::step, 0], pred_fwd[::step, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale, color="gray", alpha=0.8, width=0.003
    )
    q_ext = axes[0].quiver(
        ext_t[::step, 0], ext_t[::step, 1], ext_fwd[::step, 0], ext_fwd[::step, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale, color="blue", alpha=0.75, width=0.003
    )
    q_tgt = axes[0].quiver(
        tgt_t[::step, 0], tgt_t[::step, 1], tgt_fwd[::step, 0], tgt_fwd[::step, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale, color="red", alpha=0.75, width=0.003
    )
    q_bl = axes[0].quiver(
        bl_t[::step, 0], bl_t[::step, 1], bl_fwd[::step, 0], bl_fwd[::step, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale, color="green", alpha=0.85, width=0.0035
    )
    p_split = axes[0].scatter([pred_t[-1, 0]], [pred_t[-1, 1]], color="black", s=32, label="Pred horizon")
    axes[0].set_title("Trajectory XY")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(alpha=0.25)
    axes[0].axis("equal")
    axes[0].legend(loc="best")

    # Yaw
    l_pred_yaw, = axes[1].plot(np.arange(pred_q.shape[0]), yaw_from_wxyz(pred_q), color="gray", lw=2.0, label="Pred yaw")
    l_ext_yaw, = axes[1].plot(idx, yaw_from_wxyz(ext_q), color="blue", lw=1.8, label="Ext yaw")
    l_tgt_yaw, = axes[1].plot(idx, yaw_from_wxyz(tgt_q), color="red", lw=1.8, label="Target yaw")
    l_bl_yaw, = axes[1].plot(idx, yaw_from_wxyz(bl_q), color="green", lw=2.2, label="Blend yaw")
    v_split = axes[1].axvline(pred_q.shape[0] - 1, color="black", ls="--", lw=1.0)
    axes[1].set_title("Yaw Over Horizon")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Yaw (rad)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    # Component view
    l_ext_x, = axes[2].plot(idx, ext_t[:, 0], color="blue", lw=1.8, label="Ext X")
    l_tgt_x, = axes[2].plot(idx, tgt_t[:, 0], color="red", lw=1.2, alpha=0.8, label="Target X")
    l_bl_x, = axes[2].plot(idx, bl_t[:, 0], color="green", lw=2.0, label="Blend X")
    l_ext_y, = axes[2].plot(idx, ext_t[:, 1], color="blue", lw=1.8, ls="--", label="Ext Y")
    l_tgt_y, = axes[2].plot(idx, tgt_t[:, 1], color="red", lw=1.2, alpha=0.8, ls="--", label="Target Y")
    l_bl_y, = axes[2].plot(idx, bl_t[:, 1], color="green", lw=2.0, ls="--", label="Blend Y")
    axes[2].set_title("X/Y Over Horizon")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Position")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best", ncol=3)

    def add_slider(y, label, vmin, vmax, vinit, valstep=None, right=False):
        x = 0.10 if not right else 0.57
        w = 0.36
        sax = plt.axes([x, y, w, 0.023])
        return Slider(sax, label, vmin, vmax, valinit=vinit, valstep=valstep)

    s_total = add_slider(0.27, "Total len", 20, 90, init["total_len"], valstep=1, right=False)
    s_pred_len = add_slider(0.24, "Pred len", 2, 90, init["pred_len"], valstep=1, right=False)
    s_pred_x = add_slider(0.21, "Pred X end", -15, 15, init["pred_x_end"], right=False)
    s_pred_y = add_slider(0.18, "Pred Y end", -15, 15, init["pred_y_end"], right=False)
    s_pred_yaw = add_slider(0.15, "Pred yaw end", -np.pi, np.pi, init["pred_yaw_end"], right=False)
    s_pred_cxy = add_slider(0.12, "Pred curve XY", -6, 6, init["pred_curve_xy"], right=False)
    s_pred_cyaw = add_slider(0.09, "Pred curve yaw", -2.0, 2.0, init["pred_curve_yaw"], right=False)

    s_tgt_x = add_slider(0.27, "Target X end", -15, 15, init["target_x_end"], right=True)
    s_tgt_y = add_slider(0.24, "Target Y end", -15, 15, init["target_y_end"], right=True)
    s_tgt_yaw = add_slider(0.21, "Target yaw end", -np.pi, np.pi, init["target_yaw_end"], right=True)
    s_tgt_cxy = add_slider(0.18, "Target curve XY", -6, 6, init["target_curve_xy"], right=True)
    s_tgt_cyaw = add_slider(0.15, "Target curve yaw", -2.0, 2.0, init["target_curve_yaw"], right=True)
    s_bias_pos = add_slider(0.12, "Bias pos", 0.1, 3.0, init["bias_pos"], right=True)
    s_bias_rot = add_slider(0.09, "Bias rot", 0.1, 3.0, init["bias_rot"], right=True)

    reset_ax = plt.axes([0.47, 0.03, 0.08, 0.04])
    b_reset = Button(reset_ax, "Reset")

    def update(_):
        nonlocal q_pred, q_ext, q_tgt, q_bl
        total_len = int(s_total.val)
        pred_len = int(np.clip(s_pred_len.val, 2, total_len))
        seq_ = build_sequences(
            total_len=total_len,
            pred_len=pred_len,
            pred_x_end=float(s_pred_x.val),
            pred_y_end=float(s_pred_y.val),
            pred_yaw_end=float(s_pred_yaw.val),
            pred_curve_xy=float(s_pred_cxy.val),
            pred_curve_yaw=float(s_pred_cyaw.val),
            target_x_end=float(s_tgt_x.val),
            target_y_end=float(s_tgt_y.val),
            target_yaw_end=float(s_tgt_yaw.val),
            target_curve_xy=float(s_tgt_cxy.val),
            target_curve_yaw=float(s_tgt_cyaw.val),
            bias_pos=float(s_bias_pos.val),
            bias_rot=float(s_bias_rot.val),
        )
        p_t, p_q, e_t, e_q, t_t, t_q, b_t, b_q = seq_
        ii = np.arange(total_len)

        l_pred_xy.set_data(p_t[:, 0], p_t[:, 1])
        l_ext_xy.set_data(e_t[:, 0], e_t[:, 1])
        l_tgt_xy.set_data(t_t[:, 0], t_t[:, 1])
        l_bl_xy.set_data(b_t[:, 0], b_t[:, 1])
        s_pred_xy.set_offsets(np.c_[p_t[:, 0], p_t[:, 1]])
        s_ext_xy.set_offsets(np.c_[e_t[:, 0], e_t[:, 1]])
        s_tgt_xy.set_offsets(np.c_[t_t[:, 0], t_t[:, 1]])
        s_bl_xy.set_offsets(np.c_[b_t[:, 0], b_t[:, 1]])
        p_split.set_offsets(np.array([[p_t[-1, 0], p_t[-1, 1]]]))

        # Rebuild quivers for updated lengths.
        for qv in (q_pred, q_ext, q_tgt, q_bl):
            qv.remove()
        step2 = max(1, total_len // 20)
        pf = forward_xy_from_wxyz(p_q)
        ef = forward_xy_from_wxyz(e_q)
        tf = forward_xy_from_wxyz(t_q)
        bf = forward_xy_from_wxyz(b_q)
        q_pred = axes[0].quiver(
            p_t[::step2, 0], p_t[::step2, 1], pf[::step2, 0], pf[::step2, 1],
            angles="xy", scale_units="xy", scale=1.0 / scale, color="gray", alpha=0.8, width=0.003
        )
        q_ext = axes[0].quiver(
            e_t[::step2, 0], e_t[::step2, 1], ef[::step2, 0], ef[::step2, 1],
            angles="xy", scale_units="xy", scale=1.0 / scale, color="blue", alpha=0.75, width=0.003
        )
        q_tgt = axes[0].quiver(
            t_t[::step2, 0], t_t[::step2, 1], tf[::step2, 0], tf[::step2, 1],
            angles="xy", scale_units="xy", scale=1.0 / scale, color="red", alpha=0.75, width=0.003
        )
        q_bl = axes[0].quiver(
            b_t[::step2, 0], b_t[::step2, 1], bf[::step2, 0], bf[::step2, 1],
            angles="xy", scale_units="xy", scale=1.0 / scale, color="green", alpha=0.85, width=0.0035
        )

        l_pred_yaw.set_data(np.arange(p_q.shape[0]), yaw_from_wxyz(p_q))
        l_ext_yaw.set_data(ii, yaw_from_wxyz(e_q))
        l_tgt_yaw.set_data(ii, yaw_from_wxyz(t_q))
        l_bl_yaw.set_data(ii, yaw_from_wxyz(b_q))
        v_split.set_xdata([p_q.shape[0] - 1, p_q.shape[0] - 1])

        l_ext_x.set_data(ii, e_t[:, 0]); l_tgt_x.set_data(ii, t_t[:, 0]); l_bl_x.set_data(ii, b_t[:, 0])
        l_ext_y.set_data(ii, e_t[:, 1]); l_tgt_y.set_data(ii, t_t[:, 1]); l_bl_y.set_data(ii, b_t[:, 1])

        # Keep slider consistent if total length shrinks below pred_len.
        if int(s_pred_len.val) != pred_len:
            s_pred_len.set_val(pred_len)

        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        axes[0].axis("equal")
        fig.canvas.draw_idle()

    def on_reset(_):
        for s in (
            s_total, s_pred_len, s_pred_x, s_pred_y, s_pred_yaw, s_pred_cxy, s_pred_cyaw,
            s_tgt_x, s_tgt_y, s_tgt_yaw, s_tgt_cxy, s_tgt_cyaw, s_bias_pos, s_bias_rot
        ):
            s.reset()

    for s in (
        s_total, s_pred_len, s_pred_x, s_pred_y, s_pred_yaw, s_pred_cxy, s_pred_cyaw,
        s_tgt_x, s_tgt_y, s_tgt_yaw, s_tgt_cxy, s_tgt_cyaw, s_bias_pos, s_bias_rot
    ):
        s.on_changed(update)
    b_reset.on_clicked(on_reset)

    plt.show()


if __name__ == "__main__":
    main()
