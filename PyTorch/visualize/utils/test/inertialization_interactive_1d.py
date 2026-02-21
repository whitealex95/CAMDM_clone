"""
Interactive 1D inertialization test.

This script helps verify and understand scalar inertialization behavior by
comparing CAMDM vs Spring transitions side by side.

Controls (sliders):
- Motion A/B amplitude, frequency, phase, offset
- Switch frame
- CAMDM blend time
- Spring halflife
"""

import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required for this interactive test.\n"
        "Install it with: pip install matplotlib"
    ) from exc


def inertialize_scalar_from_xv(x0, v0, tf, t):
    """CAMDM scalar inertialization polynomial (single scalar)."""
    tf1 = -5.0 * x0 / v0 if abs(v0) > 1e-8 else -1.0
    if tf1 > 0.0:
        tf = min(tf, tf1)
    t = min(t, tf)

    if tf < 1e-5:
        return 0.0

    tf2 = tf * tf
    tf3 = tf2 * tf
    tf4 = tf3 * tf
    tf5 = tf4 * tf
    a0 = (-8.0 * v0 * tf - 20.0 * x0) / tf2
    A = -(a0 * tf2 + 6.0 * v0 * tf + 12.0 * x0) / (2.0 * tf5)
    B = (3.0 * a0 * tf2 + 16.0 * v0 * tf + 30.0 * x0) / (2.0 * tf4)
    C = -(3.0 * a0 * tf2 + 12.0 * v0 * tf + 20.0 * x0) / (2.0 * tf3)

    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    return A * t5 + B * t4 + C * t3 + 0.5 * a0 * t2 + v0 * t + x0


def inertialize_scalar(prev, curr, target, dt, tf, t):
    x0 = curr - target
    v0 = (curr - prev) / dt
    return inertialize_scalar_from_xv(x0, v0, tf, t)


def spring_decay_scalar(offset, offset_vel, dt, halflife):
    """Critically-damped style decay in offset space (scalar version)."""
    if halflife <= 0.0:
        return 0.0, 0.0
    y = np.log(2.0) / halflife
    j1 = offset_vel + y * offset
    e = np.exp(-y * dt)
    new_offset = e * (offset + j1 * dt)
    new_offset_vel = e * (offset_vel - y * j1 * dt)
    return new_offset, new_offset_vel


def make_wave(t, amp, freq, phase, offset):
    return amp * np.sin(2.0 * np.pi * freq * t + phase) + offset


def build_sequences(
    n_frames,
    fps,
    switch_idx,
    amp_a,
    freq_a,
    phase_a,
    off_a,
    amp_b,
    freq_b,
    phase_b,
    off_b,
    blend_time_camdm,
    halflife_spring,
):
    dt = 1.0 / fps
    t = np.arange(n_frames) * dt
    motion_a = make_wave(t, amp_a, freq_a, phase_a, off_a)
    motion_b = make_wave(t, amp_b, freq_b, phase_b, off_b)

    raw = motion_a.copy()
    raw[switch_idx:] = motion_b[switch_idx:]

    out_camdm = raw.copy()
    out_spring = raw.copy()
    if switch_idx >= 1:
        # Keep CAMDM and Spring state separate to avoid cross-contamination.
        prev_camdm = raw[switch_idx - 2] if switch_idx >= 2 else raw[switch_idx - 1]
        curr_camdm = raw[switch_idx - 1]
        elapsed = 0.0
        prev_spring = raw[switch_idx - 2] if switch_idx >= 2 else raw[switch_idx - 1]
        curr_spring = raw[switch_idx - 1]
        target0 = motion_b[switch_idx]
        target1 = motion_b[min(switch_idx + 1, n_frames - 1)]
        spring_offset = curr_spring - target0
        curr_vel = (curr_spring - prev_spring) / dt
        target_vel = (target1 - target0) / dt
        spring_offset_vel = curr_vel - target_vel

        for i in range(switch_idx, n_frames):
            target = motion_b[i]
            tf = max(1e-4, blend_time_camdm - elapsed)
            offset = inertialize_scalar(prev_camdm, curr_camdm, target, dt, tf, dt)
            val_camdm = target + offset
            out_camdm[i] = val_camdm
            prev_camdm, curr_camdm = curr_camdm, val_camdm
            elapsed += dt

            spring_offset, spring_offset_vel = spring_decay_scalar(
                spring_offset, spring_offset_vel, dt, halflife_spring
            )
            out_spring[i] = target + spring_offset

    return t, motion_a, motion_b, raw, out_camdm, out_spring


def main():
    parser = argparse.ArgumentParser(description="Interactive 1D inertialization test")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    n_frames = args.frames
    fps = args.fps

    init = {
        "amp_a": 1.0,
        "freq_a": 0.6,
        "phase_a": 0.0,
        "off_a": 0.0,
        "amp_b": 1.2,
        "freq_b": 1.0,
        "phase_b": 1.0,
        "off_b": 0.8,
        "switch_idx": int(0.45 * n_frames),
        "blend_time_camdm": 0.2,
        "halflife_spring": 0.12,
    }

    t, m1, m2, raw, out_camdm, out_spring = build_sequences(n_frames=n_frames, fps=fps, **init)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.40)

    l1, = ax.plot(t, m1, lw=1.0, color="gray", alpha=0.7, label="Motion A")
    l2, = ax.plot(t, m2, lw=1.0, color="orange", alpha=0.7, label="Motion B")
    lraw, = ax.plot(t, raw, lw=2.0, color="red", label="Raw stitched")
    lcamdm, = ax.plot(t, out_camdm, lw=2.0, color="green", label="CAMDM")
    lspring, = ax.plot(t, out_spring, lw=1.8, color="blue", label="Spring")
    vline = ax.axvline(t[init["switch_idx"]], color="black", ls="--", lw=1.0, label="Switch")

    ax.set_title("1D Inertialization: Raw vs CAMDM vs Spring")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    def add_slider(y, label, vmin, vmax, vinit, valstep=None):
        sax = plt.axes([0.10, y, 0.35, 0.025])
        return Slider(sax, label, vmin, vmax, valinit=vinit, valstep=valstep)

    s_amp_a = add_slider(0.32, "A amp", 0.1, 3.0, init["amp_a"])
    s_freq_a = add_slider(0.28, "A freq", 0.1, 2.5, init["freq_a"])
    s_phase_a = add_slider(0.24, "A phase", -np.pi, np.pi, init["phase_a"])
    s_off_a = add_slider(0.20, "A offset", -2.0, 2.0, init["off_a"])

    s_amp_b = add_slider(0.32, "B amp", 0.1, 3.0, init["amp_b"])
    s_freq_b = add_slider(0.28, "B freq", 0.1, 2.5, init["freq_b"])
    s_phase_b = add_slider(0.24, "B phase", -np.pi, np.pi, init["phase_b"])
    s_off_b = add_slider(0.20, "B offset", -2.0, 2.0, init["off_b"])
    s_switch = add_slider(0.16, "Switch frame", 1, n_frames - 2, init["switch_idx"], valstep=1)
    s_blend = add_slider(0.12, "CAMDM blend (s)", 0.01, 1.0, init["blend_time_camdm"])
    s_half = add_slider(0.08, "Spring hl (s)", 0.01, 1.0, init["halflife_spring"])

    # Put B sliders on right side
    for s in (s_amp_b, s_freq_b, s_phase_b, s_off_b, s_switch, s_blend, s_half):
        x, y, w, h = s.ax.get_position().bounds
        s.ax.set_position([0.57, y, 0.35, h])

    reset_ax = plt.axes([0.45, 0.03, 0.10, 0.04])
    b_reset = Button(reset_ax, "Reset")

    def update(_):
        params = dict(
            n_frames=n_frames,
            fps=fps,
            switch_idx=int(s_switch.val),
            amp_a=float(s_amp_a.val),
            freq_a=float(s_freq_a.val),
            phase_a=float(s_phase_a.val),
            off_a=float(s_off_a.val),
            amp_b=float(s_amp_b.val),
            freq_b=float(s_freq_b.val),
            phase_b=float(s_phase_b.val),
            off_b=float(s_off_b.val),
            blend_time_camdm=float(s_blend.val),
            halflife_spring=float(s_half.val),
        )
        tt, mm1, mm2, rr, oo_camdm, oo_spring = build_sequences(**params)
        l1.set_ydata(mm1)
        l2.set_ydata(mm2)
        lraw.set_ydata(rr)
        lcamdm.set_ydata(oo_camdm)
        lspring.set_ydata(oo_spring)
        vline.set_xdata([tt[params["switch_idx"]], tt[params["switch_idx"]]])

        ymin = min(np.min(mm1), np.min(mm2), np.min(rr), np.min(oo_camdm), np.min(oo_spring)) - 0.4
        ymax = max(np.max(mm1), np.max(mm2), np.max(rr), np.max(oo_camdm), np.max(oo_spring)) + 0.4
        ax.set_ylim(ymin, ymax)
        fig.canvas.draw_idle()

    def on_reset(_):
        for s in (s_amp_a, s_freq_a, s_phase_a, s_off_a, s_amp_b, s_freq_b, s_phase_b, s_off_b, s_switch, s_blend, s_half):
            s.reset()

    for s in (s_amp_a, s_freq_a, s_phase_a, s_off_a, s_amp_b, s_freq_b, s_phase_b, s_off_b, s_switch, s_blend, s_half):
        s.on_changed(update)
    b_reset.on_clicked(on_reset)

    plt.show()


if __name__ == "__main__":
    main()
