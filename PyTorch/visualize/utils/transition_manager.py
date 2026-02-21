import numpy as np


def _quat_normalize(q):
    # q: (4,), single quaternion in wxyz order (no batch support).
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_mul(q1, q2):
    # q1, q2: (4,), single quaternions in wxyz order (no batch support).
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_inv(q):
    # q: (4,), single quaternion in wxyz order (no batch support).
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_axis_angle(axis, angle):
    # axis: (3,), angle: scalar -> returns quaternion (4,) in wxyz (no batch support).
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_norm
    half = 0.5 * angle
    return _quat_normalize(np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64))


def _to_axis_angle(q):
    # q: (4,), wxyz -> returns axis (3,) and angle scalar (no batch support).
    q = _quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-3:
        axis = q[1:]
    else:
        axis = q[1:] / s
    return axis, angle


def _inertialize_scalar_from_xv(x0, v0, dt, tf, t):
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


def _inertialize_scalar(prev, curr, target, dt, tf, t):
    x0 = curr - target
    v0 = (curr - prev) / dt
    return _inertialize_scalar_from_xv(x0, v0, dt, tf, t)


def _inertialize_position(prev, curr, target, dt, tf, t):
    vx0 = curr - target
    vxn1 = prev - target
    x0 = np.linalg.norm(vx0)

    if x0 > 1e-5:
        vx0_dir = vx0 / x0
    elif np.linalg.norm(vxn1) > 1e-5:
        vx0_dir = vxn1 / np.linalg.norm(vxn1)
    else:
        vx0_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    xn1 = np.dot(vxn1, vx0_dir)
    v0 = (x0 - xn1) / dt
    xt = _inertialize_scalar_from_xv(x0, v0, dt, tf, t)
    return xt * vx0_dir + target


def _inertialize_rotation(prev, curr, target, dt, tf, t):
    # prev/curr/target: (4,) root quaternions in wxyz order (no batch support).
    prev = _quat_normalize(prev)
    curr = _quat_normalize(curr)
    target = _quat_normalize(target)

    if np.dot(curr, target) < 0.0:
        target = -target

    q0 = _quat_normalize(_quat_mul(curr, _quat_inv(target)))
    qn1 = _quat_normalize(_quat_mul(prev, _quat_inv(target)))

    axis, x0 = _to_axis_angle(q0)
    w = qn1[0]
    if abs(w) < 1e-8:
        w = np.sign(w) * 1e-8 if w != 0 else 1e-8
    xn1 = 2.0 * np.arctan(np.dot(qn1[1:], axis) / w)
    v0 = (x0 - xn1) / dt
    xt = _inertialize_scalar_from_xv(x0, v0, dt, tf, t)

    qt = _quat_mul(_quat_axis_angle(axis, xt), target)
    return _quat_normalize(qt)


class InertialTransitionManager:
    """CAMDM-style inertialization manager for autoregressive motion chunks."""

    def __init__(self, frame_dt, blend_time_rotation=0.2, blend_time_position=0.2, quat_slice=slice(3, 7)):
        self.frame_dt = float(frame_dt)
        self.blend_time_rotation = float(blend_time_rotation)
        self.blend_time_position = float(blend_time_position)
        # quat_slice points to a single quaternion segment in qpos (default qpos[3:7], wxyz).
        self.quat_slice = quat_slice
        self.prev_state = None
        self.curr_state = None
        self.elapsed = 0.0
        self.active = False

    def start_transition(self, qpos_history, current_qpos, generated_qpos):
        """Start a new inertialized transition at chunk boundary."""
        if len(qpos_history) >= 2:
            self.prev_state = np.array(qpos_history[-2], dtype=np.float64)
            self.curr_state = np.array(qpos_history[-1], dtype=np.float64)
        else:
            curr = np.array(current_qpos, dtype=np.float64)
            self.prev_state = curr.copy()
            self.curr_state = curr.copy()
        self.elapsed = 0.0
        self.active = True

    def apply(self, raw_target_qpos):
        """Apply one CAMDM-style inertialized update to target pose."""
        target = np.array(raw_target_qpos, dtype=np.float64)
        if not self.active or self.curr_state is None or self.prev_state is None:
            return target

        dt = self.frame_dt
        tf_pos = max(1e-4, self.blend_time_position - self.elapsed)
        tf_rot = max(1e-4, self.blend_time_rotation - self.elapsed)

        out = target.copy()

        # Root position (XYZ) uses vector inertialization in CAMDM.
        out[0:3] = _inertialize_position(self.prev_state[0:3], self.curr_state[0:3], target[0:3], dt, tf_pos, dt)

        # Root rotation quaternion from out[self.quat_slice]: shape (4,), wxyz, single sample.
        out[self.quat_slice] = _inertialize_rotation(
            self.prev_state[self.quat_slice],
            self.curr_state[self.quat_slice],
            target[self.quat_slice],
            dt, tf_rot, dt
        )

        # Remaining scalar DOFs use scalar inertialization.
        scalar_start = self.quat_slice.stop
        if scalar_start < out.shape[0]:
            for idx in range(scalar_start, out.shape[0]):
                offset = _inertialize_scalar(
                    self.prev_state[idx], self.curr_state[idx], target[idx], dt, tf_rot, dt
                )
                out[idx] = target[idx] + offset

        # Advance state as CAMDM does with Previous/Current updates each frame.
        self.prev_state = self.curr_state.copy()
        self.curr_state = out.copy()
        self.elapsed += dt
        if self.elapsed >= max(self.blend_time_position, self.blend_time_rotation):
            self.active = False

        return out
