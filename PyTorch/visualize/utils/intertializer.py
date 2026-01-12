import numpy as np

LN2 = np.log(2.0)

def quat_normalize(q):
    return q / (np.linalg.norm(q) + 1e-12)

def quat_mul(q1, q2):
    # wxyz
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def ensure_same_hemisphere(q, q_ref):
    # Avoid q and -q discontinuity
    return -q if np.dot(q, q_ref) < 0.0 else q

def quat_to_rotvec(q):
    """log map: unit quat (wxyz) -> rotation vector (3,)"""
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(nv, w)
    # wrap to shortest rotation
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = v / nv
    return axis * angle

def rotvec_to_quat(r):
    """exp map: rotation vector (3,) -> unit quat (wxyz)"""
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = r / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


class Inertializer:
    """
    Drop-in replacement:
    - Treats indices [3:7] as a quaternion (wxyz).
    - Internally stores a 3D rotation-vector offset for the quaternion part.
    - update(dt) still returns an "offset" array of shape (dim,), but:
        * offset[3:7] is NOT meant to be added to a quaternion.
    - Use apply(target_pos, dt) to get the corrected blended pose safely.
    """
    def __init__(self, dim, halflife=0.1, quat_slice=slice(3, 7)):
        self.dim = dim
        self.halflife = halflife
        self.quat_slice = quat_slice

        # Euclidean (everything except quaternion slots)
        self.offset = np.zeros(dim, dtype=np.float64)
        self.offset_vel = np.zeros(dim, dtype=np.float64)

        # Quaternion offset represented in rotvec space (3,)
        self.rot_off = np.zeros(3, dtype=np.float64)
        self.rot_off_vel = np.zeros(3, dtype=np.float64)

    def transition(self, curr_pos, curr_vel, target_pos, target_vel, curr_angvel=None, target_angvel=None):
        """
        Call ONCE at discontinuity.

        curr_pos/target_pos: (dim,) with quaternion at quat_slice (wxyz)
        curr_vel/target_vel: (dim,) (for quat slots you can pass zeros; we won't use them)

        Optional (recommended):
          curr_angvel, target_angvel: (3,) root angular velocity (rad/s)
          If omitted, we assume zero angular-velocity offset.
        """
        curr_pos = curr_pos.astype(np.float64, copy=False)
        target_pos = target_pos.astype(np.float64, copy=False)

        # --- Euclidean offset for all non-quat dims ---
        mask = np.ones(self.dim, dtype=bool)
        mask[self.quat_slice] = False
        self.offset[:] = 0.0
        self.offset_vel[:] = 0.0
        self.offset[mask] = curr_pos[mask] - target_pos[mask]
        self.offset_vel[mask] = curr_vel[mask] - target_vel[mask]

        # --- Quaternion offset as rotvec ---
        q_curr = quat_normalize(curr_pos[self.quat_slice])
        q_tgt  = quat_normalize(target_pos[self.quat_slice])
        q_tgt  = ensure_same_hemisphere(q_tgt, q_curr)

        # rotation to apply on top of target to match current
        q_err = quat_mul(q_curr, quat_inv(q_tgt))
        self.rot_off = quat_to_rotvec(q_err)

        if curr_angvel is None or target_angvel is None:
            self.rot_off_vel[:] = 0.0
        else:
            self.rot_off_vel = (np.asarray(curr_angvel, dtype=np.float64) -
                                np.asarray(target_angvel, dtype=np.float64))

    def _decay_vec(self, x, v, dt):
        """Halflife exponential decay (stable)."""
        if self.halflife <= 0:
            x[:] = 0.0
            v[:] = 0.0
            return x, v

        y = LN2 / self.halflife
        j1 = v + y * x
        e = np.exp(-y * dt)
        x_new = e * (x + j1 * dt)
        v_new = e * (v - y * j1 * dt)
        return x_new, v_new

    def update(self, dt):
        """
        Call EVERY FRAME to decay internal offsets.
        Returns an offset array (dim,), but DO NOT add offset[3:7] to a quaternion.
        Prefer apply(target_pos, dt).
        """
        # decay Euclidean offsets
        self.offset, self.offset_vel = self._decay_vec(self.offset, self.offset_vel, dt)

        # decay quaternion rotvec offset
        self.rot_off, self.rot_off_vel = self._decay_vec(self.rot_off, self.rot_off_vel, dt)

        # Provide something in offset[3:7] for debugging only (not additive!)
        self.offset[self.quat_slice] = 0.0
        return self.offset

    def apply(self, target_pos, dt):
        """
        Safe "drop-in" usage:
            out = inertializer.apply(target_pos, dt)

        - Applies decayed Euclidean offsets by addition
        - Applies decayed quaternion offset by multiplication
        """
        target_pos = target_pos.astype(np.float64, copy=True)

        # decay first
        self.update(dt)

        # apply Euclidean
        mask = np.ones(self.dim, dtype=bool)
        mask[self.quat_slice] = False
        target_pos[mask] += self.offset[mask]

        # apply quaternion
        q_tgt = quat_normalize(target_pos[self.quat_slice])
        q_off = rotvec_to_quat(self.rot_off)
        q_out = quat_mul(q_off, q_tgt)
        target_pos[self.quat_slice] = quat_normalize(q_out)

        return target_pos
