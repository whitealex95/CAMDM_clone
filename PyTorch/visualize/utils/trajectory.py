import numpy as np
from scipy.spatial.transform import Rotation as R

def _blend_scale(t, T, bias):
    """CAMDM-baseline blend scale used by trajectory synthesis.

    CAMDM baseline computes:
      weight = (i - RootPointIndex) / FuturePoints, i in [RootPointIndex+1, ..., RootPointIndex+FuturePoints]
      scale  = 1 - (1 - weight)^bias

    For Python t in [0, T-1], this maps to:
      weight = (t + 1) / T
    """
    if T <= 0:
        return 0.0
    weight = (t + 1) / float(T)
    weight = np.clip(weight, 0.0, 1.0)
    bias = max(float(bias), 1e-6)
    return 1.0 - (1.0 - weight) ** bias


def blend_qpos(pred_qpos, target_trans, target_pose, blend=0.4, blend_rot=None):
    """Blend predicted future trajectory with target trajectory using CAMDM-baseline coefficients.

    Args:
        blend: CAMDM position bias exponent (`bias_HFTE`, default 0.4).
        blend_rot: CAMDM rotation bias exponent. If None, reuse `blend`.
    """
    blended_qpos = pred_qpos.copy()
    pred_trans = pred_qpos[:, :2]
    pred_pose = pred_qpos[:, 3:7]
    
    blended_trans, blended_pose = blend_trajectory(
        pred_trans, pred_pose, target_trans, target_pose, blend=blend, blend_rot=blend_rot
    )
    blended_qpos[:, :2] = blended_trans
    blended_qpos[:, 3:7] = blended_pose
        
    return blended_qpos


def blend_trajectory(pred_trans, pred_pose, target_trans, target_pose, blend=0.4, blend_rot=None):
    """Blend predicted future trajectory with target trajectory using CAMDM-baseline coefficients.

    Position blend follows CAMDM trajectory update curve:
        scale = 1 - (1 - w)^bias, w = (t+1)/T
    then:
        pos = (1-scale) * pred + scale * target

    Rotation is blended as horizontal forward directions (lerp + normalize),
    then converted back to yaw quaternions. This matches CAMDM trajectory
    direction handling more closely than quaternion slerp.

    Args:
        pred_trans (np.ndarray): [T, 2] predicted trajectory (X,Y)
        pred_pose (np.ndarray): [T, 4] predicted orientation (wxyz)
        target_trans (np.ndarray): [T, 2] target trajectory (X,Y)
        target_pose (np.ndarray): [T, 4] target orientation (wxyz)
        blend (float): position bias exponent (CAMDM `bias_HFTE`, default 0.4)
        blend_rot (float|None): rotation bias exponent (CAMDM `bias_dir` analogue).
            If None, reuses `blend`.
    Returns:
        blended_trans (np.ndarray): [T, 2] blended trajectory (X,Y)
        blended_pose (np.ndarray): [T, 4] blended orientation (wxyz)
    """
    blended_trans = pred_trans.copy()
    assert pred_trans.shape == target_trans.shape, "Trajectory shape mismatch"
    blended_pose = pred_pose.copy()
    T = len(pred_trans)
    if blend_rot is None:
        blend_rot = blend

    for t in range(T):
        # 1. Position Blend (XY only, preserve Z from generation (physics))
        pred_pos_2d = pred_trans[t]
        tgt_pos_2d = target_trans[t]
        
        # CAMDM-baseline coefficient curve.
        pos_scale = _blend_scale(t, T, blend)
        blended_trans[t, :] = (1 - pos_scale) * pred_pos_2d + pos_scale * tgt_pos_2d
        
        # 2. Rotation/Direction Blend (direction lerp + normalize)
        pred_quat = blended_pose[t] # wxyz
        tgt_quat = target_pose[t]   # wxyz

        # Scipy uses xyzw.
        r_pred = R.from_quat([pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
        r_tgt = R.from_quat([tgt_quat[1], tgt_quat[2], tgt_quat[3], tgt_quat[0]])
        pred_dir = r_pred.apply(np.array([1.0, 0.0, 0.0]))[:2]
        tgt_dir = r_tgt.apply(np.array([1.0, 0.0, 0.0]))[:2]

        rot_scale = _blend_scale(t, T, blend_rot)
        blended_dir = (1.0 - rot_scale) * pred_dir + rot_scale * tgt_dir
        n = np.linalg.norm(blended_dir)
        if n < 1e-8:
            blended_dir = pred_dir if np.linalg.norm(pred_dir) > 1e-8 else np.array([1.0, 0.0])
            n = np.linalg.norm(blended_dir)
        blended_dir = blended_dir / (n + 1e-8)
        yaw = np.arctan2(blended_dir[1], blended_dir[0])  # XY-plane yaw
        b_q_xyzw = R.from_euler("z", yaw).as_quat()
        blended_pose[t, :] = b_q_xyzw[[3, 0, 1, 2]]  # wxyz
        
    return blended_trans, blended_pose


def extend_future_traj_heusristic(model_pred_future_traj, model_pred_future_orient, t_total, K=4):
    """
    HFTE(Heuristic Future Trajectory Extension)-style trajectory extension (CAMDM baseline analogue).

    Let predicted points be P[0..L-1]. A single HFTE pass creates L-1 extension points:
      1) First symmetry around last point:
         E1[j] = 2*P[L-1] - P[L-2-j], j=0..L-2
      2) Second symmetry around midpoint of E1 (in-place for second half):
         m = floor((L-1)/2)
         E1[j] = 2*E1[m] - E1[2m-j], j=m+1..L-2

    Extension orientations for added points are held constant at the latest
    available predicted orientation (tail direction carry-forward analogue).

    If one pass is still shorter than t_total, the pass is repeated on the
    current sequence until enough points exist.

    Args:
        model_pred_future_traj (np.ndarray): [TFcurr, 2] predicted future trajectory (X,Y), TFcurr <= t_total
        model_pred_future_orient (np.ndarray): [TFcurr, 4] predicted future orientation (wxyz)
        t_total (int): required total future length, should match 
        K (int): unused, kept for backward compatibility
    """
    pred_traj = np.asarray(model_pred_future_traj, dtype=float)
    pred_orient = np.asarray(model_pred_future_orient, dtype=float)

    if pred_traj.shape[0] == 0:
        return np.zeros((t_total, 2), dtype=float), np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (t_total, 1))
    if pred_orient.shape[0] == 0:
        pred_orient = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (pred_traj.shape[0], 1))

    ext_traj = pred_traj.copy()
    ext_orient = pred_orient.copy()

    def _single_hfte_pass(pos_xy):
        L = pos_xy.shape[0]
        if L < 2:
            return np.repeat(pos_xy, 2, axis=0)

        anchor = pos_xy[-1]
        # Stage 1: central symmetry around anchor, using reversed past points.
        e = 2.0 * anchor[None, :] - pos_xy[-2::-1]
        # Stage 2: central symmetry around midpoint for the second half.
        mid = e.shape[0] // 2
        for j in range(mid + 1, e.shape[0]):
            e[j] = 2.0 * e[mid] - e[2 * mid - j]
        return np.concatenate([pos_xy, e], axis=0)

    while ext_traj.shape[0] < t_total:
        prev_len = ext_traj.shape[0]
        ext_traj = _single_hfte_pass(ext_traj)
        # Carry forward tail direction/orientation for extrapolated part.
        add_n = ext_traj.shape[0] - prev_len
        add_orient = np.repeat(ext_orient[prev_len - 1:prev_len], add_n, axis=0)
        ext_orient = np.concatenate([ext_orient, add_orient], axis=0)

    return ext_traj[:t_total], ext_orient[:t_total]
    
def align_trajectory_to_pose(future_traj, future_orient, ref_qpos, curr_qpos):
    """
    Aligns a global trajectory from the dataset to match the robot's current pose.
    The output is still in WORLD coordinates, but shifted/rotated so it starts
    at the robot's current position.
    
    Args:
        future_traj: (T, 2) XY positions from dataset (Global)
        future_orient: (T, 4) wxyz quaternions from dataset (Global)
        ref_qpos: (7,) Reference pose (dataset state at t=0)
        curr_qpos: (7,) Current robot pose (simulation state)
        
    Returns:
        aligned_traj: (T, 2) Aligned XY positions (Global)
        aligned_orient: (T, 4) Aligned wxyz quaternions (Global)
    """
    # Extract states
    ref_pos_xy = ref_qpos[:2]
    curr_pos_xy = curr_qpos[:2]
    
    # Calculate Yaw Difference
    # Scipy expects (x, y, z, w), input is (w, x, y, z)
    r_ref = R.from_quat(ref_qpos[3:7][[1, 2, 3, 0]]) 
    r_curr = R.from_quat(curr_qpos[3:7][[1, 2, 3, 0]])
    
    yaw_ref = r_ref.as_euler('zyx')[0]
    yaw_curr = r_curr.as_euler('zyx')[0]
    delta_yaw = yaw_curr - yaw_ref

    # --- 1. Align Positions (XY) ---
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    rot_mat = np.array([[c, -s], [s, c]])
    
    # Formula: P_aligned = P_current + R_delta * (P_dataset - P_ref)
    rel_pos = future_traj - ref_pos_xy
    aligned_traj = (rot_mat @ rel_pos.T).T + curr_pos_xy

    # --- 2. Align Orientations (Quaternions) ---
    # Formula: Q_aligned = Q_delta * Q_dataset
    r_delta = R.from_euler('z', delta_yaw)
    future_quats_scipy = R.from_quat(future_orient[:, [1, 2, 3, 0]]) # to xyzw
    
    aligned_quats_scipy = r_delta * future_quats_scipy
    
    # Convert back to wxyz
    aligned_orient = aligned_quats_scipy.as_quat()[:, [3, 0, 1, 2]]

    return aligned_traj, aligned_orient

if __name__ == "__main__":
    # Simple test gen_qpos trajectory of 45 steps
    blend = 0.5
    T = 45
    n_joints = 29
    gen_qpos = np.zeros((T, 7 + n_joints))  # 7 for root (pos + quat_wxyz), rest for joints
    
    for t in range(T):
        # Linearly move in x from 0 to 10
        gen_qpos[t, 0] = (10.0 / (T - 1)) * t
        # Keep y and z at 0
        gen_qpos[t, 1] = 0.0
        gen_qpos[t, 2] = 0.0
        # No rotation (identity quaternion)
        gen_qpos[t, 3:7] = [1.0, 0.0, 0.0, 0.0]
    
    # Target trajectory: move in y from 0 to 10, no change in x
    target_trans = np.zeros((T, 2))
    target_pose = np.zeros((T, 4))
    for t in range(T):
        target_trans[t, 0] = 0.0  # x
        target_trans[t, 1] = (10.0 / (T - 1)) * t  # y
        # 90 degree rotation around z-axis
        angle = (np.pi / 2) * (t / (T - 1))
        half_sin = np.sin(angle / 2)
        target_pose[t] = [np.cos(angle / 2), 0.0, 0.0, half_sin]  # wxyz    
    
    blended_qpos = blend_qpos(gen_qpos, target_trans, target_pose, blend=0.5)
    print("Blended QPos:\n", blended_qpos)