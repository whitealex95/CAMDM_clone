import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def blend_qpos(pred_qpos, target_trans, target_pose, blend=0.5):
    """Blend generated(predicted) future trajectory with target trajectory.
    Note that while blend=0 implies pure target, blend =1 doesn't follow pure generation
    Outputs blended trajectory: (1-t^blend) predicted + t^blend target, t ∈ [0, 1]
    
    """
    blended_qpos = pred_qpos.copy()
    pred_trans = pred_qpos[:, :2]
    pred_pose = pred_qpos[:, 3:7]
    
    blended_trans, blended_pose = blend_trajectory(pred_trans, pred_pose, target_trans, target_pose, blend)
    blended_qpos[:, :2] = blended_trans
    blended_qpos[:, 3:7] = blended_pose
        
    return blended_qpos


def blend_trajectory(pred_trans, pred_pose, target_trans, target_pose, blend=0.5):
    """Blend generated(predicted) future trajectory with target trajectory.
    Note that while blend=0 implies pure target, blend =1 doesn't follow pure generation
    Outputs blended trajectory: (1-t^blend) predicted + t^blend target, t ∈ [0, 1]
    Args:
        pred_trans (np.ndarray): [T, 2] predicted trajectory (X,Y)
        pred_pose (np.ndarray): [T, 4] predicted orientation (wxyz)
        target_trans (np.ndarray): [T, 2] target trajectory (X,Y)
        target_pose (np.ndarray): [T, 4] target orientation (wxyz)
        blend (float): blending factor in [0, inf], 0 means pure target. Note that blend=1 is not pure prediction.
    Returns:
        blended_trans (np.ndarray): [T, 2] blended trajectory (X,Y)
        blended_pose (np.ndarray): [T, 4] blended orientation (wxyz)
    """
    blended_trans = pred_trans.copy()
    blended_pose = pred_pose.copy()
    T = len(pred_trans)

    for t in range(T):
        # 1. Position Blend (XY only, preserve Z from generation (physics))
        pred_pos_2d = pred_trans[t]
        tgt_pos_2d = target_trans[t]
        
        # Linear Interpolate XY
        blend_weight = (t/(T-1)) ** blend
        blended_trans[t, :] = (1 - blend_weight) * pred_pos_2d + blend_weight * tgt_pos_2d
        
        # 2. Rotation Blend (Slerp)
        pred_quat = blended_pose[t] # wxyz
        tgt_quat = target_pose[t]   # wxyz
        
        # Scipy uses xyzw
        r_pred = R.from_quat([pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
        r_tgt = R.from_quat([tgt_quat[1], tgt_quat[2], tgt_quat[3], tgt_quat[0]])
        
        # Slerp
        slerp = Slerp([0, 1], R.from_matrix(np.stack([r_pred.as_matrix(), r_tgt.as_matrix()])))
        r_blended = slerp([blend_weight])[0]
        
        # Back to wxyz
        b_q = r_blended.as_quat()
        blended_pose[t, :] = b_q[[3, 0, 1, 2]]  # wxyz
        
    return blended_trans, blended_pose


def extend_future_traj_heusristic(model_pred_future_traj, model_pred_future_orient, t_total, K=4):
    """
    We reuse the last K predicted future trajectory points multiple times, if necessary,
    until the length matches that of the user-supplied synthetic future trjaectory.
    For each recylcle, we establish a local frame at the last trajectory point based on its position and orientation,
    then flip the position and copy the orientation of the last K points to extend the trajectory.

    Args:
        model_pred_future_traj (np.ndarray): [TFcurr, 2] predicted future trajectory (X,Y), TFcurr <= t_total
        model_pred_future_orient (np.ndarray): [TFcurr, 4] predicted future orientation (wxyz)
        t_total (int): required total future length, should match 
        K (int): number of last points to recycle
    """
    
    extended_future_traj = model_pred_future_traj.tolist()
    extended_future_orient = model_pred_future_orient.tolist()
    
    while len(extended_future_traj) < t_total:
        # Get last K points
        last_k_traj = extended_future_traj[-K:]
        last_k_orient = extended_future_orient[-K:]
        
        # Establish local frame at the last point
        last_pos = np.array(last_k_traj[-1])  # (X,Y)
        last_orient = last_k_orient[-1]       # (wxyz)
        r_last = R.from_quat([last_orient[1], last_orient[2], last_orient[3], last_orient[0]])
        forward_vec = r_last.apply(np.array([1.0, 0.0, 0.0]))[:2]  # Project to XY plane
        forward_vec /= np.linalg.norm(forward_vec) + 1e-8
        right_vec = np.array([-forward_vec[1], forward_vec[0]])
        
        # Recycle last K points with flipping
        for i in range(K):
            rel_pos = np.array(last_k_traj[i]) - last_pos  # Relative position to last point
            flipped_rel_pos = rel_pos - 2 * np.dot(rel_pos, forward_vec) * forward_vec  # Flip along forward vector
            new_pos = last_pos + flipped_rel_pos
            extended_future_traj.append(new_pos.tolist())
            extended_future_orient.append(last_k_orient[i])  # Copy orientation
        
    # Trim to required length
    extended_future_traj = np.array(extended_future_traj[:t_total])
    extended_future_orient = np.array(extended_future_orient[:t_total])
    
    return extended_future_traj, extended_future_orient
    
    

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
    
    