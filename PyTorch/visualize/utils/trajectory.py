import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def blend_trajectories(gen_qpos, target_trans, target_pose, blend=0.5):
    """Blend generated(predicted) future trajectory with target trajectory.
    Note that while blend=0 implies pure target, blend =1 doesn't follow pure generation
    Outputs blended trajectory: (1-t^blend) predicted + t^blend target, t ∈ [0, 1]
    """
    blended_qpos = gen_qpos.copy()
    T = len(gen_qpos)
    # if blend <= 0.0:
    #     return blended_qpos
    for t in range(T):
        # 1. Position Blend (XY only, preserve Z from generation (physics))
        gen_pos = gen_qpos[t, :3]
        tgt_pos_2d = target_trans[t]
        
        # Linear Interpolate XY
        blend_weight = (t/(T-1)) ** blend
        new_x = (1 - blend_weight) * gen_pos[0] + blend_weight * tgt_pos_2d[0]
        new_y = (1 - blend_weight) * gen_pos[1] + blend_weight * tgt_pos_2d[1]
        blended_qpos[t, 0] = new_x
        blended_qpos[t, 1] = new_y
        
        # 2. Rotation Blend (Slerp)
        gen_quat = gen_qpos[t, 3:7] # wxyz
        tgt_quat = target_pose[t]   # wxyz
        
        # Scipy uses xyzw
        r_gen = R.from_quat([gen_quat[1], gen_quat[2], gen_quat[3], gen_quat[0]])
        r_tgt = R.from_quat([tgt_quat[1], tgt_quat[2], tgt_quat[3], tgt_quat[0]])
        
        # Slerp
        slerp = Slerp([0, 1], R.from_matrix(np.stack([r_gen.as_matrix(), r_tgt.as_matrix()])))
        r_blended = slerp([blend_weight])[0]
        
        # Back to wxyz
        b_q = r_blended.as_quat()
        blended_qpos[t, 3] = b_q[3] # w
        blended_qpos[t, 4] = b_q[0] # x
        blended_qpos[t, 5] = b_q[1] # y
        blended_qpos[t, 6] = b_q[2] # z
        
    return blended_qpos


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
    
    blended_qpos = blend_trajectories(gen_qpos, target_trans, target_pose, blend=0.5)
    print("Blended QPos:\n", blended_qpos)
    
    