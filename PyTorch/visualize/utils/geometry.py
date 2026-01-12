import numpy as np
import mujoco
from mujoco.viewer import Handle
from visualize.utils.rotations import rot_from_wxyz

# --- HELPER TO DRAW AXES ---
def can_draw(viewer: Handle, n=1):
    return viewer.user_scn.ngeom + n <= viewer.user_scn.maxgeom

def init_geom(geom, color):
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=[1, 0, 0],     # Will be overridden by connector
        pos=[0, 0, 0],      # Will be overridden by connector
        mat=np.eye(3).flatten(),
        rgba=color
    )

def draw_orientation_arrow(scene, pos, quat, color=[0, 1, 0, 1]):
    """Draws a single Forward (X-axis) arrow for the orientation"""
    rot = rot_from_wxyz(quat)

    # Assuming X is forward in the data (Different convention from Unity)
    forward_vec = rot[:, 0] 
    endpoint = pos + forward_vec * 0.3 # 0.3m length

    init_geom(scene.geoms[scene.ngeom], color)
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW, # Use ARROW instead of LINE
        0.03,                        # Arrow thickness
        pos,
        endpoint
    )
    scene.ngeom += 1

def draw_trajectory_lines(scene, traj_pos, color=[0.2, 0.5, 1.0, 1.0]):
    """Draws lines connecting trajectory points."""
    for i in range(len(traj_pos) - 1):
        if scene.ngeom >= scene.maxgeom: break
        init_geom(scene.geoms[scene.ngeom], color)
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE, 10.0,
            traj_pos[i], traj_pos[i+1],
        )
        scene.ngeom += 1

def draw_trajectory_arrows(scene, traj_pos, traj_orient, color=[0.2, 0.5, 1.0, 1.0]):
    """Draws orientation arrows along the trajectory."""
    for i in range(0, len(traj_pos), 5):  # Every 5th frame
        if scene.ngeom >= scene.maxgeom: break
        draw_orientation_arrow(scene, traj_pos[i], traj_orient[i], color)

def draw_trajectory(scene, traj_pos, traj_orient, color=[0.2, 0.5, 1.0, 1.0]):
    """Draws both lines and orientation arrows for a trajectory."""
    if traj_pos.shape[1] != 3:
        traj_pos = np.hstack([traj_pos, np.zeros((traj_pos.shape[0], 1))])  # Add Z=0 plane
    # Ensure data is contiguous float64 arrays
    traj_pos = np.ascontiguousarray(traj_pos, dtype=np.float64)
    traj_orient = np.ascontiguousarray(traj_orient, dtype=np.float64)
    draw_trajectory_lines(scene, traj_pos, color)
    draw_trajectory_arrows(scene, traj_pos, traj_orient, color)

def draw_label(scene, position: np.ndarray, label: str, size: float = 0.2):
    # create an invisibale geom and add label on it
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0, 0, 0]),  # size doesnt matter because it is invisible
        pos=position,  # label position
        mat=np.eye(3).flatten(),  # label orientation, here is no rotation
        rgba=np.array([0, 0, 0, 0])  # invisible
    )
    geom.label = label  # receive string input only
    scene.ngeom += 1
