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

def draw_orientation_arrow(viewer: Handle, pos, quat, color=[0, 1, 0, 1]):
    """Draws a single Forward (X-axis) arrow for the orientation"""
    rot = rot_from_wxyz(quat)

    # Assuming X is forward in the data (Different convention from Unity)
    forward_vec = rot[:, 0] 
    endpoint = pos + forward_vec * 0.3 # 0.3m length

    init_geom(viewer.user_scn.geoms[viewer.user_scn.ngeom], color)
    mujoco.mjv_connector(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW, # Use ARROW instead of LINE
        0.03,                        # Arrow thickness
        pos,
        endpoint
    )
    viewer.user_scn.ngeom += 1

def draw_trajectory_lines(viewer: Handle, traj_pos, color=[0.2, 0.5, 1.0, 1.0]):
    """Draws lines connecting trajectory points."""
    for i in range(len(traj_pos) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        init_geom(viewer.user_scn.geoms[viewer.user_scn.ngeom], color)
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE, 10.0,
            traj_pos[i], traj_pos[i+1],
        )
        viewer.user_scn.ngeom += 1

def draw_trajectory_arrows(viewer: Handle, traj_pos, traj_orient, color=[0.2, 0.5, 1.0, 1.0]):
    """Draws orientation arrows along the trajectory."""
    for i in range(0, len(traj_pos), 5):  # Every 5th frame
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        draw_orientation_arrow(viewer, traj_pos[i], traj_orient[i], color)

def draw_trajectory(viewer: Handle, traj_pos, traj_orient, color=[0.2, 0.5, 1.0, 1.0]):
    """Draws both lines and orientation arrows for a trajectory."""
    # Ensure data is contiguous float64 arrays
    traj_pos = np.ascontiguousarray(traj_pos, dtype=np.float64)
    traj_orient = np.ascontiguousarray(traj_orient, dtype=np.float64)
    draw_trajectory_lines(viewer, traj_pos, color)
    draw_trajectory_arrows(viewer, traj_pos, traj_orient, color)

def draw_label(viewer, position: np.ndarray, label: str, size: float = 0.2):
    # create an invisibale geom and add label on it
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0, 0, 0]),  # size doesnt matter because it is invisible
        pos=position,  # label position
        mat=np.eye(3).flatten(),  # label orientation, here is no rotation
        rgba=np.array([0, 0, 0, 0])  # invisible
    )
    geom.label = label  # receive string input only
    viewer.user_scn.ngeom += 1
