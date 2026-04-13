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

def draw_sensor_readings(
    scene,
    robot_pos: np.ndarray,
    readings: np.ndarray,
    hit_points: np.ndarray,
    z_height: float = 0.08,
    dot_radius: float = 0.04,
    draw_lines: bool = True,
):
    """
    Visualise EnvironmentSensor scan-dot readings in a MuJoCo scene.

    Draws:
      - A thin line from the robot to each ray endpoint.
      - A small sphere (dot) at each ray endpoint.
    Color: green = free (0), red = occupied (1).

    Args:
        scene:      MuJoCo viewer user scene.
        robot_pos:  (3,) or (2,) world-frame robot position.
        readings:   (n_rays,) float array, 0 = free, 1 = occupied.
        hit_points: (n_rays, 2) world XY of ray endpoints.
        z_height:   Height above ground at which rays are drawn.
        dot_radius: Radius of the scan-dot spheres.
        draw_lines: Whether to draw the ray lines (can be toggled off for
                    a cleaner dot-only display).
    """
    pos_3d = np.array([robot_pos[0], robot_pos[1], z_height], dtype=np.float64)

    for reading, hp in zip(readings, hit_points):
        is_hit = reading > 0.5
        line_rgba = np.array([1.0, 0.2, 0.2, 0.5], dtype=np.float32) if is_hit \
               else np.array([0.2, 1.0, 0.2, 0.2], dtype=np.float32)
        dot_rgba  = np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32) if is_hit \
               else np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)

        hp_3d = np.array([hp[0], hp[1], z_height], dtype=np.float64)

        if draw_lines and scene.ngeom < scene.maxgeom:
            init_geom(scene.geoms[scene.ngeom], line_rgba)
            mujoco.mjv_connector(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE, 1.5,
                pos_3d, hp_3d,
            )
            scene.ngeom += 1

        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([dot_radius, dot_radius, dot_radius], dtype=np.float64),
                pos=hp_3d,
                mat=np.eye(3).flatten(),
                rgba=dot_rgba,
            )
            scene.ngeom += 1


def draw_obstacle_box(
    scene,
    center_xy: np.ndarray,
    half_extents_xy: np.ndarray,
    yaw: float = 0.0,
    height: float = 0.2,
    color=None,
):
    """
    Draw a rectangular obstacle as a semi-transparent box sitting on the ground.

    Args:
        scene:           MuJoCo viewer user scene.
        center_xy:       (2,) obstacle centre in world XY.
        half_extents_xy: (2,) half-widths [hx, hy] in the obstacle's local frame.
        yaw:             Obstacle orientation in radians (CCW from world +X).
        height:          Full height of the box in metres (default 0.2 m).
        color:           RGBA list/array.  Defaults to semi-transparent orange.
    """
    if scene.ngeom >= scene.maxgeom:
        return
    if color is None:
        color = [0.9, 0.45, 0.1, 0.55]

    c, s = np.cos(yaw), np.sin(yaw)
    mat = np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pos = np.array([center_xy[0], center_xy[1], height * 0.5], dtype=np.float64)
    size = np.array([half_extents_xy[0], half_extents_xy[1], height * 0.5], dtype=np.float64)

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        mat=mat,
        rgba=np.array(color, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_obstacle_circle(
    scene,
    center_xy: np.ndarray,
    radius: float,
    height: float = 0.2,
    color=None,
):
    """
    Draw a circular obstacle as a semi-transparent cylinder on the ground.

    Args:
        scene:     MuJoCo viewer user scene.
        center_xy: (2,) obstacle centre in world XY.
        radius:    Obstacle radius in metres.
        height:    Full height of the cylinder in metres.
        color:     RGBA list/array.  Defaults to semi-transparent red-orange.
    """
    if scene.ngeom >= scene.maxgeom:
        return
    if color is None:
        color = [0.9, 0.25, 0.15, 0.55]

    pos  = np.array([center_xy[0], center_xy[1], height * 0.5], dtype=np.float64)
    size = np.array([radius, radius, height * 0.5], dtype=np.float64)

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=size,
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=np.array(color, dtype=np.float32),
    )
    scene.ngeom += 1


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
