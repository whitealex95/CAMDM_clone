"""
Environment Sensor for 2D Obstacle Avoidance
---------------------------------------------
Implements a spherical (2D circular) scan-dot sensor inspired by
Neural State Machine's sensing approach.

Each sensor ray is cast in the robot's local frame (yaw-aligned),
uniformly distributed around 360 degrees. Returns binary 0/1 occupancy
per ray.

Usage:
    sensor = EnvironmentSensor(n_rays=36, max_range=3.0)
    readings, hit_points = sensor.compute(robot_pos, robot_yaw, obstacles)
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    """
    Extract yaw angle (rotation around Z-up axis) from a wxyz quaternion.

    Assumes:
        - Z is the gravity-up axis
        - X is the robot forward axis (consistent with LAFAN G1 convention)

    Args:
        quat_wxyz: (4,) array [w, x, y, z]

    Returns:
        yaw in radians (CCW from world +X axis)
    """
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# Obstacle base class and implementations
# ---------------------------------------------------------------------------

class Obstacle2D:
    """Abstract 2D obstacle for ray-casting intersection tests."""

    def intersect_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_dist: float,
    ) -> Tuple[bool, float]:
        """
        Test whether a ray hits this obstacle.

        Args:
            origin:    (2,) world-frame ray origin [x, y]
            direction: (2,) unit ray direction [dx, dy]  (must be normalised)
            max_dist:  maximum ray distance to test

        Returns:
            (hit, distance)  –  distance is np.inf when there is no hit.
        """
        raise NotImplementedError

    def min_distance_to_point(self, point: np.ndarray) -> float:
        """Minimum distance from the obstacle surface to a 2-D point."""
        raise NotImplementedError


class CircleObstacle(Obstacle2D):
    """Circular 2-D obstacle."""

    def __init__(self, center: np.ndarray, radius: float):
        """
        Args:
            center: (2,) world-frame centre [x, y]
            radius: obstacle radius in metres
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)

    def intersect_ray(self, origin, direction, max_dist):
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        oc = origin - self.center
        # Quadratic: t^2 + 2*b*t + c = 0  (a == 1 because direction is unit)
        b = float(np.dot(oc, direction))
        c = float(np.dot(oc, oc)) - self.radius ** 2
        disc = b * b - c
        if disc < 0.0:
            return False, np.inf
        sqrtd = np.sqrt(disc)
        t = -b - sqrtd          # smaller root first
        if 0.0 < t <= max_dist:
            return True, float(t)
        t = -b + sqrtd
        if 0.0 < t <= max_dist:
            return True, float(t)
        return False, np.inf

    def min_distance_to_point(self, point):
        return max(
            0.0,
            float(np.linalg.norm(np.asarray(point, dtype=np.float64) - self.center))
            - self.radius,
        )


class BoxObstacle(Obstacle2D):
    """Rectangular 2-D obstacle (axis-aligned or optionally yaw-rotated)."""

    def __init__(
        self,
        center: np.ndarray,
        half_extents: np.ndarray,
        yaw: float = 0.0,
    ):
        """
        Args:
            center:       (2,) world-frame centre [x, y]
            half_extents: (2,) half-widths [hx, hy] in the obstacle's local frame
            yaw:          rotation angle in radians (CCW from world +X)
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.half_extents = np.asarray(half_extents, dtype=np.float64)
        self.yaw = float(yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        # Inverse rotation: world → local  (= R^T)
        self._R_inv = np.array([[c, s], [-s, c]], dtype=np.float64)

    def _to_local(self, v: np.ndarray, is_point: bool = True) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if is_point:
            return self._R_inv @ (v - self.center)
        return self._R_inv @ v

    def intersect_ray(self, origin, direction, max_dist):
        o = self._to_local(origin, is_point=True)
        d = self._to_local(direction, is_point=False)

        tmin, tmax = -np.inf, np.inf
        for i in range(2):
            if abs(d[i]) < 1e-10:
                if abs(o[i]) > self.half_extents[i]:
                    return False, np.inf
            else:
                t1 = (-self.half_extents[i] - o[i]) / d[i]
                t2 = (self.half_extents[i] - o[i]) / d[i]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))

        if tmin > tmax or tmax < 0.0:
            return False, np.inf
        t = tmin if tmin >= 0.0 else tmax
        if t > max_dist:
            return False, np.inf
        return True, float(t)

    def min_distance_to_point(self, point):
        p = self._to_local(point, is_point=True)
        dx = max(abs(p[0]) - self.half_extents[0], 0.0)
        dy = max(abs(p[1]) - self.half_extents[1], 0.0)
        return float(np.sqrt(dx * dx + dy * dy))


# ---------------------------------------------------------------------------
# Environment Sensor
# ---------------------------------------------------------------------------

class EnvironmentSensor:
    """
    2-D circular scan-dot environment sensor.

    N rays are distributed uniformly around the robot in the horizontal (XY)
    plane.  Ray 0 points along the robot's local +X axis (forward); rays
    proceed counter-clockwise.

    Binary readings (0 = free, 1 = occupied) are returned together with the
    world-space 2-D hit-points (or the max-range endpoints when free).

    Example
    -------
    >>> sensor = EnvironmentSensor(n_rays=36, max_range=3.0)
    >>> readings, hit_pts = sensor.compute(qpos[:3], quat_wxyz_to_yaw(qpos[3:7]), obstacles)
    """

    def __init__(self, n_rays: int = 36, max_range: float = 3.0):
        """
        Args:
            n_rays:    Number of scan rays (evenly spaced in 360°).
            max_range: Maximum sensing distance in metres.
        """
        self.n_rays = int(n_rays)
        self.max_range = float(max_range)

        angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
        # Local frame directions (robot forward = local +X)
        self.ray_dirs_local = np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        ).astype(np.float64)   # (n_rays, 2)
        self.ray_angles_local = angles  # (n_rays,) – useful for visualisation

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        obstacles: List[Obstacle2D],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sensor readings at the robot's current pose.

        Args:
            robot_pos:  (2,) or (3,) world-frame robot position.
            robot_yaw:  Heading angle in radians (Z-up, CCW from world +X).
            obstacles:  List of Obstacle2D objects in the world.

        Returns:
            readings:   (n_rays,) float32 – 0.0 = free, 1.0 = occupied.
            hit_points: (n_rays, 2) world XY of each ray's endpoint
                        (actual hit position, or max-range point if free).
        """
        pos_2d = np.asarray(robot_pos[:2], dtype=np.float64)
        ray_dirs = self._world_ray_dirs(robot_yaw)  # (n_rays, 2)

        distances = np.full(self.n_rays, self.max_range, dtype=np.float64)
        readings = np.zeros(self.n_rays, dtype=np.float32)

        for obs in obstacles:
            for i, ray_dir in enumerate(ray_dirs):
                hit, dist = obs.intersect_ray(pos_2d, ray_dir, distances[i])
                if hit:
                    readings[i] = 1.0
                    distances[i] = min(distances[i], dist)

        hit_points = pos_2d[np.newaxis, :] + ray_dirs * distances[:, np.newaxis]
        return readings, hit_points   # (n_rays,), (n_rays, 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _world_ray_dirs(self, robot_yaw: float) -> np.ndarray:
        """Transform local ray directions to world frame."""
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)  # local → world
        return (R @ self.ray_dirs_local.T).T  # (n_rays, 2)

    @property
    def feature_dim(self) -> int:
        """Sensor output dimensionality (= n_rays)."""
        return self.n_rays
