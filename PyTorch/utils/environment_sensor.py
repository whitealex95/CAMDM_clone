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
from typing import List, Optional, Tuple


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


# ---------------------------------------------------------------------------
# Obstacle generator (data augmentation)
# ---------------------------------------------------------------------------

class ObstacleGenerator:
    """
    Generates random 2-D obstacles for data augmentation in three density modes.

    Modes
    -----
    ``sparse``
        A small number of randomly placed obstacles scattered within the
        sensor range.  The robot walks through mostly open space.

    ``dense``
        Many randomly placed obstacles fill most of the reachable area around
        the robot's path.  The path remains clear but obstacles are tightly
        packed.

    ``packed``
        Grid-based filling: obstacles are placed on a regular grid across the
        entire sensing area.  Every grid cell that does NOT overlap with the
        robot's safe corridor gets an obstacle, producing a tight "hallway"
        effect.

    Design guarantees (all modes)
    ------------------------------
    * No obstacle comes within ``robot_safe_radius`` of any robot position in
      the trajectory window (including an optional look-ahead window).

    Usage
    -----
    >>> gen = ObstacleGenerator(mode='packed', seed=42)
    >>> obstacles = gen.generate_for_window(robot_xy_window)
    >>> readings, hit_pts = sensor.compute(robot_pos, yaw, obstacles)
    """

    MODES = {'sparse', 'dense', 'packed'}

    # ---- Per-mode defaults -------------------------------------------------
    _MODE_DEFAULTS = {
        'sparse': dict(
            n_obstacles_range=(3, 8),
            circle_radius_range=(0.15, 0.55),
            box_halfextent_range=(0.12, 0.45),
            placement_radius=4.5,
        ),
        'dense': dict(
            n_obstacles_range=(12, 25),
            circle_radius_range=(0.12, 0.40),
            box_halfextent_range=(0.10, 0.35),
            placement_radius=5.0,
        ),
        'packed': dict(
            grid_spacing=0.70,       # distance between grid-cell centres (m)
            obs_half_extent=0.28,    # half-extent of each box in the grid (m)
            fill_radius=6.0,         # grid extends this far from trajectory centroid
        ),
    }

    def __init__(
        self,
        mode: str = 'sparse',
        robot_safe_radius: float = 0.5,
        max_attempts: int = 40,
        circle_prob: float = 0.5,
        seed: Optional[int] = None,
        # Override per-mode defaults (optional)
        n_obstacles_range: Optional[Tuple[int, int]] = None,
        circle_radius_range: Optional[Tuple[float, float]] = None,
        box_halfextent_range: Optional[Tuple[float, float]] = None,
        placement_radius: Optional[float] = None,
        grid_spacing: Optional[float] = None,
        obs_half_extent: Optional[float] = None,
        fill_radius: Optional[float] = None,
    ):
        """
        Args:
            mode:              Obstacle density mode: ``'sparse'``, ``'dense'``, or
                               ``'packed'``.
            robot_safe_radius: Minimum clearance between any obstacle and any
                               robot position in the trajectory window (m).
            max_attempts:      Rejection-sampling retries per obstacle (sparse/dense).
            circle_prob:       Probability of placing a circle rather than a box
                               (sparse/dense modes only).
            seed:              Optional RNG seed for reproducibility.
            n_obstacles_range, circle_radius_range, box_halfextent_range,
            placement_radius:  Overrides for sparse/dense mode parameters.
            grid_spacing, obs_half_extent, fill_radius:
                               Overrides for packed mode parameters.
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")

        self.mode = mode
        self.robot_safe_radius = float(robot_safe_radius)
        self.max_attempts = int(max_attempts)
        self.circle_prob = float(circle_prob)
        self._rng = np.random.default_rng(seed)

        # Merge mode defaults with caller overrides
        defaults = dict(self._MODE_DEFAULTS[mode])

        if mode in ('sparse', 'dense'):
            nr = n_obstacles_range or defaults['n_obstacles_range']
            self.n_min, self.n_max = nr
            self.r_min, self.r_max = circle_radius_range or defaults['circle_radius_range']
            self.b_min, self.b_max = box_halfextent_range or defaults['box_halfextent_range']
            self.placement_radius  = float(placement_radius or defaults['placement_radius'])
        else:  # packed
            self.grid_spacing    = float(grid_spacing    or defaults['grid_spacing'])
            self.obs_half_extent = float(obs_half_extent or defaults['obs_half_extent'])
            self.fill_radius     = float(fill_radius     or defaults['fill_radius'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, s: int):
        """Re-seed the internal RNG."""
        self._rng = np.random.default_rng(s)

    def generate_for_window(
        self,
        robot_xy: np.ndarray,
        lookahead_xy: Optional[np.ndarray] = None,
    ) -> List[Obstacle2D]:
        """
        Generate a set of obstacles compatible with the given trajectory window.

        Args:
            robot_xy:      (T, 2) XY positions of the robot during this window.
            lookahead_xy:  Optional (T2, 2) future positions for extra safety.

        Returns:
            List of Obstacle2D guaranteed to be collision-free with the path.
        """
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        check_xy = robot_xy if lookahead_xy is None else np.concatenate(
            [robot_xy, np.asarray(lookahead_xy, dtype=np.float64)], axis=0
        )

        if self.mode == 'packed':
            return self._generate_packed(robot_xy, check_xy)
        else:
            return self._generate_random(robot_xy, check_xy)

    # ------------------------------------------------------------------
    # Random placement (sparse / dense)
    # ------------------------------------------------------------------

    def _generate_random(
        self,
        robot_xy: np.ndarray,
        check_xy: np.ndarray,
    ) -> List[Obstacle2D]:
        centroid = robot_xy.mean(axis=0)
        n_want = int(self._rng.integers(self.n_min, self.n_max + 1))
        obstacles: List[Obstacle2D] = []
        for _ in range(n_want):
            obs = self._try_place_random(centroid, check_xy)
            if obs is not None:
                obstacles.append(obs)
        return obstacles

    def _try_place_random(
        self,
        centroid: np.ndarray,
        check_xy: np.ndarray,
    ) -> Optional[Obstacle2D]:
        """Rejection-sample one obstacle that does not collide with check_xy."""
        for _ in range(self.max_attempts):
            r     = self._rng.uniform(self.robot_safe_radius * 1.2, self.placement_radius)
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            center = centroid + r * np.array([np.cos(angle), np.sin(angle)])

            if self._rng.random() < self.circle_prob:
                radius = self._rng.uniform(self.r_min, self.r_max)
                obs: Obstacle2D = CircleObstacle(center, radius)
                min_gap = self.robot_safe_radius + radius
            else:
                hx  = self._rng.uniform(self.b_min, self.b_max)
                hy  = self._rng.uniform(self.b_min, self.b_max)
                yaw = self._rng.uniform(0.0, np.pi)
                obs = BoxObstacle(center, np.array([hx, hy]), yaw)
                min_gap = self.robot_safe_radius

            if self._is_safe(obs, check_xy, min_gap):
                return obs

        return None

    # ------------------------------------------------------------------
    # Grid-based packed generation
    # ------------------------------------------------------------------

    def _generate_packed(
        self,
        robot_xy: np.ndarray,
        check_xy: np.ndarray,
    ) -> List[Obstacle2D]:
        """
        Fill a grid with box obstacles, leaving the robot's corridor clear.

        A regular grid is created centred at the trajectory centroid.  Every
        cell whose centre is outside the ``robot_safe_radius + obs_half_extent``
        exclusion zone around every trajectory point gets an obstacle.
        """
        centroid = robot_xy.mean(axis=0)
        step     = self.grid_spacing
        R        = self.fill_radius
        hx = hy  = self.obs_half_extent
        min_gap  = self.robot_safe_radius  # uses BoxObstacle.min_distance_to_point

        xs = np.arange(-R, R + step * 0.5, step)
        ys = np.arange(-R, R + step * 0.5, step)

        obstacles: List[Obstacle2D] = []
        for dx in xs:
            for dy in ys:
                center = centroid + np.array([dx, dy])
                # Add small random jitter so the grid doesn't look perfectly regular
                jitter = self._rng.uniform(-step * 0.15, step * 0.15, size=2)
                center = center + jitter

                obs = BoxObstacle(center, np.array([hx, hy]),
                                  yaw=self._rng.uniform(0.0, np.pi * 0.25))
                if self._is_safe(obs, check_xy, min_gap):
                    obstacles.append(obs)

        return obstacles

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _is_safe(
        self,
        obs: Obstacle2D,
        check_xy: np.ndarray,
        min_gap: float,
    ) -> bool:
        """Return True if the obstacle is collision-free with every check point."""
        for pt in check_xy:
            if obs.min_distance_to_point(pt) < min_gap:
                return False
        return True


# ---------------------------------------------------------------------------
# Per-clip sensor augmentation (used by the dataset builder)
# ---------------------------------------------------------------------------

def compute_clip_sensor_readings(
    all_qpos: np.ndarray,
    sensor: EnvironmentSensor,
    generator: ObstacleGenerator,
    obstacle_interval: int = 30,
    lookahead_frames: int = 30,
) -> Tuple[np.ndarray, List]:
    """
    Compute sensor readings for an entire motion clip with time-varying obstacles.

    Obstacles are regenerated every ``obstacle_interval`` frames.
    Each window's obstacles are checked against the robot path in that window
    plus ``lookahead_frames`` into the future so the obstacles don't block the
    immediately upcoming path.

    Args:
        all_qpos:          (T, 36) qpos sequence for the full clip.
        sensor:            EnvironmentSensor instance.
        generator:         ObstacleGenerator instance.
        obstacle_interval: Frames between obstacle regeneration.
        lookahead_frames:  Future frames included in collision check.

    Returns:
        readings:          (T, n_rays) float32 sensor readings.
        window_obstacles:  List of (start, end, List[Obstacle2D]) for visualisation.
    """
    T = all_qpos.shape[0]
    readings = np.zeros((T, sensor.n_rays), dtype=np.float32)
    window_obstacles = []

    starts = list(range(0, T, obstacle_interval))
    for w_start in starts:
        w_end = min(w_start + obstacle_interval, T)
        la_end = min(w_end + lookahead_frames, T)

        # Robot XY in this window (for collision check)
        window_xy = all_qpos[w_start:w_end, :2]
        lookahead_xy = all_qpos[w_end:la_end, :2] if la_end > w_end else None

        obstacles = generator.generate_for_window(window_xy, lookahead_xy)
        window_obstacles.append((w_start, w_end, obstacles))

        # Compute readings for every frame in this window
        for t in range(w_start, w_end):
            pos = all_qpos[t, :3]
            yaw = quat_wxyz_to_yaw(all_qpos[t, 3:7])
            r, _ = sensor.compute(pos, yaw, obstacles)
            readings[t] = r

    return readings, window_obstacles
