"""
NSM Cylindrical Environment Sensor
------------------------------------
Implements the exact cylindrical sensor from the Neural State Machine paper
(Starke et al., SIGGRAPH Asia 2019).  Sampling spheres are placed on a
cylindrical-polar grid centred at the robot's root and each sphere returns a
**continuous** occupancy value in [0, 1].

Sensor Topology (2D mode, Layers = 1)
--------------------------------------
    Resolution : number of radial rings  (paper default: 10)
    max_range  : outermost ring radius   (paper: Size/2)
    Layers     : height layers (1 = 2D, 10 = 3D as in the paper)
    Overlap    : True = overlapping spheres (recommended)

    Derived quantities
    ------------------
    coverage   = 0.5 * (2*max_range) / (Resolution - 1)    [ring spacing]
    Ring z has radius  = z * coverage,  z = 0 … Resolution-1
    Ring z has count   = round(2π * z)  spheres             (0 at z=0)
    sphere_radius      = 0.5 * √2 * coverage  (Overlap=True)
                       = 0.5   * coverage      (Overlap=False)

    Example (max_range=3.0, Resolution=10):
        coverage  = 0.333 m
        feature_dim = 283 spheres per layer  (6+13+19+25+31+38+44+50+57)

Continuous Occupancy Formula
-----------------------------
    s = max(0, min(1, 1 - d / sphere_radius))

    d : min distance from sphere centre to the nearest obstacle surface

Usage
-----
    sensor = EnvironmentSensor(max_range=3.0, resolution=10)
    occupancy, centers = sensor.compute(robot_pos, robot_yaw, obstacles)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


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


def world_traj_to_local(
    traj_xy_world: np.ndarray,
    traj_pose_world_wxyz: np.ndarray,
    curr_xy: np.ndarray,
    curr_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a future trajectory from world frame into the robot-local
    (yaw-frame) at the current frame. This matches the canonical storage
    used by the env2d dataset.

    Args:
        traj_xy_world:        (F, 2) future world XY.
        traj_pose_world_wxyz: (F, 4) future world wxyz quaternions
                              (yaw-only — pitch/roll are dropped).
        curr_xy:              (2,)   current world XY.
        curr_yaw:             current world yaw (radians).

    Returns:
        traj_xy_local:        (F, 2) float32, robot-local XY.
        traj_pose_local_wxyz: (F, 4) float32, yaw-only wxyz relative to
                              ``curr_yaw`` (current frame is identity).
    """
    cos_y, sin_y = np.cos(curr_yaw), np.sin(curr_yaw)
    R_inv = np.array([[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float64)

    dxy = np.asarray(traj_xy_world, dtype=np.float64) - np.asarray(curr_xy, dtype=np.float64)
    traj_xy_local = (R_inv @ dxy.T).T.astype(np.float32)

    yaws = np.array(
        [quat_wxyz_to_yaw(np.asarray(q, dtype=np.float64))
         for q in traj_pose_world_wxyz],
        dtype=np.float64,
    ) - float(curr_yaw)
    traj_pose_local_wxyz = np.stack([
        np.cos(yaws / 2.0),
        np.zeros_like(yaws),
        np.zeros_like(yaws),
        np.sin(yaws / 2.0),
    ], axis=1).astype(np.float32)

    return traj_xy_local, traj_pose_local_wxyz


# ---------------------------------------------------------------------------
# Obstacle base class and implementations
# ---------------------------------------------------------------------------

class Obstacle2D:
    """Abstract 2D obstacle."""

    def min_distance_to_point(self, point: np.ndarray) -> float:
        """Minimum distance from the obstacle surface to a single 2-D point.
        Returns 0 when the point is inside the obstacle."""
        raise NotImplementedError

    def batch_min_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorised minimum distances from the obstacle surface to N points.

        Args:
            points: (N, 2) array of query points.

        Returns:
            (N,) float64 array of distances (0 when inside).
        """
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

    def min_distance_to_point(self, point):
        return max(
            0.0,
            float(np.linalg.norm(np.asarray(point, dtype=np.float64) - self.center))
            - self.radius,
        )

    def batch_min_distance(self, points: np.ndarray) -> np.ndarray:
        # (N,) = ||points - center|| - radius, clipped to 0
        dists = np.linalg.norm(points - self.center, axis=1) - self.radius
        return np.maximum(dists, 0.0)


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
        self._R_inv = np.array([[c, s], [-s, c]], dtype=np.float64)

    def _to_local(self, v: np.ndarray, is_point: bool = True) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if is_point:
            return self._R_inv @ (v - self.center)
        return self._R_inv @ v

    def min_distance_to_point(self, point):
        p = self._to_local(point, is_point=True)
        dx = max(abs(p[0]) - self.half_extents[0], 0.0)
        dy = max(abs(p[1]) - self.half_extents[1], 0.0)
        return float(np.sqrt(dx * dx + dy * dy))

    def batch_min_distance(self, points: np.ndarray) -> np.ndarray:
        # Rotate all points into the box's local frame at once: (N, 2)
        local = (points - self.center) @ self._R_inv.T
        # Per-axis excess beyond half-extents, clamped to 0
        d = np.maximum(np.abs(local) - self.half_extents, 0.0)  # (N, 2)
        return np.sqrt((d * d).sum(axis=1))                      # (N,)


# ---------------------------------------------------------------------------
# NSM Polar Environment Sensor
# ---------------------------------------------------------------------------

class EnvironmentSensor:
    """
    NSM Cylindrical Environment Sensor (Starke et al., SIGGRAPH Asia 2019).

    Replicates CylinderMap.cs exactly.  Sampling spheres are placed on a
    cylindrical-polar grid where each ring z has ``round(2π*z)`` spheres
    (outer rings are denser), and the sphere radius is derived from the grid
    spacing so spheres always overlap their neighbours.

    Parameters
    ----------
    max_range  : outermost sensing radius in metres  (= Size/2 in the paper)
    resolution : number of radial rings              (paper default: 10)
    layers     : height layers (1 = 2D flat sensor)  (paper default: 10 for 3D)
    overlap    : True = sphere_radius = √2/2 * coverage  (recommended)
                 False = sphere_radius = 0.5   * coverage  (no overlap)

    Derived
    -------
    coverage      = max_range / (resolution - 1)
    sphere_radius = 0.5 * √2 * coverage  (Overlap=True)
    ring_slices   : list of (start, end) index ranges, one per ring z

    Example
    -------
    >>> sensor = EnvironmentSensor(max_range=3.0, resolution=10)
    >>> occupancy, centers = sensor.compute(qpos[:3], quat_wxyz_to_yaw(qpos[3:7]), obstacles)
    >>> occupancy.shape   # (283,)  for resolution=10
    """

    def __init__(
        self,
        max_range: float = 2.0,
        resolution: int = 9,
        layers: int = 1,
        overlap: bool = True,
    ):
        """
        Args:
            max_range:  Outermost ring radius in metres (= Size/2 in paper).
                        Paper uses Size=4 → max_range=2.0m.
            resolution: Number of radial rings (= paper's Resolution).
                        Paper default: 9.
            layers:     Height layers (1 for 2D, 9 for full 3D as in paper).
            overlap:    Whether spheres overlap their neighbours.
        """
        self.max_range  = float(max_range)
        self.resolution = int(resolution)
        self.layers     = int(layers)
        self.overlap    = bool(overlap)

        # Paper's CylinderMap.cs:
        #   diameter = Size / (Resolution - 1)   where Size = 2 * max_range
        #   coverage = 0.5 * diameter             (half sphere-diameter step)
        size     = 2.0 * self.max_range
        diameter = size / max(self.resolution - 1, 1)
        coverage = 0.5 * diameter
        self._coverage = coverage

        # Sphere radius derived from coverage (not an independent parameter)
        r_scale = np.sqrt(2.0) if overlap else 1.0
        self.sphere_radius = 0.5 * r_scale * coverage

        # Build 2D polar grid: ring z has round(2π*z) spheres at radius z*coverage
        points: List[Tuple[float, float]] = []
        ring_slices: List[Tuple[int, int]] = []

        for z in range(self.resolution):
            distance = z * coverage
            arc      = 2.0 * np.pi * distance
            count    = int(round(arc / coverage)) if z > 0 else 0

            start = len(points)
            for x in range(count):
                angle = x / count * 2.0 * np.pi
                points.append((distance * np.cos(angle), distance * np.sin(angle)))
            ring_slices.append((start, len(points)))

        self._centers_local = np.array(points, dtype=np.float64)  # (N, 2)
        self.ring_slices    = ring_slices   # per-ring index ranges in _centers_local

        # Maximum nearest-neighbour distance over all scandots. Used by
        # scandot-fill obstacle augmentation: a CircleObstacle of half this
        # radius placed on every scandot guarantees that a fully-filled
        # region has no gaps between adjacent obstacles.
        if len(self._centers_local) > 1:
            diffs = self._centers_local[:, None, :] - self._centers_local[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            np.fill_diagonal(dists, np.inf)
            self.max_adjacent_distance = float(dists.min(axis=1).max())
        else:
            self.max_adjacent_distance = 0.0

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
        Compute continuous occupancy for every sphere at the robot's current pose.

        Args:
            robot_pos:  (2,) or (3,) world-frame robot position.
            robot_yaw:  Heading in radians (Z-up, CCW from world +X).
            obstacles:  List of Obstacle2D objects in the world.

        Returns:
            occupancy:      (feature_dim,) float32 values in [0, 1].
                            0 = clear, 1 = fully inside an obstacle.
            sphere_centers: (feature_dim, 2) world XY of each sphere centre.
        """
        pos_2d = np.asarray(robot_pos[:2], dtype=np.float64)

        # Rotate local sphere centres into the world frame
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        R_mat = np.array([[c, -s], [s, c]], dtype=np.float64)
        centers_world = pos_2d + (R_mat @ self._centers_local.T).T  # (N, 2)

        # Compute min distance from each sphere centre to any obstacle surface.
        # batch_min_distance processes all N centres in one vectorised call,
        # replacing the previous Python-level loop over centres.
        n = len(centers_world)
        min_dists = np.full(n, np.inf, dtype=np.float64)

        for obs in obstacles:
            np.minimum(min_dists, obs.batch_min_distance(centers_world), out=min_dists)

        # Continuous occupancy: s = clamp(1 - d/r, 0, 1)
        r = self.sphere_radius
        occupancy = np.clip(1.0 - min_dists / r, 0.0, 1.0).astype(np.float32)

        return occupancy, centers_world

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Total sensor output dimension = number of spheres per height layer."""
        return len(self._centers_local)


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

    ``compact``
        Flood-fill the entire sensor area with flush square box obstacles,
        leaving only a clear corridor along the trajectory.  Every sensor
        sphere not near the robot path reads as fully occupied.

    Design guarantees (all modes)
    ------------------------------
    * No obstacle comes within ``robot_safe_radius`` of any robot position in
      the trajectory window (including an optional look-ahead window).

    Usage
    -----
    >>> gen = ObstacleGenerator(mode='compact', seed=42)
    >>> obstacles = gen.generate_for_window(robot_xy_window)
    >>> occupancy, centers = sensor.compute(robot_pos, yaw, obstacles)
    """

    MODES = {'sparse', 'dense', 'compact', 'none'}

    _MODE_DEFAULTS = {
        'sparse': dict(
            n_obstacles_range=(3, 8),
            circle_radius_range=(0.10, 0.40),
            box_halfextent_range=(0.08, 0.35),
            placement_radius=3.0,
        ),
        'dense': dict(
            n_obstacles_range=(12, 25),
            circle_radius_range=(0.08, 0.30),
            box_halfextent_range=(0.08, 0.25),
            placement_radius=3.5,
        ),
        # None: no obstacles placed; all sensor readings will be 0.
        'none': dict(),
        # Compact: flood-fill the entire sensor area with square box obstacles,
        # leaving only a clear corridor along the trajectory.  Every sensor
        # sphere that is not near the robot path will read as fully occupied.
        #
        # Coverage guarantee (box version):
        #   coverage_zone = h + sphere_radius = 0.255 + 0.177 = 0.432 m
        #   max grid gap   = spacing * √2 / 2 = 0.50 * 0.707 = 0.354 m
        #   0.354 < 0.432  →  every non-corridor sphere is covered ✓
        #   h = spacing/2 + ε → boxes are flush (no visible gaps between them)
        #
        # Grid bounds are computed from the trajectory AABB + sensor_buffer so
        # that the full 2.0 m sensor range is filled for every frame, even when
        # the robot travels far from the window centroid.
        'compact': dict(
            grid_spacing=0.50,      # distance between box centres (m)
            box_half_extent=0.255,  # square half-size (m); slightly > spacing/2 → boxes flush
            sensor_buffer=2.2,      # grid extends this many metres beyond traj AABB
        ),
    }

    def __init__(
        self,
        mode: str = 'sparse',
        robot_safe_radius: float = 0.5,
        max_attempts: int = 40,
        circle_prob: float = 0.5,
        seed: Optional[int] = None,
        n_obstacles_range: Optional[Tuple[int, int]] = None,
        circle_radius_range: Optional[Tuple[float, float]] = None,
        box_halfextent_range: Optional[Tuple[float, float]] = None,
        placement_radius: Optional[float] = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")

        self.mode = mode
        self.robot_safe_radius = float(robot_safe_radius)
        self.max_attempts = int(max_attempts)
        self.circle_prob = float(circle_prob)
        self._rng = np.random.default_rng(seed)

        defaults = dict(self._MODE_DEFAULTS[mode])

        if mode == 'none':
            pass  # no obstacle parameters needed
        elif mode == 'compact':
            self._grid_spacing    = float(defaults['grid_spacing'])
            self._box_half_extent = float(defaults['box_half_extent'])
            self._sensor_buffer   = float(defaults['sensor_buffer'])
            self.placement_radius = self._sensor_buffer  # kept for API consistency
        else:  # sparse / dense
            self.placement_radius = float(placement_radius or defaults['placement_radius'])
            nr = n_obstacles_range or defaults['n_obstacles_range']
            self.n_min, self.n_max = nr
            self.b_min, self.b_max = box_halfextent_range or defaults['box_halfextent_range']
            self.r_min, self.r_max = circle_radius_range or defaults['circle_radius_range']

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

        if self.mode == 'none':
            return []
        elif self.mode == 'compact':
            return self._generate_compact(robot_xy, check_xy)
        else:
            return self._generate_random(robot_xy, check_xy)

    # ------------------------------------------------------------------
    # Random placement (sparse / dense)
    # ------------------------------------------------------------------

    def _generate_random(self, robot_xy, check_xy):
        centroid = robot_xy.mean(axis=0)
        n_want = int(self._rng.integers(self.n_min, self.n_max + 1))
        obstacles: List[Obstacle2D] = []
        for _ in range(n_want):
            obs = self._try_place_random(centroid, check_xy)
            if obs is not None:
                obstacles.append(obs)
        return obstacles

    def _try_place_random(self, centroid, check_xy):
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
    # Compact generation (flood-fill with trajectory corridor)
    # ------------------------------------------------------------------

    def _generate_compact(self, robot_xy, check_xy):
        """
        Flood-fill the entire sensor area with square box obstacles, leaving
        only the trajectory corridor clear.

        Algorithm
        ---------
        1. Compute the axis-aligned bounding box (AABB) of ``check_xy`` and
           extend it by ``_sensor_buffer`` in every direction.  This ensures
           that the grid covers the full sensor range (2.0 m) from *any* robot
           position in the window, even when the robot travels far from the
           window centroid.
        2. Build a uniform Cartesian grid with step ``_grid_spacing`` inside
           those extended bounds.
        3. Vectorised O(T×N) min-distance check: skip any grid point whose
           closest trajectory point is within ``robot_safe_radius + h``
           (conservative surface-to-path clearance for a box of half-extent h).
        4. Place a square ``BoxObstacle`` (axis-aligned) at each remaining point.

        Coverage guarantee
        ------------------
        For box half-extent h and sphere_radius r_s:
          coverage_zone  = h + r_s  = 0.20 + 0.177 = 0.377 m
          max grid gap   = spacing * √2 / 2 = 0.50 * 0.707 = 0.354 m
          0.354 < 0.377  →  every non-corridor sphere is guaranteed to read > 0
        """
        gs  = self._grid_spacing
        h   = self._box_half_extent
        buf = self._sensor_buffer

        # Grid bounds: trajectory AABB expanded by sensor_buffer
        x_lo = check_xy[:, 0].min() - buf
        x_hi = check_xy[:, 0].max() + buf
        y_lo = check_xy[:, 1].min() - buf
        y_hi = check_xy[:, 1].max() + buf

        xs = np.arange(x_lo, x_hi + gs * 0.5, gs)
        ys = np.arange(y_lo, y_hi + gs * 0.5, gs)
        gx, gy = np.meshgrid(xs, ys)
        grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2)

        # Vectorised min-distance from each grid point to any trajectory point
        # (T, 1, 2) - (1, N, 2) → (T, N) norms → (N,) min over T
        diffs     = check_xy[:, np.newaxis, :] - grid_points[np.newaxis, :, :]
        min_dists = np.linalg.norm(diffs, axis=2).min(axis=0)  # (N,)

        safe_mask   = min_dists >= (self.robot_safe_radius + h)
        safe_points = grid_points[safe_mask]

        half = np.array([h, h], dtype=np.float64)
        return [BoxObstacle(pt, half) for pt in safe_points]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _is_safe(self, obs, check_xy, min_gap):
        dists = obs.batch_min_distance(check_xy)
        return bool(np.all(dists >= min_gap))


# ---------------------------------------------------------------------------
# Multi-mode cycling generator
# ---------------------------------------------------------------------------

class CyclingObstacleGenerator:
    """
    Wraps multiple ObstacleGenerators and cycles through their modes on each
    successive ``generate_for_window`` call.

    This lets training / visualisation see sparse, dense, and compact obstacles
    in a single session without any manual intervention.

    Example
    -------
    >>> gen = CyclingObstacleGenerator(['sparse', 'dense', 'compact'], seed=0)
    >>> # window 0 → sparse, window 1 → dense, window 2 → compact, window 3 → sparse …
    >>> obstacles = gen.generate_for_window(robot_xy)

    Args:
        modes:             List of mode strings (any subset of
                           ``{'sparse', 'dense', 'compact'}``).
        robot_safe_radius: Forwarded to every underlying ObstacleGenerator.
        seed:              Optional base seed; each generator gets ``seed + i``.
        **kwargs:          Additional keyword arguments forwarded to every
                           ObstacleGenerator (e.g. ``max_attempts``).
    """

    def __init__(
        self,
        modes: List[str],
        robot_safe_radius: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if not modes:
            raise ValueError("modes must be a non-empty list")
        for m in modes:
            if m not in ObstacleGenerator.MODES:
                raise ValueError(f"Unknown mode '{m}'. Choose from {ObstacleGenerator.MODES}.")

        self._generators = [
            ObstacleGenerator(
                mode=m,
                robot_safe_radius=robot_safe_radius,
                seed=(seed + i) if seed is not None else None,
                **kwargs,
            )
            for i, m in enumerate(modes)
        ]
        self.modes = list(modes)
        self._call_count = 0

    @property
    def current_mode(self) -> str:
        return self.modes[self._call_count % len(self.modes)]

    def seed(self, s: int):
        """Re-seed all underlying generators."""
        for i, g in enumerate(self._generators):
            g.seed(s + i)

    def generate_for_window(
        self,
        robot_xy: np.ndarray,
        lookahead_xy: Optional[np.ndarray] = None,
    ) -> List[Obstacle2D]:
        """
        Generate obstacles using the next mode in the cycle.

        The mode advances by one on every call, wrapping around when the end
        of the list is reached.
        """
        gen = self._generators[self._call_count % len(self._generators)]
        self._call_count += 1
        return gen.generate_for_window(robot_xy, lookahead_xy)


def make_generator(
    modes: Union[str, List[str]],
    robot_safe_radius: float = 0.5,
    seed: Optional[int] = None,
    **kwargs,
) -> Union[ObstacleGenerator, CyclingObstacleGenerator]:
    """
    Factory: return a plain ObstacleGenerator for a single mode, or a
    CyclingObstacleGenerator when multiple modes are requested.

    Args:
        modes:  A single mode string *or* a list of mode strings.
        robot_safe_radius, seed, **kwargs: forwarded to the generator(s).
    """
    if isinstance(modes, str):
        modes = [modes]
    if len(modes) == 1:
        return ObstacleGenerator(
            mode=modes[0], robot_safe_radius=robot_safe_radius, seed=seed, **kwargs
        )
    return CyclingObstacleGenerator(
        modes=modes, robot_safe_radius=robot_safe_radius, seed=seed, **kwargs
    )


# ---------------------------------------------------------------------------
# Per-clip sensor augmentation (used by the dataset builder)
# ---------------------------------------------------------------------------

def compute_clip_sensor_readings(
    all_qpos: np.ndarray,
    sensor: EnvironmentSensor,
    generator: Union[ObstacleGenerator, CyclingObstacleGenerator],
    obstacle_interval: int = 30,
    lookahead_frames: int = 30,
    use_detour: bool = False,
    cmd_aug: str = "linear",
    cmd_aug_weight: float = 1.0,
    fill_method: str = "band",
    robot_safe_radius: float = 0.25,
    past_frames: int = 10,
    future_frames: int = 45,
) -> Tuple[np.ndarray, List, np.ndarray, np.ndarray]:
    """
    Compute sensor readings for an entire motion clip with time-varying
    obstacles.

    Random obstacles are regenerated every ``obstacle_interval`` frames
    (lookahead = ``lookahead_frames``) so they cannot block the upcoming
    path. When ``use_detour`` is True, an additional set of *scandot-fill*
    obstacles is generated **per frame** from the current robot pose:
    sensor scandots within ``robot_safe_radius`` of the command (yellow)
    trajectory, but at least ``robot_safe_radius`` from the actual (red)
    trajectory, become CircleObstacles. The yellow trajectory is built
    from ``cmd_aug`` / ``cmd_aug_weight`` (see ``visualize/utils/detour.md``).

    Returns
    -------
    readings        : (T, feature_dim) float32 occupancy readings.
    window_obstacles: list of ``(w_start, w_end, random_obstacles)`` for
                      visualisation. Scandot-fill obstacles are not stored
                      here because they are per-frame and would dominate
                      the pkl size.
    traj_trans_per_frame : (T, future_frames, 2) float32. For each frame
                      t, the future trajectory the policy should be
                      conditioned on, expressed in the **robot-local
                      (yaw-frame)** at frame t — translated by ``-p_t``
                      and rotated by ``R(-yaw_t)`` so that frame t sits
                      at the origin facing ``+x``. Detour frames store
                      the yellow command; non-detour frames store the
                      gt (red) trajectory. Padded with the last available
                      value at the tail of the motion.
    traj_pose_per_frame  : (T, future_frames, 4) float32 wxyz quaternion,
                      yaw-only, **relative to ``yaw_t``** (so the current
                      frame's pose is identity). Same yellow/red layout
                      as ``traj_trans_per_frame``.
    """
    # Late import to avoid a circular dependency between
    # `utils.environment_sensor` and `visualize.utils.detour`.
    from visualize.utils.detour import (
        make_command_xy,
        make_command_yaws,
        make_scandot_fill_obstacles,
    )

    T = all_qpos.shape[0]
    readings = np.zeros((T, sensor.feature_dim), dtype=np.float32)
    window_obstacles = []

    traj_trans_per_frame = np.zeros((T, future_frames, 2), dtype=np.float32)
    traj_pose_per_frame  = np.zeros((T, future_frames, 4), dtype=np.float32)
    traj_pose_per_frame[..., 0] = 1.0   # default identity wxyz quaternion

    def _pad_to_future(arr: np.ndarray, n: int) -> np.ndarray:
        """Right-pad ``arr`` with its last row up to length ``n``."""
        if len(arr) >= n:
            return arr[:n]
        if len(arr) == 0:
            return arr  # caller falls back to defaults
        pad = np.repeat(arr[-1:], n - len(arr), axis=0)
        return np.concatenate([arr, pad], axis=0)

    def _yaws_to_wxyz(yaws: np.ndarray) -> np.ndarray:
        out = np.zeros((len(yaws), 4), dtype=np.float32)
        out[:, 0] = np.cos(yaws / 2.0)
        out[:, 3] = np.sin(yaws / 2.0)
        return out

    for w_start in range(0, T, obstacle_interval):
        w_end  = min(w_start + obstacle_interval, T)
        la_end = min(w_end + lookahead_frames, T)

        window_xy    = all_qpos[w_start:w_end, :2]
        lookahead_xy = all_qpos[w_end:la_end, :2] if la_end > w_end else None
        random_obs   = generator.generate_for_window(window_xy, lookahead_xy)
        window_obstacles.append((w_start, w_end, random_obs))

        for t in range(w_start, w_end):
            pos = all_qpos[t, :3]
            yaw = quat_wxyz_to_yaw(all_qpos[t, 3:7])

            # Inverse heading rotation R(-yaw_t): converts world XY into
            # robot-local (yaw-frame) coords at frame t.
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            R_inv = np.array(
                [[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float64,
            )

            obstacles = list(random_obs)

            # gt (red) trajectory for the future horizon. The polyline
            # is computed in world frame (because obstacle generation
            # below operates in world coordinates), then converted to
            # robot-local at frame t for storage.
            red_xy_world   = all_qpos[t : t + future_frames, :2]
            red_yaws_world = np.array(
                [quat_wxyz_to_yaw(q) for q in all_qpos[t : t + future_frames, 3:7]],
                dtype=np.float64,
            )

            red_xy_local   = (R_inv @ (red_xy_world - pos[:2]).T).T
            red_yaws_local = red_yaws_world - yaw
            red_quat_local = _yaws_to_wxyz(red_yaws_local)

            red_xy_padded   = _pad_to_future(red_xy_local,   future_frames)
            red_quat_padded = _pad_to_future(red_quat_local, future_frames)
            if len(red_xy_padded) == future_frames:
                traj_trans_per_frame[t] = red_xy_padded.astype(np.float32)
                traj_pose_per_frame[t]  = red_quat_padded.astype(np.float32)

            if use_detour:
                past_xy = all_qpos[max(0, t - past_frames):t, :2]
                if len(red_xy_world) >= 2:
                    yellow_xy_world = make_command_xy(
                        red_xy_world, cmd_aug, past_xy, weight=cmd_aug_weight
                    )
                    fill_obs, info = make_scandot_fill_obstacles(
                        sensor, pos[:2], yaw, yellow_xy_world, red_xy_world,
                        past_xy=past_xy,
                        robot_safe_radius=robot_safe_radius,
                        fill_method=fill_method,
                    )
                    obstacles += fill_obs

                    # Override storage with yellow values when augmentation
                    # actually placed obstacles; otherwise the gt values
                    # written above remain.
                    if info["has_detour"]:
                        past_yaws = np.array(
                            [quat_wxyz_to_yaw(q)
                             for q in all_qpos[max(0, t - past_frames):t, 3:7]],
                            dtype=np.float64,
                        )
                        yellow_yaws_world = make_command_yaws(
                            red_yaws_world, cmd_aug,
                            past_xy=past_xy, past_yaws=past_yaws,
                            current_yaw=float(yaw),
                            weight=cmd_aug_weight,
                        )

                        yellow_xy_local   = (R_inv @ (yellow_xy_world - pos[:2]).T).T
                        yellow_yaws_local = yellow_yaws_world - yaw
                        yellow_quat_local = _yaws_to_wxyz(yellow_yaws_local)

                        yellow_xy_padded   = _pad_to_future(
                            yellow_xy_local.astype(np.float32), future_frames
                        )
                        yellow_quat_padded = _pad_to_future(
                            yellow_quat_local, future_frames
                        )
                        if len(yellow_xy_padded) == future_frames:
                            traj_trans_per_frame[t] = yellow_xy_padded
                            traj_pose_per_frame[t]  = yellow_quat_padded

            occ, _ = sensor.compute(pos, yaw, obstacles)
            readings[t] = occ

    return (readings, window_obstacles,
            traj_trans_per_frame, traj_pose_per_frame)
