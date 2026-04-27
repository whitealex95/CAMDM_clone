# Detour Obstacle Placement & Command Trajectory Modes

Two obstacle-placement strategies live in `visualize/utils/detour.py`:

1. **`make_scandot_fill_obstacles`** *(used by step2)* — per-frame; fills
   sensor scandots that lie close to the command (yellow) trajectory but
   outside the safety zone of the actual (red) trajectory. See the
   "Scandot-fill obstacle augmentation" section below.

2. **`compute_detour_obstacles`** *(used by step3 demos)* — per-window;
   places up to two circular obstacles in the gap between yellow and red
   so the model has to navigate around them. Documented further down.

Both strategies receive the *yellow* command trajectory from the same
builder, `make_command_xy(actual_xy, cmd_aug, past_xy, weight)`.

## Common notation

The function signature is `make_command_xy(actual_xy, cmd_aug, past_xy)`.

| Symbol | Meaning | Source | Default value |
|--------|---------|--------|---------------|
| $P$ | number of past frames available  | `len(past_xy)`            | `--past-frames` = **10** |
| $N$ | length of the obstacle window    | `len(actual_xy)`          | `obstacle_interval + future_frames` = 30 + 45 = **75** |
| $\mathbf{p}_i \in \mathbb{R}^2$ | past XY, oldest first | `past_xy[i]`,  $i = 0, \dots, P-1$ | — |
| $\mathbf{a}_i \in \mathbb{R}^2$ | actual XY in the window | `actual_xy[i]`, $i = 0, \dots, N-1$ | — |
| $\mathbf{c}_i \in \mathbb{R}^2$ | command XY (output) | `command_xy[i]`, $i = 0, \dots, N-1$ | — |

The current robot position is $\mathbf{a}_0$ (= the first frame of the
obstacle window). All command trajectories satisfy $\mathbf{c}_0 = \mathbf{a}_0$
so the command starts wherever the robot currently is. By construction the
last past frame $\mathbf{p}_{P-1}$ is the frame immediately preceding
$\mathbf{a}_0$ in the source data.

## 1. `linear` — straight start→end

Linear interpolation from the window start to the window end of the actual
trajectory. This is the legacy behaviour and the only one independent of past
state.

$$
\mathbf{c}_i = \mathbf{a}_0 + \frac{i}{N-1}\bigl(\mathbf{a}_{N-1} - \mathbf{a}_0\bigr),
\quad i = 0, 1, \dots, N-1
$$

**Use it when**: you want the policy to learn "go from A to B as the crow
flies, deal with whatever's in between."

## 2. `extrap_pos` — constant-velocity extrapolation of past

Average per-frame velocity over the past window, then extend in a straight
line from the current position.

$$
\mathbf{v} = \frac{\mathbf{p}_{P-1} - \mathbf{p}_0}{P - 1}
$$

$$
\mathbf{c}_i = \mathbf{a}_0 + i\,\mathbf{v},
\quad i = 0, 1, \dots, N-1
$$

Falls back to `linear` if $P < 2$.

**Properties**:
- Always a straight line, but oriented along *past* velocity rather than the
  chord to the goal.
- Magnitude of $\mathbf{v}$ is the average past speed — slow walking yields a
  short command, fast running yields a long one.
- Acceleration, deceleration, and turning are entirely ignored.

**Use it when**: you want a "naive sim-to-real" command — the kind of
trajectory a downstream controller would produce by integrating its current
velocity estimate forward.

> **Note on orientation (visualiser-only).** `make_command_xy` returns XY only.
> The companion `_cmd_aug_extrap_pos` extrapolates the past yaw sequence at a
> *constant rate*, independent of position direction (no assumption that the
> robot faces its velocity). Using past quaternions converted to yaw and
> unwrapped:
> $\dot\theta = (\theta_{P-1} - \theta_0)/(P-1)$,
> $\theta_i = \theta_{P-1} + (i+1)\,\dot\theta$ for $i = 0, \dots, N-1$.

## 2a. `extrap_pos_noyaw` — same XY, yaw locked to motion direction

Identical XY to `extrap_pos`. Orientation is **not** taken from past yaw
rate; instead it linearly interpolates from the current heading
$\theta_0 = \mathrm{yaw}(\text{future\_orient}_0)$ at frame 0 to the
extrapolated motion direction $\theta_v = \arctan2(v_y, v_x)$ at frame $N-1$,
along the shortest signed path:

$$
\Delta\theta = (\theta_v - \theta_0 + \pi) \bmod 2\pi - \pi \in (-\pi, \pi]
$$

$$
\theta_i = \theta_0 + \frac{i}{N-1}\,\Delta\theta,
\quad i = 0, \dots, N-1.
$$

So $\theta_0$ matches the gt current frame (no teleport at frame 0) and
$\theta_{N-1} = \theta_v$ (robot faces its motion direction by the end).

**Use it when**: the robot should "settle" into facing its motion direction
within the horizon (e.g. mid-stride re-orientation), but past yaw history
isn't reliable enough to extrapolate as a rate.

## 2b. `extrap_pos_preserve_len` — past direction, gt step lengths

Same direction as `extrap_pos`, but the per-frame distances are taken from
the actual trajectory so the cumulative path length matches gt. The cmd
becomes a perfectly straight line in the past direction whose total length
equals the gt curve's arc length.

$$
\hat{\mathbf{d}} = \frac{\mathbf{v}}{\lVert \mathbf{v} \rVert},
\qquad
d_i = \lVert \mathbf{a}_{i+1} - \mathbf{a}_i \rVert,
\qquad
s_i = \sum_{k=0}^{i-1} d_k
$$

$$
\mathbf{c}_i = \mathbf{a}_0 + s_i\,\hat{\mathbf{d}},
\quad i = 0, 1, \dots, N-1
$$

so $\mathbf{c}_0 = \mathbf{a}_0$, $\mathbf{c}_{i+1} - \mathbf{c}_i$ has the
same magnitude as $\mathbf{a}_{i+1} - \mathbf{a}_i$, and the entire cmd lies
on the ray from $\mathbf{a}_0$ along $\hat{\mathbf{d}}$.

Falls back to `linear` if $P < 2$ or $\lVert \mathbf{v} \rVert < 10^{-9}$
(stationary past, no direction defined).

**Use it when**: you want to preserve the speed profile (e.g. acceleration,
deceleration, foot-step rhythm) of the actual motion but force the path
to be straight in the past heading.

The visualiser uses the same constant-rate yaw extrapolation as
`extrap_pos` for orientation.

## 3. `extrap_hfte` — HFTE central-symmetry extension

Heuristic Future Trajectory Extension (CAMDM baseline), applied to the past
trajectory rather than the model's prediction. Extends the past via two
nested point reflections so that local curvature trends are preserved.

### Seed

Concatenate the $P$ past frames with the current position $\mathbf{a}_0$
(which equals `actual_xy[0]`, the first frame of the obstacle window):

$$
\mathbf{s} = [\,\mathbf{p}_0,\; \mathbf{p}_1,\; \dots,\; \mathbf{p}_{P-1},\; \mathbf{a}_0\,]
$$

with seed length

$$
L = P + 1.
$$

With the default `--past-frames` = 10, this gives $L = 11$ — ten past frames
followed by the current robot position.

### Single HFTE pass

Given a sequence $\mathbf{s}$ of length $L \geq 2$, build $L-1$ extension
points $\mathbf{e}$:

**Stage 1 — central symmetry around the anchor $\mathbf{s}_{L-1}$:**

$$
\mathbf{e}_j = 2\,\mathbf{s}_{L-1} - \mathbf{s}_{L-2-j},
\quad j = 0, 1, \dots, L-2
$$

Geometrically: reflect each prior point across the latest point. $\mathbf{e}_0$
is the constant-velocity successor; $\mathbf{e}_1$ accounts for one frame of
acceleration; and so on.

**Stage 2 — second reflection around the midpoint $m = \lfloor (L-1)/2 \rfloor$
of stage 1, applied to the second half only:**

$$
\mathbf{e}_j \leftarrow 2\,\mathbf{e}_m - \mathbf{e}_{2m-j},
\quad j = m+1, \dots, L-2
$$

This second reflection bends the tail of the extension back, so a turning
trend keeps turning instead of continuing straight forever.

The pass output is $[\mathbf{s}, \mathbf{e}]$ of length $2L-1$.

### Iteration

Each pass roughly doubles the sequence length ($L \to 2L-1$). The pass is
repeated until the sequence reaches the required total length:

$$
T_{\text{total}} = L + (N - 1) = P + N.
$$

The first $L$ entries of the result equal $\mathbf{s}$; the last $N$ entries
form $\mathbf{c}_i$ and start with $\mathbf{a}_0$.

With the defaults $P = 10$, $N = 75$:

| pass | length |
|------|--------|
| 0 (seed)  | $L = 11$ |
| 1 | $2L-1 = 21$ |
| 2 | $41$ |
| 3 | $81$ |
| 4 | $\geq T_{\text{total}} = 85$ ✓ |

So four passes are needed; the last $N = 75$ entries of the final 161-length
sequence are returned as the command.

```python
ext_traj, _ = extend_future_traj_heusristic(seed, dummy_orient, total=P + N)
command_xy = ext_traj[-N:]
```

Falls back to `linear` if $P < 1$.

**Properties**:
- Preserves both speed and curvature of the past window.
- A turning past produces a *curved* command, unlike `extrap_pos`.
- Discontinuities at boundaries of HFTE passes are smooth in practice
  because each pass concatenates point-reflected segments.
- Same algorithm used for `extend_future_traj_heusristic` in step3 demos to
  pad short model predictions, so command and inference share extrapolation
  semantics.

**Use it when**: past motion exhibits a clear turning trend that should
continue (e.g. circling, weaving) and a straight extrapolation would
underestimate the curve.

> **Note on orientation (visualiser-only).** `make_command_xy` returns XY
> only. The companion `_cmd_aug_extrap_hfte` runs the same HFTE algorithm on
> the past *yaw* sequence (independent of position). The seed
> $\mathbf{s}_\theta = [\theta_0, \dots, \theta_{P-1}, \theta_{\text{now}}]$
> is unwrapped with `np.unwrap` so $\pm\pi$ jumps don't break the central
> reflections, then extended to total length $P + N$; the last $N$ entries
> are converted back to wxyz quaternions. This way back-stepping or
> side-stepping motions (where heading $\neq$ velocity direction) still
> produce a meaningful orientation extrapolation.

## Comparison

| Mode | Input | Shape | Speed | Curvature |
|------|-------|-------|-------|-----------|
| `linear`                  | `actual_xy` only         | straight | depends on goal distance     | none |
| `extrap_pos`              | `past_xy` (P ≥ 2)        | straight | matches past avg speed       | none |
| `extrap_pos_noyaw`        | `past_xy` (P ≥ 2)        | straight | matches past avg speed       | none (XY same as extrap_pos; yaw differs) |
| `extrap_pos_preserve_len` | `past_xy` (P ≥ 2) + gt step lengths | straight | per-frame matches gt (total path length matches gt) | none |
| `extrap_hfte`             | `past_xy` (P ≥ 1) + a_0  | curved   | matches past avg speed       | preserved from past |

For both `extrap_pos` and `extrap_hfte`, the obstacle generator places
detour circles in the gap between *that* command and the actual trajectory,
so the policy is challenged on the augmentation it's actually being asked
to follow.

## 4. Blending with ground truth: `--cmd-aug-weight`

Each mode above produces an *extrap* trajectory. The final command is a
linear blend with the dataset (gt) trajectory controlled by a scalar
weight $w \in [0, 1]$:

$$
\mathbf{c}_i = w \,\mathbf{c}^{\text{extrap}}_i + (1 - w)\,\mathbf{a}_i
$$

- $w = 1.0$ (default) — pure extrap, ignores the gt curve in between.
- $w = 0.0$           — pure gt, command equals the actual trajectory.
- intermediate         — interpolated path; useful for sweeping how strongly
                         the policy should be pulled away from the gt motion.

For the visualiser, the same weight is also applied to yaw via the shortest
signed path with cross-frame unwrap:

$$
\theta_i^{\text{blended}} = \theta_i^{\text{gt}} + w\,\Delta\theta_i,
\qquad
\Delta\theta_i = \mathrm{unwrap}\!\left[(\theta_i^{\text{cmd}} - \theta_i^{\text{gt}} + \pi) \bmod 2\pi - \pi\right]
$$

so the lerp doesn't flip direction when the cmd crosses $\pm\pi$.


# Scandot-fill obstacle augmentation

Used by `step2_visualize_data_env2d.py` and `step2_debug_2d.py`. Replaces
the legacy two-circle detour with a per-frame, sensor-aligned filling
scheme.

## Inputs (per frame)

- `sensor` — `EnvironmentSensor`. Provides scandot positions and
  `sensor.max_adjacent_distance` (precomputed at construction).
- `robot_pos`, `robot_yaw` — current robot pose. Scandots are placed
  relative to this pose; obstacles regenerate every frame
  ($\text{window} = 1$).
- `command_xy` — yellow trajectory over the future horizon
  ($N$ frames), built from `make_command_xy(actual, cmd_aug, past)`.
- `actual_xy` — red (gt) trajectory, $N$ frames.
- `robot_safe_radius` — single threshold reused for both the
  "close to yellow" inclusion and the "safe distance from red"
  exclusion.

## Filter

Two strategies are selectable via `fill_method`. In both cases
$d(\cdot, \text{polyline})$ is the minimum point-to-segment distance
over the polyline.

### `band` (default)

A per-scandot distance band — currently the red-safety constraint only,
since the historical "close to yellow" inclusion has been disabled
experimentally:

$$
\text{fill}(s) \;=\;
  d(s,\,\text{red}) \geq r_{\text{safe}}
$$

(Re-enable the yellow constraint by uncommenting the
`d(s, yellow) < r_safe` line in `make_scandot_fill_obstacles`.)

This produces dense fill: every scandot outside the red safety zone
becomes an obstacle.

### `yellow_circle`

Not scandot-based: place one ``CircleObstacle`` *at every yellow arrow
point*, each with its own *largest* radius such that the circle stays
at least $r_{\text{safe}}$ away from the red polyline.

Per-point radius (variable across the trajectory):

$$
r_i = \max\!\big(d(\mathbf{y}_i,\,\text{red}) - r_{\text{safe}},\; 0\big)
$$

Yellow points whose $r_i$ falls below `min_radius` (= 0.05 m) are
dropped. This naturally trims the boundary points: linear and
`extrap_pos*` trajectories share endpoints with red, so
$d(\mathbf{y}_i, \text{red}) = 0$ near the start/end and the resulting
$r_i$ is below the threshold there. If *every* point fails the test,
no obstacles are produced at all.

Past exclusion uses the geometrically correct check
$d(\mathbf{c}, \text{past}) - r \geq r_{\text{safe}}$ so a circle's
*outer edge* — not just its centre — stays clear of the past safety
zone. (The scandot `band` method uses the simpler centre-only check
because each scandot obstacle is small.)

### Past exclusion (both methods)

When `past_xy` (blue) is supplied, scandots within $r_{\text{safe}}$
of the past polyline are removed from the fill mask regardless of
method, so obstacles never appear behind the robot where it has just
walked.

A "no detour" early-out runs first: if
$\max_i \lVert \text{yellow}_i - \text{red}_i \rVert$ is below
`_DETOUR_DEVIATION_MIN` (= 0.10 m) the function returns no obstacles
and `info["has_detour"] = False`.

## Obstacle radius

Each filled scandot becomes a `CircleObstacle(s, r_{\text{obs}})` with

$$
r_{\text{obs}} = \tfrac{1}{2}\,d_{\text{adj}}^{\max}
$$

where $d_{\text{adj}}^{\max}$ is the precomputed max nearest-neighbour
distance between any two scandots. With this radius, two scandots that
are at the maximum nearest-neighbour distance produce circles that just
touch — so a fully-filled region has no gaps between adjacent
obstacles.

For the cylindrical sensor with `max_range=2.0m`, `resolution=9`, this
gives $d_{\text{adj}}^{\max} \approx 0.253$ m, hence
$r_{\text{obs}} \approx 0.126$ m. Inner rings have heavy overlap (their
neighbours are much closer than the outer rings); outer-ring obstacles
just touch their neighbours. Sensor occupancy is unchanged by overlap,
so the redundancy is harmless.

## Per-frame regeneration

Unlike the legacy two-circle detour (which was per `obstacle_interval`),
scandot-fill regenerates every frame because:

- The robot pose changes every frame, so its scandot grid moves.
- The yellow / red trajectories also slide forward by one frame.

Random environment obstacles (`sparse` / `dense` / `compact`) are still
cached per `obstacle_interval`. The visualiser combines them as
`obstacles = random_obstacles + scandot_fill_obstacles` per frame.

## What gets stored in the dataset

`compute_clip_sensor_readings` (in `utils/environment_sensor.py`) calls
`make_scandot_fill_obstacles` per frame and stores

- `sensor_readings`: $(T, \text{feature\_dim})$ occupancy from the
  combined obstacle set;
- `command_detour_flags`: $(T,)$ boolean = `info["has_detour"]` per
  frame.

Per-frame scandot-fill obstacle lists are **not** stored in the pkl —
they would dominate file size and they are reproducible from
`(qpos, cmd_aug, robot_safe_radius, sensor)` anyway.
