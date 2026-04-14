"""
Signed-Distance-Function guidance for diffusion sampling.

Provides `make_obstacle_cond_fn`, a factory that returns a cond_fn compatible
with GaussianDiffusion.p_sample_loop / ddim_sample_loop when called with
    cond_fn=cond_fn, cond_fn_with_grad=True

Theory
------
Classifier guidance (Dhariwal & Nichol 2021) modifies the reverse-diffusion
mean by adding  variance * ∇_x log p(y | x_t).

Here the "classifier" is an obstacle-clearance energy:

    E(x̂₀) = Σ_t  max(0,  margin − SDF_min(x̂₀_t))²

where x̂₀ is the model's predicted denoised sample (pred_xstart), and
SDF_min is the minimum signed distance from the predicted XY position to any
BoxObstacle (positive = outside walls, negative = penetrating).

The guidance gradient is ∇_x (−E), so the reverse step is biased toward
trajectories that stay at least `margin` metres from every obstacle.

Gradient path
-------------
    x_t  (requires_grad=True)
      ↓  p_mean_variance  (model forward)
    pred_xstart  = (x_t − √(1−ᾱ)·ε_θ(x_t)) / √ᾱ
      ↓  extract positions
    xy_world  = pred_xstart[:, 30, :2, :] + curr_xy
      ↓  batched box SDF
    energy  = Σ clamp(margin − SDF, 0)²
      ↓  autograd.grad
    grad w.r.t. x_t

Usage
-----
    cond_fn = make_obstacle_cond_fn(obstacles, curr_xy, device,
                                    margin=0.3, scale=3.0)
    out = diffusion.p_sample_loop(
        model, shape, ...,
        cond_fn=cond_fn, cond_fn_with_grad=True,
    )
"""

import torch


def make_obstacle_cond_fn(
    obstacles,
    curr_xy,
    device,
    margin: float = 0.3,
    scale: float = 1.0,
):
    """
    Build a reconstruction-guidance cond_fn for obstacle avoidance.

    Parameters
    ----------
    obstacles : list of BoxObstacle
        Static obstacles in the scene.  Only ``center``, ``half_extents``,
        and ``yaw`` attributes are used.
    curr_xy : array-like (2,)
        World XY of the robot at this generation step.  The diffusion model
        works in a root-relative frame, so we add curr_xy to convert predicted
        positions back to world coordinates before computing distances.
    device : torch.device
    margin : float
        Minimum clearance in metres.  The hinge loss activates when any
        predicted waypoint is closer than this value to an obstacle.
    scale : float
        Guidance strength.  Analogous to the classifier-guidance scale γ:
        larger values push the trajectory harder away from obstacles but may
        reduce motion quality.  Typical useful range: 1–10.

    Returns
    -------
    cond_fn : callable  (x, t, p_mean_var, **kwargs) → gradient tensor
        Suitable for GaussianDiffusion.*_with_grad sampling functions.
    """
    # ── Pre-build obstacle tensors (evaluated once per generation call) ──
    centers  = torch.tensor(
        [[float(o.center[0]), float(o.center[1])] for o in obstacles],
        dtype=torch.float32, device=device,
    )  # (N, 2)
    half_ext = torch.tensor(
        [[float(o.half_extents[0]), float(o.half_extents[1])] for o in obstacles],
        dtype=torch.float32, device=device,
    )  # (N, 2)
    yaws_t   = torch.tensor(
        [float(o.yaw) for o in obstacles],
        dtype=torch.float32, device=device,
    )  # (N,)
    curr_xy_t = torch.tensor(
        [float(curr_xy[0]), float(curr_xy[1])],
        dtype=torch.float32, device=device,
    )  # (2,)

    cos_y = torch.cos(yaws_t)  # (N,)
    sin_y = torch.sin(yaws_t)  # (N,)

    def _min_box_sdf(xy_world: torch.Tensor) -> torch.Tensor:
        """
        Minimum signed distance to any obstacle for each predicted waypoint.

        Parameters
        ----------
        xy_world : (B, 2, T)

        Returns
        -------
        sdf_min : (B, T)
            Positive  → outside all obstacles.
            Negative  → inside at least one obstacle.
        """
        B, _, T = xy_world.shape
        N = centers.shape[0]

        # Relative position to each obstacle centre
        # (B, 1, 2, T) − (1, N, 2, 1)  →  (B, N, 2, T)
        rel = xy_world.unsqueeze(1) - centers.view(1, N, 2, 1)

        # Rotate into box-local frame  (R⁻¹ = [[cos, sin], [−sin, cos]])
        lx = ( cos_y.view(1, N, 1) * rel[:, :, 0, :]
              + sin_y.view(1, N, 1) * rel[:, :, 1, :])   # (B, N, T)
        ly = (-sin_y.view(1, N, 1) * rel[:, :, 0, :]
              + cos_y.view(1, N, 1) * rel[:, :, 1, :])   # (B, N, T)

        hx = half_ext[:, 0].view(1, N, 1)   # (1, N, 1)
        hy = half_ext[:, 1].view(1, N, 1)

        dx = lx.abs() - hx   # (B, N, T)  positive outside box in x
        dy = ly.abs() - hy

        # Standard box SDF: positive outside, negative inside
        outside = (torch.clamp(dx, min=0.0) ** 2
                   + torch.clamp(dy, min=0.0) ** 2).sqrt()
        inside  = torch.clamp(torch.maximum(dx, dy), max=0.0)
        sdf     = outside + inside   # (B, N, T)

        return sdf.min(dim=1).values   # (B, T) – closest obstacle

    def cond_fn(x, t, p_mean_var, **kwargs):
        """
        Compute  ∇_x log p(safe | x_t)  for obstacle avoidance.

        Parameters
        ----------
        x          : (B, 31, 6, T)  noisy sample with requires_grad=True
        t          : timestep tensor
        p_mean_var : dict from p_mean_variance(); must contain 'pred_xstart'
        **kwargs   : ignored extra model conditioning inputs

        Returns
        -------
        gradient : tensor matching x.shape
            Added to the denoising mean as  new_mean += variance * gradient.
        """
        pred_xstart = p_mean_var["pred_xstart"]   # (B, 31, 6, T)

        # Joint slot 30, features 0–1 = relative XY (see qpos_to_model_format)
        xy_rel   = pred_xstart[:, 30, :2, :]              # (B, 2, T)
        xy_world = xy_rel + curr_xy_t.view(1, 2, 1)       # (B, 2, T) world coords

        sdf_min = _min_box_sdf(xy_world)                   # (B, T)

        # Hinge loss: squared penalty for any point inside the safety margin
        violation = torch.clamp(margin - sdf_min, min=0.0) ** 2
        energy    = violation.sum()

        # ∇_x (−E)  →  guidance nudges denoising away from obstacles
        (grad,) = torch.autograd.grad(energy, x)
        return -scale * grad

    return cond_fn
