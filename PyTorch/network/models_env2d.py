"""
MotionDiffusionEnv – motion diffusion model with environment-sensor conditioning.

Extends MotionDiffusion (models.py) by adding an NSM-style EnvSensorEncoder for
the cylindrical occupancy sensor readings.  The sensor embedding is concatenated
into the transformer sequence alongside the trajectory embeddings.

Architecture reference:
    Starke et al. "Neural State Machine for Character-Scene Interaction"
    SIGGRAPH Asia 2019 — environment encoder: MLP [env_sensor_dim → hidden → latent]
    with ELU activations.

Sensor condition tensor shape at inference time:
    sensor: (batch, env_sensor_dim)  – current-frame snapshot
"""

import torch
import torch.nn as nn
from network.models import MotionDiffusion


class EnvSensorEncoder(nn.Module):
    """
    NSM-style MLP encoder for cylindrical environment-sensor readings.

    Maps per-frame sensor feature vectors to latent vectors via a 2-layer MLP
    with ELU activations, matching the environment encoder in the NSM paper.

    Args:
        env_sensor_dim: Number of sensor occupancy values per frame
                        (= EnvironmentSensor.feature_dim).
        latent_dim:     Output dimension (same as transformer latent_dim).
        hidden_dims:    Hidden layer sizes. Default [512] matches NSM paper.
    """

    def __init__(self, env_sensor_dim: int, latent_dim: int, hidden_dims: list = [512]):
        super().__init__()
        dims = [env_sensor_dim] + list(hidden_dims) + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, env_sensor_dim) – current-frame sensor snapshot.

        Returns:
            (1, bs, latent_dim) – single environment token,
            ready to concat into the transformer input sequence.
        """
        return self.net(x).unsqueeze(0)  # (1, bs, latent_dim)


class MotionDiffusionEnv(MotionDiffusion):
    """
    Motion diffusion model conditioned on 2-D cylindrical environment-sensor readings.

    Identical to MotionDiffusion except for an extra ``sensor_encoder`` (NSM-style
    2-layer MLP with ELU) that embeds the per-frame occupancy readings into the
    latent space.

    Args (additions to MotionDiffusion):
        env_sensor_dim: Number of sensor occupancy values (= EnvironmentSensor.feature_dim).
    """

    def __init__(self, input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
                 env_sensor_dim: int = 226,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4,
                 dropout=0.2, ablation=None, activation="gelu", legacy=False,
                 arch='trans_enc', cond_mask_prob=0, sensor_cond_mask_prob=0, device=None):

        super().__init__(
            input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, ablation=ablation,
            activation=activation, legacy=legacy, arch=arch,
            cond_mask_prob=cond_mask_prob, device=device,
        )

        self.env_sensor_dim        = env_sensor_dim
        self.sensor_cond_mask_prob = float(sensor_cond_mask_prob)
        # NSM-style environment encoder: 2-layer MLP with ELU
        self.sensor_encoder = EnvSensorEncoder(env_sensor_dim, self.latent_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, style_idx, sensor=None):
        """
        Args:
            sensor: (batch, env_sensor_dim) float tensor – current-frame snapshot, or None.
                    When None the sensor embedding is zeroed out (unconditional).
        """
        bs, njoints, nfeats, nframes = x.shape

        time_emb        = self.embed_timestep(timesteps)               # (1, bs, L)
        style_emb       = self.embed_style(style_idx).unsqueeze(0)     # (1, bs, L)
        traj_trans_emb  = self.traj_trans_process(traj_trans)          # (TF, bs, L)
        traj_pose_emb   = self.traj_pose_process(traj_pose)            # (TF, bs, L)
        past_motion_emb = self.past_motion_process(past_motion)        # (TP, bs, L)
        future_motion_emb = self.future_motion_process(x)              # (TF, bs, L)

        if sensor is not None:
            sensor_emb = self.sensor_encoder(sensor)                   # (1, bs, L)
        else:
            sensor_emb = torch.zeros(1, bs, self.latent_dim, device=x.device, dtype=x.dtype)

        xseq = torch.cat((
            time_emb, style_emb,
            traj_trans_emb, traj_pose_emb, sensor_emb,
            past_motion_emb, future_motion_emb,
        ), dim=0)

        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:]
        output = self.output_process(output)
        return output

    # ------------------------------------------------------------------
    # Interface (called by training loop and sampler)
    # ------------------------------------------------------------------

    def interface(self, x, timesteps, y=None):
        """
        y keys:
            past_motion: (bs, njoints, nfeats, past_frames)
            traj_pose:   (bs, 6, future_frames)
            traj_trans:  (bs, 2, future_frames)
            style_idx:   (bs,)
            sensor:      (bs, env_sensor_dim)  – current-frame snapshot, optional
        """
        bs = x.shape[0]

        style_idx   = y['style_idx']
        past_motion = y['past_motion']
        traj_pose   = y['traj_pose']
        traj_trans  = y['traj_trans']
        sensor      = y.get('sensor', None)

        # CFG masking on past_motion
        keep = torch.rand(bs, device=past_motion.device) < (1.0 - self.cond_mask_prob)
        past_motion = past_motion * keep.view(bs, 1, 1, 1)

        # CFG masking on sensor
        if sensor is not None and self.sensor_cond_mask_prob > 0:
            keep_s = torch.rand(bs, device=sensor.device) < (1.0 - self.sensor_cond_mask_prob)
            sensor = sensor * keep_s.view(bs, 1)

        return self.forward(x, timesteps, past_motion, traj_pose, traj_trans, style_idx, sensor)
