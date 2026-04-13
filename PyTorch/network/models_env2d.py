"""
MotionDiffusionEnv – motion diffusion model with environment-sensor conditioning.

Extends MotionDiffusion (models.py) by adding a TrajProcess for the binary
scan-dot sensor readings.  The sensor embedding is concatenated into the
transformer sequence alongside the trajectory embeddings.

Sensor condition tensor shape at inference time:
    sensor: (batch, n_rays, future_frames)
"""

import torch
from network.models import MotionDiffusion, TrajProcess


class MotionDiffusionEnv(MotionDiffusion):
    """
    Motion diffusion model conditioned on 2-D environment-sensor readings.

    Identical to MotionDiffusion except for an extra ``sensor_process`` that
    embeds the per-frame binary scan-dot readings into the latent space.

    Args (additions to MotionDiffusion):
        sensor_dim: Number of sensor rays (= EnvironmentSensor.n_rays).
    """

    def __init__(self, input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
                 sensor_dim: int = 36,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4,
                 dropout=0.2, ablation=None, activation="gelu", legacy=False,
                 arch='trans_enc', cond_mask_prob=0, device=None):

        super().__init__(
            input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, ablation=ablation,
            activation=activation, legacy=legacy, arch=arch,
            cond_mask_prob=cond_mask_prob, device=device,
        )

        self.sensor_dim = sensor_dim
        # Sensor readings share the same TrajProcess embedding structure as traj_trans/traj_pose
        self.sensor_process = TrajProcess(sensor_dim, self.latent_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, style_idx, sensor=None):
        """
        Args:
            sensor: (batch, sensor_dim, future_frames) float tensor, or None.
                    When None the sensor embedding is zeroed out (unconditional).
        """
        bs, njoints, nfeats, nframes = x.shape

        time_emb        = self.embed_timestep(timesteps)               # (1, bs, L)
        style_emb       = self.embed_style(style_idx).unsqueeze(0)     # (1, bs, L)
        traj_trans_emb  = self.traj_trans_process(traj_trans)          # (N, bs, L)
        traj_pose_emb   = self.traj_pose_process(traj_pose)            # (N, bs, L)
        past_motion_emb = self.past_motion_process(past_motion)        # (past, bs, L)
        future_motion_emb = self.future_motion_process(x)              # (future, bs, L)

        if sensor is not None:
            sensor_emb = self.sensor_process(sensor)                   # (N, bs, L)
        else:
            sensor_emb = torch.zeros_like(traj_trans_emb)

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
            sensor:      (bs, sensor_dim, future_frames)  – optional
        """
        bs = x.shape[0]

        style_idx   = y['style_idx']
        past_motion = y['past_motion']
        traj_pose   = y['traj_pose']
        traj_trans  = y['traj_trans']
        sensor      = y.get('sensor', None)

        # CFG masking on past_motion only (matching base class behaviour)
        keep = torch.rand(bs, device=past_motion.device) < (1.0 - self.cond_mask_prob)
        past_motion = past_motion * keep.view(bs, 1, 1, 1)

        return self.forward(x, timesteps, past_motion, traj_pose, traj_trans, style_idx, sensor)
