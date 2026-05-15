"""
MotionDiffusionDipTraj – DiP-style (CLoSD diffusion_planner) motion diffusion
adapted for CAMDM G1 humanoid with environment-sensor conditioning instead
of CLIP/BERT text. This is the "per-frame traj added INTO the future motion
tokens" variant (compact layout, single fused conditioning token).

Architecture mirrors closd/diffusion_planner/model/mdm.py::MDM but:
  * Drops the text encoder branch entirely; the only "context" input is
    the sensor reading (current-frame snapshot, 226-D by default).
  * Drops keyframe_cond_type, multi_target_cond, GRU branch — keeps just
    trans_enc and trans_dec.
  * Replaces DiP's free-running prefix mechanic with CAMDM's explicit
    past_motion conditioning: past and noisy future are concatenated into a
    single per-frame token sequence (DiP prefix-completion style), then the
    model outputs only the last TF frames.
  * Adds CAMDM's structured conditioning (style index, per-frame
    trajectory) on top: style is folded into the single context token (DiP
    "emb_policy=add"); traj is projected per-frame and added to the
    future-frame motion tokens.

Conditioning tensor at inference time (passed via the ``y`` dict that the
training portal builds from the DataLoader batch):

    past_motion : (bs, J, F, TP)       past motion frames
    traj_pose   : (bs, 6,    TF)       per-frame yaw-frame heading
    traj_trans  : (bs, 2,    TF)       per-frame yaw-frame XY translation
    style_idx   : (bs,)                style/action index
    sensor      : (bs, env_sensor_dim) current-frame occupancy snapshot
    mask        : (bs, TF) or (TF,)    per-future-frame validity (1 valid)
"""

import torch
import torch.nn as nn

from network.models import (
    MotionProcess, TrajProcess, PositionalEncoding,
    TimestepEmbedder, OutputProcess, EmbedStyle,
)
from network.models_env2d import EnvSensorEncoder


class MotionDiffusionDipTraj(nn.Module):

    def __init__(self, input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
                 env_sensor_dim: int = 226,
                 past_frame: int = 10, future_frame: int = 45,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 4,
                 dropout: float = 0.2, activation: str = "gelu",
                 arch: str = 'trans_enc',
                 cond_mask_prob: float = 0.0,
                 sensor_cond_mask_prob: float = 0.0,
                 mask_frames: bool = False,
                 device=None):
        super().__init__()

        self.training = True
        self.rot_req = rot_req
        self.nfeats = nfeats
        self.njoints = njoints
        self.clip_len = clip_len
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.arch = arch
        self.cond_mask_prob = float(cond_mask_prob)
        self.sensor_cond_mask_prob = float(sensor_cond_mask_prob)
        self.env_sensor_dim = env_sensor_dim
        self.past_frame = past_frame
        self.future_frame = future_frame
        self.mask_frames = mask_frames

        # Shared per-frame motion projector for past + noisy future
        self.motion_process = MotionProcess(self.input_feats, self.latent_dim)

        # Per-frame trajectory conditioning (added to future motion tokens)
        self.traj_trans_process = TrajProcess(2, self.latent_dim)
        self.traj_pose_process  = TrajProcess(6, self.latent_dim)

        # NSM-style scene encoder (same as MotionDiffusionEnv)
        self.sensor_encoder = EnvSensorEncoder(env_sensor_dim, self.latent_dim)

        # Time/style/positional embedders
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep       = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_style          = EmbedStyle(nstyles, self.latent_dim)

        # DiP-style backbone
        if self.arch == 'trans_enc':
            print("DipTraj TRANS_ENC init")
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim, nhead=self.num_heads,
                dim_feedforward=self.ff_size, dropout=self.dropout,
                activation=self.activation,
            )
            self.seqEncoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("DipTraj TRANS_DEC init")
            dec_layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim, nhead=self.num_heads,
                dim_feedforward=self.ff_size, dropout=self.dropout,
                activation=self.activation,
            )
            self.seqEncoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_layers)
        else:
            raise ValueError(f"DiP model supports [trans_enc, trans_dec]; got '{arch}'")

        self.output_process = OutputProcess(self.input_feats, self.latent_dim,
                                            self.njoints, self.nfeats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, style_idx,
                sensor=None, frames_mask=None):
        """
        Args:
            x:           (bs, J, F, TF) noisy future motion
            timesteps:   (bs,) int
            past_motion: (bs, J, F, TP)
            traj_pose:   (bs, 6, TF)
            traj_trans:  (bs, 2, TF)
            style_idx:   (bs,)
            sensor:      (bs, env_sensor_dim) or None
            frames_mask: (bs, TP+TF) bool, True = pad. Optional.

        Returns:
            (bs, J, F, TF) predicted denoised future motion
        """
        bs, njoints, nfeats, nframes = x.shape
        TP = past_motion.shape[-1]

        # ---------- single context token: time + style + sensor ----------
        time_emb  = self.embed_timestep(timesteps)              # (1, bs, L)
        style_emb = self.embed_style(style_idx).unsqueeze(0)    # (1, bs, L)
        if sensor is not None:
            sensor_emb = self.sensor_encoder(sensor)            # (1, bs, L)
        else:
            sensor_emb = torch.zeros(
                1, bs, self.latent_dim, device=x.device, dtype=x.dtype
            )
        cond_emb = time_emb + style_emb + sensor_emb            # (1, bs, L)

        # ---------- prefix-completion: past + noisy future as one seq ----
        combined = torch.cat([past_motion, x], dim=-1)          # (bs, J, F, TP+TF)
        motion_seq = self.motion_process(combined)              # (TP+TF, bs, L)

        # Per-frame trajectory added to the *future* portion only
        traj_trans_emb = self.traj_trans_process(traj_trans)    # (TF, bs, L)
        traj_pose_emb  = self.traj_pose_process(traj_pose)      # (TF, bs, L)
        traj_emb       = traj_trans_emb + traj_pose_emb         # (TF, bs, L)

        past_emb   = motion_seq[:TP]                            # (TP, bs, L)
        future_emb = motion_seq[TP:] + traj_emb                 # (TF, bs, L)
        motion_seq = torch.cat([past_emb, future_emb], dim=0)   # (TP+TF, bs, L)

        # ---------- transformer ----------
        if self.arch == 'trans_enc':
            xseq = torch.cat([cond_emb, motion_seq], dim=0)     # (1+TP+TF, bs, L)
            xseq = self.sequence_pos_encoder(xseq)

            kp_mask = None
            if frames_mask is not None:
                step_pad = torch.zeros((bs, 1), dtype=torch.bool, device=xseq.device)
                kp_mask  = torch.cat([step_pad, frames_mask], dim=1)

            output = self.seqEncoder(xseq, src_key_padding_mask=kp_mask)
            output = output[-nframes:]                          # only future
        else:  # trans_dec
            tgt = self.sequence_pos_encoder(motion_seq)
            output = self.seqEncoder(
                tgt=tgt, memory=cond_emb,
                tgt_key_padding_mask=frames_mask,
            )
            output = output[-nframes:]

        return self.output_process(output)

    # ------------------------------------------------------------------
    # Interface (called by HumanoidTrainingPortal.diffuse)
    # ------------------------------------------------------------------

    def interface(self, x, timesteps, y=None):
        bs = x.shape[0]

        style_idx   = y['style_idx']
        past_motion = y['past_motion']
        traj_pose   = y['traj_pose']
        traj_trans  = y['traj_trans']
        sensor      = y.get('sensor', None)

        # CFG: mask past_motion to null (DiP/MotionDiffusion style)
        if self.cond_mask_prob > 0:
            keep = torch.rand(bs, device=past_motion.device) < (1.0 - self.cond_mask_prob)
            past_motion = past_motion * keep.view(bs, 1, 1, 1)

        # CFG: mask sensor to null
        if sensor is not None and self.sensor_cond_mask_prob > 0:
            keep_s = torch.rand(bs, device=sensor.device) < (1.0 - self.sensor_cond_mask_prob)
            sensor = sensor * keep_s.view(bs, 1)

        # Optional per-frame padding mask (True = pad), DiP-style
        frames_mask = None
        if self.mask_frames and 'mask' in y and torch.is_tensor(y['mask']):
            m = y['mask']
            TP = past_motion.shape[-1]
            TF = x.shape[-1]
            total = TP + TF
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(bs, -1)
            m = m.reshape(bs, -1)
            if m.shape[-1] >= total:
                valid = m[:, :total].to(torch.bool)
            else:
                past_valid = torch.ones((bs, TP), dtype=torch.bool, device=m.device)
                fut_valid  = m[:, :TF].to(torch.bool)
                valid = torch.cat([past_valid, fut_valid], dim=1)
            frames_mask = ~valid

        return self.forward(
            x, timesteps, past_motion, traj_pose, traj_trans, style_idx,
            sensor=sensor, frames_mask=frames_mask,
        )
