"""
MotionDiffusionDipTraj – DIPTRAJ architecture: trans_dec backbone +
per-frame trajectory tokens kept as separate cross-attention memory
(CAMDM-style routing, DiP-style backbone).

Conditioning layout (``trans_dec``):

    memory  (read-only cross-attn keys/values):
      ┌────┐ ┌─────┐ ┌──────┐ ┌────────────┐ ┌────────────┐
      │time│ │style│ │sensor│ │ traj_trans │ │ traj_pose  │     (3 + 2*TF, bs, L)
      │(1) │ │ (1) │ │ (1)  │ │   (TF)     │ │   (TF)     │
      └────┘ └─────┘ └──────┘ └────────────┘ └────────────┘

    target  (self-attn; queries memory):
      ┌────────────┐ ┌─────────────────┐
      │ past       │ │ noisy future    │                         (TP + TF, bs, L)
      │ motion (TP)│ │ motion (TF)     │
      └────────────┘ └─────────────────┘
                     └── output sliced (last TF)

The training portal passes in this ``y`` dict:

    past_motion : (bs, J, F, TP)
    traj_pose   : (bs, 6,  TF)
    traj_trans  : (bs, 2,  TF)
    style_idx   : (bs,)
    sensor      : (bs, env_sensor_dim)
    mask        : (bs, TF) or (TF,)
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
                 cond_mask_prob: float = 0.0,
                 sensor_cond_mask_prob: float = 0.0,
                 traj_cond_mask_prob: float = 0.0,
                 mask_frames: bool = False,
                 memory_pe: bool = True,
                 device=None,
                 # accept and ignore for cross-variant CLI symmetry
                 arch: str = 'trans_dec'):
        super().__init__()

        if arch != 'trans_dec':
            print(f"[DipTraj] note: arch='{arch}' requested; this model "
                  f"is hard-wired to trans_dec, the value will be ignored.")

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
        self.arch = 'trans_dec'
        self.cond_mask_prob = float(cond_mask_prob)
        self.sensor_cond_mask_prob = float(sensor_cond_mask_prob)
        self.traj_cond_mask_prob = float(traj_cond_mask_prob)
        self.env_sensor_dim = env_sensor_dim
        self.past_frame = past_frame
        self.future_frame = future_frame
        self.mask_frames = mask_frames
        self.memory_pe = bool(memory_pe)

        # Motion projector (shared past + noisy future, prefix-completion)
        self.motion_process = MotionProcess(self.input_feats, self.latent_dim)

        # Per-frame trajectory: separate tokens at front of memory (CAMDM-style)
        self.traj_trans_process = TrajProcess(2, self.latent_dim)
        self.traj_pose_process  = TrajProcess(6, self.latent_dim)

        # Global cond encoders, each produces (1, bs, L)
        self.sensor_encoder = EnvSensorEncoder(env_sensor_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep       = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_style          = EmbedStyle(nstyles, self.latent_dim)

        print("DipTraj TRANS_DEC + CONCAT init")
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim, nhead=self.num_heads,
            dim_feedforward=self.ff_size, dropout=self.dropout,
            activation=self.activation,
        )
        self.seqDecoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_layers)

        self.output_process = OutputProcess(self.input_feats, self.latent_dim,
                                            self.njoints, self.nfeats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, style_idx,
                sensor=None, frames_mask=None):
        """
        x:           (bs, J, F, TF)
        past_motion: (bs, J, F, TP)
        traj_pose:   (bs, 6, TF)
        traj_trans:  (bs, 2, TF)
        style_idx:   (bs,)
        sensor:      (bs, env_sensor_dim) or None
        frames_mask: (bs, TP+TF) bool padding mask, True = pad. Optional.
        Returns:     (bs, J, F, TF) predicted denoised future motion
        """
        bs, njoints, nfeats, nframes = x.shape
        TP = past_motion.shape[-1]
        TF = nframes

        combined   = torch.cat([past_motion, x], dim=-1)        # (bs, J, F, TP+TF)
        motion_seq = self.motion_process(combined)              # (TP+TF, bs, L)
        tgt        = self.sequence_pos_encoder(motion_seq)

        time_emb  = self.embed_timestep(timesteps)              # (1, bs, L)
        style_emb = self.embed_style(style_idx).unsqueeze(0)    # (1, bs, L)
        if sensor is not None:
            sensor_emb = self.sensor_encoder(sensor)            # (1, bs, L)
        else:
            sensor_emb = torch.zeros(
                1, bs, self.latent_dim, device=x.device, dtype=x.dtype
            )
        traj_trans_emb = self.traj_trans_process(traj_trans)    # (TF, bs, L)
        traj_pose_emb  = self.traj_pose_process(traj_pose)      # (TF, bs, L)

        # Per-frame PE on memory: add the SAME PE block that motion frames
        # TP..TP+TF-1 receive in tgt, so cross-attention from motion frame k
        # to traj_*_k has positional alignment. Globals (time, style, sensor)
        # stay PE-free since they're already content-distinct.
        if self.memory_pe:
            pe_frames = self.sequence_pos_encoder.pe[TP : TP + TF]   # (TF, 1, L)
            traj_trans_emb = traj_trans_emb + pe_frames
            traj_pose_emb  = traj_pose_emb  + pe_frames

        memory = torch.cat([
            time_emb, style_emb, sensor_emb,
            traj_trans_emb, traj_pose_emb,
        ], dim=0)                                               # (3 + 2*TF, bs, L)

        output = self.seqDecoder(
            tgt=tgt, memory=memory,
            tgt_key_padding_mask=frames_mask,
        )
        output = output[-nframes:]

        return self.output_process(output)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def interface(self, x, timesteps, y=None):
        bs = x.shape[0]

        style_idx   = y['style_idx']
        past_motion = y['past_motion']
        traj_pose   = y['traj_pose']
        traj_trans  = y['traj_trans']
        sensor      = y.get('sensor', None)

        if self.cond_mask_prob > 0:
            keep = torch.rand(bs, device=past_motion.device) < (1.0 - self.cond_mask_prob)
            past_motion = past_motion * keep.view(bs, 1, 1, 1)

        if sensor is not None and self.sensor_cond_mask_prob > 0:
            keep_s = torch.rand(bs, device=sensor.device) < (1.0 - self.sensor_cond_mask_prob)
            sensor = sensor * keep_s.view(bs, 1)

        if self.traj_cond_mask_prob > 0:
            keep_t = torch.rand(bs, device=traj_trans.device) < (1.0 - self.traj_cond_mask_prob)
            traj_trans = traj_trans * keep_t.view(bs, 1, 1)
            traj_pose  = traj_pose  * keep_t.view(bs, 1, 1)

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
