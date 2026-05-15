"""
MotionDiffusionDipCmd – DIPCMD architecture: trans_dec backbone +
a single body-frame twist command ``(vx, vy, ω)`` as a distinct memory
token (no per-frame traj steering).

Conditioning layout (``trans_dec``):

    memory  (read-only cross-attn keys/values):
      ┌────┐ ┌─────┐ ┌──────┐ ┌─────┐
      │time│ │style│ │sensor│ │ cmd │                       (4, bs, L)
      │(1) │ │ (1) │ │ (1)  │ │ (1) │
      └────┘ └─────┘ └──────┘ └─────┘

    target  (self-attn; queries memory):
      ┌────────────┐ ┌─────────────────┐
      │ past       │ │ noisy future    │                    (TP + TF, bs, L)
      │ motion (TP)│ │ motion (TF)     │
      └────────────┘ └─────────────────┘
                     └── output sliced (last TF)

The training portal passes in this ``y`` dict:

    past_motion : (bs, J, F, TP)
    command     : (bs, 3)                 # body-frame (vx, vy, omega)
    style_idx   : (bs,)
    sensor      : (bs, env_sensor_dim)
    mask        : (bs, TF) or (TF,)
"""

import torch
import torch.nn as nn

from network.models import (
    MotionProcess, PositionalEncoding,
    TimestepEmbedder, OutputProcess, EmbedStyle,
)
from network.models_env2d import EnvSensorEncoder


class CommandEncoder(nn.Module):
    """
    Small MLP: (bs, 3) body-frame twist → (1, bs, L) memory token.
    """

    def __init__(self, cmd_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.cmd_dim = cmd_dim
        self.net = nn.Sequential(
            nn.Linear(cmd_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(0)


class MotionDiffusionDipCmd(nn.Module):

    def __init__(self, input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
                 env_sensor_dim: int = 226,
                 cmd_dim: int = 3,
                 past_frame: int = 10, future_frame: int = 45,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 4,
                 dropout: float = 0.2, activation: str = "gelu",
                 cond_mask_prob: float = 0.0,
                 sensor_cond_mask_prob: float = 0.0,
                 cmd_cond_mask_prob: float = 0.0,
                 mask_frames: bool = False,
                 device=None,
                 # accept and ignore for cross-variant CLI symmetry
                 arch: str = 'trans_dec'):
        super().__init__()

        if arch != 'trans_dec':
            print(f"[DipCmd] note: arch='{arch}' requested; this model "
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
        self.cmd_cond_mask_prob = float(cmd_cond_mask_prob)
        self.env_sensor_dim = env_sensor_dim
        self.cmd_dim = cmd_dim
        self.past_frame = past_frame
        self.future_frame = future_frame
        self.mask_frames = mask_frames

        # Motion projector (shared past + noisy future, prefix-completion)
        self.motion_process = MotionProcess(self.input_feats, self.latent_dim)

        # Global cond encoders, each (1, bs, L)
        self.sensor_encoder  = EnvSensorEncoder(env_sensor_dim, self.latent_dim)
        self.command_encoder = CommandEncoder(cmd_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep       = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_style          = EmbedStyle(nstyles, self.latent_dim)

        print("DipCmd TRANS_DEC + CONCAT init")
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

    def forward(self, x, timesteps, past_motion, command, style_idx,
                sensor=None, frames_mask=None):
        """
        x:           (bs, J, F, TF)
        past_motion: (bs, J, F, TP)
        command:     (bs, 3)              body-frame (vx, vy, omega)
        style_idx:   (bs,)
        sensor:      (bs, env_sensor_dim) or None
        frames_mask: (bs, TP+TF) bool padding mask, True = pad. Optional.
        Returns:     (bs, J, F, TF) predicted denoised future motion
        """
        bs, njoints, nfeats, nframes = x.shape

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
        if command is not None:
            cmd_emb = self.command_encoder(command)             # (1, bs, L)
        else:
            cmd_emb = torch.zeros(
                1, bs, self.latent_dim, device=x.device, dtype=x.dtype
            )

        memory = torch.cat([
            time_emb, style_emb, sensor_emb, cmd_emb,
        ], dim=0)                                               # (4, bs, L)

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
        command     = y['command']
        sensor      = y.get('sensor', None)

        if self.cond_mask_prob > 0:
            keep = torch.rand(bs, device=past_motion.device) < (1.0 - self.cond_mask_prob)
            past_motion = past_motion * keep.view(bs, 1, 1, 1)

        if sensor is not None and self.sensor_cond_mask_prob > 0:
            keep_s = torch.rand(bs, device=sensor.device) < (1.0 - self.sensor_cond_mask_prob)
            sensor = sensor * keep_s.view(bs, 1)

        if command is not None and self.cmd_cond_mask_prob > 0:
            keep_c = torch.rand(bs, device=command.device) < (1.0 - self.cmd_cond_mask_prob)
            command = command * keep_c.view(bs, 1)

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
            x, timesteps, past_motion, command, style_idx,
            sensor=sensor, frames_mask=frames_mask,
        )
