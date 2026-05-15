"""
MotionDiffusionDipCmd – DiP-style motion diffusion conditioned on a single
body-frame twist command (vx, vy, omega) instead of a per-frame trajectory.

Differences from MotionDiffusionDipTraj (network/models_dip2d_traj.py):
  * Drops the per-frame ``traj_pose`` and ``traj_trans`` projections and
    their additive injection into the future motion tokens.
  * Adds a small ``CommandEncoder`` MLP that maps a single command vector
    ``(bs, 3)`` to one ``(1, bs, L)`` context token, summed into ``emb``
    alongside ``time + style + sensor``. This matches DiP's "single global
    cue" pattern (the way text or goal-joint locations enter MDM).

Conditioning tensor at inference time (``y`` dict):

    past_motion : (bs, J, F, TP)        past motion frames
    command     : (bs, 3)               body-frame (vx, vy, omega)
    style_idx   : (bs,)                 style/action index
    sensor      : (bs, env_sensor_dim)  current-frame occupancy
    mask        : (bs, TF) or (TF,)     per-future-frame validity (optional)

The model output is identical to MotionDiffusionDipTraj: predicted denoised
future motion of shape (bs, J, F, TF).
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
    Small MLP that lifts a (bs, 3) body-frame twist command into the
    transformer's latent space as a single (1, bs, L) token.

    Mirrors the shape contract of EnvSensorEncoder so the two embeddings
    can be summed directly into ``emb``.
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
        # x: (bs, cmd_dim) → (1, bs, latent_dim)
        return self.net(x).unsqueeze(0)


class MotionDiffusionDipCmd(nn.Module):

    def __init__(self, input_feats, nstyles, njoints, nfeats, rot_req, clip_len,
                 env_sensor_dim: int = 226,
                 cmd_dim: int = 3,
                 past_frame: int = 10, future_frame: int = 45,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 8, num_heads: int = 4,
                 dropout: float = 0.2, activation: str = "gelu",
                 arch: str = 'trans_enc',
                 cond_mask_prob: float = 0.0,
                 sensor_cond_mask_prob: float = 0.0,
                 cmd_cond_mask_prob: float = 0.0,
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
        self.cmd_cond_mask_prob = float(cmd_cond_mask_prob)
        self.env_sensor_dim = env_sensor_dim
        self.cmd_dim = cmd_dim
        self.past_frame = past_frame
        self.future_frame = future_frame
        self.mask_frames = mask_frames

        # Per-frame motion projector (shared past + noisy future, prefix-completion)
        self.motion_process = MotionProcess(self.input_feats, self.latent_dim)

        # Context-token encoders, all producing (1, bs, L)
        self.sensor_encoder  = EnvSensorEncoder(env_sensor_dim, self.latent_dim)
        self.command_encoder = CommandEncoder(cmd_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep       = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_style          = EmbedStyle(nstyles, self.latent_dim)

        if self.arch == 'trans_enc':
            print("DipCmd TRANS_ENC init")
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim, nhead=self.num_heads,
                dim_feedforward=self.ff_size, dropout=self.dropout,
                activation=self.activation,
            )
            self.seqEncoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("DipCmd TRANS_DEC init")
            dec_layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim, nhead=self.num_heads,
                dim_feedforward=self.ff_size, dropout=self.dropout,
                activation=self.activation,
            )
            self.seqEncoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_layers)
        else:
            raise ValueError(f"DipCmd supports [trans_enc, trans_dec]; got '{arch}'")

        self.output_process = OutputProcess(self.input_feats, self.latent_dim,
                                            self.njoints, self.nfeats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, timesteps, past_motion, command, style_idx,
                sensor=None, frames_mask=None):
        """
        Args:
            x:           (bs, J, F, TF)         noisy future motion
            timesteps:   (bs,) int
            past_motion: (bs, J, F, TP)
            command:     (bs, 3)                body-frame (vx, vy, omega)
            style_idx:   (bs,)
            sensor:      (bs, env_sensor_dim) or None
            frames_mask: (bs, TP+TF) bool, True = pad. Optional.

        Returns:
            (bs, J, F, TF) predicted denoised future motion
        """
        bs, njoints, nfeats, nframes = x.shape
        TP = past_motion.shape[-1]

        # ---------- single context token: time + style + sensor + command -------
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

        cond_emb = time_emb + style_emb + sensor_emb + cmd_emb  # (1, bs, L)

        # ---------- prefix-completion motion sequence (past + noisy future) ----
        combined = torch.cat([past_motion, x], dim=-1)          # (bs, J, F, TP+TF)
        motion_seq = self.motion_process(combined)              # (TP+TF, bs, L)

        # ---------- transformer ----------
        if self.arch == 'trans_enc':
            xseq = torch.cat([cond_emb, motion_seq], dim=0)     # (1+TP+TF, bs, L)
            xseq = self.sequence_pos_encoder(xseq)

            kp_mask = None
            if frames_mask is not None:
                step_pad = torch.zeros((bs, 1), dtype=torch.bool, device=xseq.device)
                kp_mask  = torch.cat([step_pad, frames_mask], dim=1)

            output = self.seqEncoder(xseq, src_key_padding_mask=kp_mask)
            output = output[-nframes:]
        else:  # trans_dec
            tgt = self.sequence_pos_encoder(motion_seq)
            output = self.seqEncoder(
                tgt=tgt, memory=cond_emb,
                tgt_key_padding_mask=frames_mask,
            )
            output = output[-nframes:]

        return self.output_process(output)

    # ------------------------------------------------------------------
    # Interface (called by HumanoidTrainingPortal.diffuse via a subclass
    # that skips the traj permutations — see train_g1_env2d_dip_cmd.py)
    # ------------------------------------------------------------------

    def interface(self, x, timesteps, y=None):
        bs = x.shape[0]

        style_idx   = y['style_idx']
        past_motion = y['past_motion']
        command     = y['command']
        sensor      = y.get('sensor', None)

        # CFG: mask past_motion to null
        if self.cond_mask_prob > 0:
            keep = torch.rand(bs, device=past_motion.device) < (1.0 - self.cond_mask_prob)
            past_motion = past_motion * keep.view(bs, 1, 1, 1)

        # CFG: mask sensor
        if sensor is not None and self.sensor_cond_mask_prob > 0:
            keep_s = torch.rand(bs, device=sensor.device) < (1.0 - self.sensor_cond_mask_prob)
            sensor = sensor * keep_s.view(bs, 1)

        # CFG: mask command (lets us drop the command for unconditional CFG)
        if command is not None and self.cmd_cond_mask_prob > 0:
            keep_c = torch.rand(bs, device=command.device) < (1.0 - self.cmd_cond_mask_prob)
            command = command * keep_c.view(bs, 1)

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
            x, timesteps, past_motion, command, style_idx,
            sensor=sensor, frames_mask=frames_mask,
        )
