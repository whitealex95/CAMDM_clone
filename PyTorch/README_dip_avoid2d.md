# `dip_avoid2d` — DiP-style network for env2d training

Branch-local notes comparing three motion-diffusion implementations:

| Label    | Class                  | File                                                     | Conditioning                                  |
| -------- | ---------------------- | -------------------------------------------------------- | --------------------------------------------- |
| Previous | `MotionDiffusionEnv`   | `network/models_env2d.py`                                | past + traj + style + sensor (CAMDM-style)    |
| Current  | `MotionDiffusionDiP`   | `network/models_dip2d.py` (this branch)                  | past + traj + style + sensor (DiP-style fuse) |
| DiP      | `MDM`                  | `CLoSD/closd/diffusion_planner/model/mdm.py`             | prefix + text (+ goal / keyframes optional)   |

The "current" model is a port of DiP's MDM into CAMDM's data and training
stack, with the CLIP/BERT text branch dropped and the environment sensor
used as the only "context" input. It reuses CAMDM's `HumanoidTrainingPortal`
unchanged.


## 1. Did the input/output change?

**No** — the model contract is identical to `MotionDiffusionEnv` from the
caller's perspective:

```python
# Public API — both models implement this exact signature
model.interface(x, timesteps, y) -> Tensor

# y dict (same keys, same shapes for both)
y = {
    "past_motion": (bs, J=31, F=6, TP=10),     # past motion frames
    "traj_pose":   (bs, 6,  TF=45),            # per-future-frame heading (6d)
    "traj_trans":  (bs, 2,  TF=45),            # per-future-frame XY (yaw-local)
    "style_idx":   (bs,),                      # action/style index
    "sensor":      (bs, 226),                  # current-frame occupancy
    "mask":        (bs, TF) or (TF,),          # valid-frame mask (optional)
}

# x:        (bs, J, F, TF=45)  noisy future motion
# returns:  (bs, J, F, TF=45)  denoised future motion
```

The dataset, the dataloader collate, the diffusion schedule, the loss
terms, and the wandb logging all remain identical. The change is purely
**inside** the model's forward pass.


## 2. The previous model — `MotionDiffusionEnv`

Builds the transformer input as **a sequence of separate condition tokens
followed by motion frame tokens**:

```
seq position:  0       1        2..TF+1     TF+2..2TF+1   2TF+2     2TF+3..2TF+TP+2  2TF+TP+3..end
content:       time    style    traj_trans  traj_pose     sensor    past_motion      future_motion
shape per pos: (1,bs,L)(1,bs,L) (TF,bs,L)   (TF,bs,L)     (1,bs,L)  (TP,bs,L)        (TF,bs,L)
```

- One transformer encoder layer-stack runs over the whole concatenation.
- Each input source has its own `nn.Linear`-style projector
  (`MotionProcess`, `TrajProcess`, `EnvSensorEncoder`, `EmbedStyle`,
  `TimestepEmbedder`).
- The output is taken from the last `nframes=TF` positions (the
  `future_motion` tail), then projected back to motion space.
- CFG: `past_motion` is zeroed with prob `cond_mask_prob`; `sensor` is
  zeroed with prob `sensor_cond_mask_prob`.

ASCII view of the forward pass:

```
time────┐
style───┤
traj_t──┤
traj_p──┼─► concat ─► PE ─► TransformerEncoder ─► take last TF ─► out
sensor──┤
past────┤
future──┘   (noisy x_t)
```


## 3. DiP — `closd/diffusion_planner/model/mdm.py::MDM`

The "real" DiP processes motion as **one per-frame sequence** with a
**single context token** carrying all non-motion conditioning:

```
context-token (emb = time + text)   ◄── ONE token, prepended (trans_enc)
                                         OR used as memory (trans_dec)

motion-tokens = InputProcess(cat([prefix, x_t]))  ◄── per-frame, TP+TF long
```

Key mechanics:

- **Prefix completion**: the past ("prefix") and the noisy future are
  concatenated *along the time axis*, projected by a single
  `InputProcess` (one MLP, not one-per-input-stream), and the model
  predicts the whole sequence; the loss is taken on `output[context_len:]`.
- **`emb_policy='add'`**: `time_emb`, `text_emb`, optional `action_emb`,
  optional `target_cond_emb` are all summed into one (1, bs, L) token.
- **`trans_enc`**: the context token is prepended once to the motion
  sequence (`xseq = [emb, motion_seq]`).
- **`trans_dec`**: the context token is the cross-attention `memory`;
  motion is the `tgt`.
- **Optional bits ignored here**: CLIP/BERT text encoder,
  `multi_target_cond` (goal-joint locations), `keyframe_cond_type`,
  `gru` backbone, `is_prefix_comp` machinery for varied context lengths.

ASCII view:

```
                          ┌─► time ─┐
text ─► CLIP/BERT ────────┤         ├─► sum ─► emb (1 token)
                          └─► action┘
                                                 │
                                                 ▼
prefix ─┐ cat along time  ┌─► InputProcess ─► motion_seq (TP+TF tokens)
x_t   ──┘                 │
                          ▼
            [emb, motion_seq] ─► PE ─► TransformerEncoder
                                       └─► output[context_len:] ─► out
```


## 4. The current model — `MotionDiffusionDiP`

Same DiP token layout, but:

- **Drops text completely.** No CLIP, no BERT, no `embed_text`.
- **Adds the scene sensor** as the only context input. Reuses
  `EnvSensorEncoder` (NSM-style 2-layer MLP, ELU) verbatim from
  `models_env2d.py` to embed `(bs, 226)` → `(1, bs, L)`.
- **Keeps the prefix-completion mechanic**: past and noisy future are
  concatenated along the time axis and projected by a single
  `MotionProcess`, then the last `TF` outputs are taken.
- **Folds style into emb** (`time + style + sensor`), matching DiP's
  `emb_policy='add'`.
- **Adds traj as per-frame additive conditioning on the future tokens**.
  CAMDM's existing `traj_trans`/`traj_pose` are projected per-frame and
  added to the *future* portion of the motion-token sequence (DiP itself
  has no analogous per-frame structured cond; this is the closest faithful
  mapping of CAMDM's traj-following signal).
- **Supports `trans_enc` and `trans_dec`** identically to DiP. The
  `gru` branch is dropped.
- **CFG masking** matches `MotionDiffusionEnv`: `cond_mask_prob` on
  `past_motion`, `sensor_cond_mask_prob` on `sensor`.
- **Optional frame padding mask** (`mask_frames=True` in config) wires
  `cond['mask']` into `src_key_padding_mask` / `tgt_key_padding_mask`
  exactly like DiP.

ASCII view:

```
time ───┐
style ──┼─► sum ─► emb (1 token)
sensor ─┘
                                                                    │
                                                                    ▼
past   ─┐ cat along time  ┌─► MotionProcess ─► motion_seq (TP+TF tokens)
x_t  ───┘                 │                          │
                          │                          ▼
                          │           future portion gets +traj_emb
                          │                          │
                          ▼                          │
            [emb, motion_seq] ─► PE ─► Transformer ──┘
                                       └─► output[-TF:] ─► out
```


## 5. Side-by-side diff table

| Aspect                         | Previous (`MotionDiffusionEnv`)               | Current (`MotionDiffusionDiP`)                | DiP (`MDM`)                                 |
| ------------------------------ | --------------------------------------------- | --------------------------------------------- | ------------------------------------------- |
| Token layout                   | many separate cond tokens + past + future     | **one** emb token + (past + future) per-frame | **one** emb token + (prefix + future) per-frame |
| Cond aggregation               | concat as separate tokens                     | additive into single emb token                | additive into single emb token              |
| Past motion handling           | own `MotionProcess`, separate tokens          | concat with noisy future, shared `MotionProcess` | same (called "prefix")                    |
| Trajectory handling            | own tokens prepended                          | per-frame additive on future tokens           | none (DiP has no per-frame XY/heading cond) |
| Style handling                 | own token                                     | added into emb                                | (no style — uses text instead)              |
| Sensor handling                | own token (NSM MLP)                           | added into emb (same NSM MLP)                 | (no sensor — uses text)                     |
| Text                           | (none)                                        | (none — dropped)                              | CLIP or BERT encoder + linear projector     |
| Backbone choices               | `trans_enc` / `trans_dec` / `gru`             | `trans_enc` / `trans_dec`                     | `trans_enc` / `trans_dec` / `gru`           |
| CFG mask on past               | yes (`cond_mask_prob`)                        | yes                                           | yes                                         |
| CFG mask on context            | yes on sensor                                 | yes on sensor                                 | yes on text                                 |
| Frame padding mask             | not used                                      | optional (`mask_frames` in config)            | optional (`mask_frames`)                    |
| Output slice                   | `output[-nframes:]`                           | `output[-nframes:]`                           | `output[context_len:]`                      |
| Default `num_layers`           | 4                                             | 8 (DiP default)                               | 8                                           |
| Param count (latent=256, L=8)  | ~5.2M                                         | ~6.8M                                         | ~depends on text encoder                    |
| `interface(x, t, y)` signature | unchanged                                     | unchanged                                     | similar but uses `y['text']`, `y['prefix']` |


## 6. What's NOT ported from DiP

These DiP features are intentionally left out (out of scope for
`dip_avoid2d`):

- CLIP / BERT text encoder and the `embed_text` projector.
- `multi_target_cond` / `EmbedTargetLocMulti|Single|Split` — goal-joint
  cond used in CLoSD's interactive sampling.
- `keyframe_cond_type` (and the extra input channel that carries the
  keyframe mask).
- `gru` backbone.
- `is_prefix_comp` flag for variable-length prefixes — here `TP`/`TF`
  are fixed by the dataset config.
- DiP's caching of `text_embed` in the `y` dict.


## 7. Files added on this branch

```
PyTorch/network/models_dip2d.py            # MotionDiffusionDiP
PyTorch/train_g1_env2d_dip.py              # entry point (mirrors train_g1_env2d.py)
PyTorch/config/default_g1_env_dip.json     # default config, geo losses OFF
PyTorch/config/default_g1_env_geo_dip.json # default config, geo losses ON
PyTorch/README_dip_avoid2d.md              # this file
```

## 8. Training invocation

The wandb flags are inherited from the existing `config/option.py` —
nothing extra needed.

```bash
python train_g1_env2d_dip.py \
    -n dip_avoid2d_yc_ep3k \
    -c config/default_g1_env_geo_dip.json \
    -i data/pkls/lafan1_g1_motion30_env2d_yellow_circle_detour_only_none.pkl \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group dip_avoid2d \
    --wandb_run_name dip_avoid2d_yc_ep3k \
    --wandb_tags dip,avoid2d,env2d,scene_only
```
