# DiP-port variants on the avoid2d data

Three primary **architectural choices** organize this work. Each one
defines a different *conditioning structure*; backbone (`trans_enc` vs
`trans_dec`) and `emb_policy` (`add` vs `concat`) are orthogonal knobs
that can be flipped within any of the three.

| Tag | Conditioning structure | Class | Files |
| --- | --- | --- | --- |
| **CAMDM** | per-frame traj as separate tokens at the front (TF traj_trans + TF traj_pose) | `MotionDiffusion` / `MotionDiffusionEnv` | `network/models.py`, `network/models_env2d.py` |
| **DIPTRAJ** | per-frame traj **added into** the future-motion tokens; single fused conditioning token | `MotionDiffusionDipTraj` | `network/models_dip2d_traj.py` |
| **DIPCMD** | single body-frame `(vx, vy, ω)` command token; no per-frame traj | `MotionDiffusionDipCmd` | `network/models_dip2d_cmd.py` |

There is also one **experimental hybrid** on its own branch
(`dip_traj_dec_avoid2d`): the **DIPTRAJ family** with CAMDM-style
per-frame traj *tokens* instead of "traj added into motion", and the
DiP-shaped `trans_dec + concat` backbone. It is not a fourth peer —
it's an instance of DIPTRAJ with different hyperparameters and a
slightly different conditioning layout.


## 1. Did the input/output change between CAMDM and the DIP-* variants?

**No** at the `interface(x, timesteps, y)` boundary. The `y` dict keys
and shapes are the same as the existing env-sensor pipeline. The
training portal, dataloader, diffusion schedule, loss terms, and wandb
logging all stay identical.

```python
y = {
    "past_motion": (bs, J=31, F=6, TP=10),     # past motion frames
    "traj_pose":   (bs, 6,  TF=45),            # per-future-frame heading      (CAMDM, DIPTRAJ)
    "traj_trans":  (bs, 2,  TF=45),            # per-future-frame XY           (CAMDM, DIPTRAJ)
    "command":     (bs, 3),                    # body-frame (vx, vy, omega)    (DIPCMD only)
    "style_idx":   (bs,),
    "sensor":      (bs, 226),
    "mask":        (bs, TF) or (TF,),
}
# x:        (bs, J, F, TF=45)
# returns:  (bs, J, F, TF=45)
```

DIPCMD trades the per-frame `traj_*` keys for a single `command` key.
Everything else (past motion, style, sensor, mask) is shared.


## 2. CAMDM — the original

Transformer input as **a concatenated sequence of separate condition
tokens followed by motion tokens**, all in one self-attention pass
(`trans_enc`):

```
 pos:   0      1       2..46          47..91          92..101         102..146
        ┌────┐ ┌─────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────┐
xseq:   │time│ │style│ │ traj_trans │ │ traj_pose  │ │ past       │ │ future      │
        │ (1)│ │ (1) │ │   (45)     │ │   (45)     │ │ motion (10)│ │ motion (45) │
        └────┘ └─────┘ └────────────┘ └────────────┘ └────────────┘ └─────────────┘
                                                                    └── output sliced
```

Total: **147 tokens** (or 148 when adding the sensor as one extra
global token in `MotionDiffusionEnv`).

Each input source has its own `nn.Linear`-shaped projector. CFG:
`past_motion` is zeroed with prob `cond_mask_prob`; `sensor` with
`sensor_cond_mask_prob`.


## 3. DIPTRAJ — DiP-style with traj-into-motion

Drops the separate per-frame traj tokens. The traj is projected
per-frame and **added on top of** the corresponding future-motion
token embeddings. Conditioning is collapsed into a single fused token
(`emb_policy='add'`):

```
 pos:   0                  1..10           11..55
        ┌────────────────┐ ┌────────────┐ ┌──────────────────────────┐
xseq:   │ time + style   │ │ past       │ │ future motion + traj     │
        │   + sensor (1) │ │ motion (10)│ │ (per-frame, traj added)  │
        └────────────────┘ └────────────┘ └──────────────────────────┘
                                          └── output sliced (last 45)
```

Total: **56 tokens**. The "traj added into motion" injection means the
future-motion token at frame k carries **both** the noisy motion and
the desired XY+heading at frame k, summed inside one 256-D vector.

The single fused conditioning token is `emb = time_emb + style_emb +
sensor_emb`. CFG: `past_motion` (prob `cond_mask_prob`), `sensor`
(prob `sensor_cond_mask_prob`).


## 4. DIPCMD — DiP-style with a single twist command

No per-frame steering at all. A body-frame twist `(vx, vy, ω)` is
projected to a single token (`CommandEncoder`: 3 → 128 → 256 MLP) and
folded into the conditioning sum:

```
 pos:   0                          1..10           11..55
        ┌────────────────────────┐ ┌────────────┐ ┌────────────────┐
xseq:   │ time + style + sensor  │ │ past       │ │ future motion  │
        │   + command (1)        │ │ motion (10)│ │ (45)           │
        └────────────────────────┘ └────────────┘ └────────────────┘
                                                  └── output sliced
```

Total: **56 tokens**. Future-motion tokens are *clean* embeddings of
the noisy motion only — no per-frame traj contamination. The single
fused conditioning token now carries the global steering cue.

The dataset (`network/dataset_g1_env2d_cmd.py`) computes the command
from the same pkl the other variants use:

```
vx = traj_trans_per_frame[current, -1, 0] / (TF * dt)
vy = traj_trans_per_frame[current, -1, 1] / (TF * dt)
ω  = yaw_of(traj_pose_per_frame[current, -1]) / (TF * dt)
```

CFG: `past_motion`, `sensor`, and additionally `command` (prob
`cmd_cond_mask_prob`).


## 5. The reference: DiP itself (`closd/diffusion_planner/model/mdm.py`)

DiP's released checkpoint uses:

```
 memory (read-only, cross-attn keys/values):
        ┌────┐ ┌─────────────────────────────────┐
mem:    │time│ │ BERT text tokens (N ≈ 20–30)    │     shape (1+N, bs, 512)
        │ (1)│ │  (with padding mask)            │
        └────┘ └─────────────────────────────────┘

 target (self-attn among these; queries the memory):
        ┌──────────────────┐ ┌─────────────────┐
tgt:    │ prefix (clean)   │ │ noisy future    │     shape (60, bs, 512)
        │   (20)           │ │   (40)          │
        └──────────────────┘ └─────────────────┘
                              └── output sliced (last 40)
```

- `arch = trans_dec` (cross-attn into the memory)
- `emb_policy = concat` (time and BERT keep distinct memory slots)
- `mask_frames = true`
- `context_len = 20`, `pred_len = 40` at 20 fps → ~1 s past, ~2 s future
- `latent_dim = 512` (we use 256), 8 layers, 10 diffusion steps

So DIPTRAJ and DIPCMD in this repo match DiP's **CLIP path** (single
conditioning token, added into emb, `trans_enc`), not the **BERT path**
(multi-token memory + cross-attn) that DiP's actual checkpoint uses.


## 6. Side-by-side summary

| Aspect | CAMDM | DIPTRAJ | DIPCMD | DiP (CLoSD) |
| --- | --- | --- | --- | --- |
| Backbone | trans_enc | trans_enc | trans_enc | trans_dec |
| emb_policy | n/a (per-source tokens) | add | add | concat |
| Past prefix tokens | 10 (separate) | 10 (concat with future) | 10 (concat with future) | 20 (concat with future) |
| Future steering | per-frame traj as tokens (TF + TF) | per-frame traj added to future motion | single command token | none (text only) |
| Total tokens at transformer | ~147 | 56 | 56 | 60 tgt + 1+N memory |
| Output slice | last TF | last TF | last TF | `output[context_len:]` |
| `interface(x, t, y)` signature | unchanged | unchanged | unchanged | similar but uses `y['text']` |


## 7. Experimental hybrid (this branch: `dip_traj_dec_avoid2d`)

The `dip_traj_dec_avoid2d` branch is **DIPTRAJ with three knobs flipped**:

1. Per-frame `traj_trans` and `traj_pose` become **separate tokens**
   at the front (CAMDM-style), not added into the future-motion tokens.
2. `emb_policy = 'concat'` — time, style, sensor are kept as separate
   memory tokens.
3. `arch = 'trans_dec'` — backbone is the transformer decoder, with
   the conditioning fed as cross-attention memory.

```
 memory (read-only, cross-attn):
        ┌────┐ ┌─────┐ ┌──────┐ ┌────────────┐ ┌────────────┐
mem:    │time│ │style│ │sensor│ │ traj_trans │ │ traj_pose  │   shape (93, bs, 256)
        │(1) │ │ (1) │ │ (1)  │ │   (45)     │ │   (45)     │
        └────┘ └─────┘ └──────┘ └────────────┘ └────────────┘

 target (self-attn; queries memory):
        ┌────────────┐ ┌─────────────────┐
tgt:    │ past       │ │ noisy future    │                       shape (55, bs, 256)
        │ motion (10)│ │ motion (45)     │
        └────────────┘ └─────────────────┘
                       └── output sliced
```

The point of the experiment is to ablate "where does the per-frame
steering signal go": as motion-token contamination (DIPTRAJ) or as
distinct memory tokens (this hybrid). Implementation is in
`network/models_dip2d_traj_dec.py` (`MotionDiffusionDipTrajDec`).


## 8. Files at-a-glance

```
# DIPTRAJ (per-frame traj added into future motion tokens)
PyTorch/network/models_dip2d_traj.py             # MotionDiffusionDipTraj
PyTorch/train_g1_env2d_dip_traj.py
PyTorch/config/default_g1_env_dip_traj.json
PyTorch/config/default_g1_env_geo_dip_traj.json
PyTorch/visualize/step6_demo_control_dip_traj.py

# DIPCMD (single (vx, vy, omega) command)
PyTorch/network/models_dip2d_cmd.py              # MotionDiffusionDipCmd
PyTorch/network/dataset_g1_env2d_cmd.py
PyTorch/train_g1_env2d_dip_cmd.py
PyTorch/config/default_g1_env_geo_dip_cmd.json
PyTorch/visualize/step6_demo_control_cmd.py

# Experimental hybrid (this branch only)
PyTorch/network/models_dip2d_traj_dec.py         # MotionDiffusionDipTrajDec
PyTorch/train_g1_env2d_dip_traj_dec.py
PyTorch/config/default_g1_env_geo_dip_traj_dec.json
PyTorch/visualize/step6_demo_control_dip_traj_dec.py
```


## 9. Training invocations

Pick the variant and use the matching script + config:

```bash
# DIPTRAJ
python train_g1_env2d_dip_traj.py \
    -n dip_traj_yc_ep3k \
    -c config/default_g1_env_geo_dip_traj.json \
    -i data/pkls/lafan1_g1_motion30_env2d_yellow_circle_detour_only_none.pkl \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group dip_traj

# DIPCMD
python train_g1_env2d_dip_cmd.py \
    -n dip_cmd_yc_ep3k \
    -c config/default_g1_env_geo_dip_cmd.json \
    -i data/pkls/lafan1_g1_motion30_env2d_yellow_circle_detour_only_none.pkl \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group dip_cmd

# Experimental hybrid (this branch only)
python train_g1_env2d_dip_traj_dec.py \
    -n dip_traj_dec_yc_ep3k \
    -c config/default_g1_env_geo_dip_traj_dec.json \
    -i data/pkls/lafan1_g1_motion30_env2d_yellow_circle_detour_only_none.pkl \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group dip_traj_dec
```
