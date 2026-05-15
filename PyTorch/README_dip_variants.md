# Three architectural variants on the avoid2d data

This work compares three motion-diffusion architectures on the same
yellow_circle-detour avoid2d data. They share the dataset, the
hyperparameters, and the training schedule; what differs is **the
transformer's conditioning structure**.

| Tag | Backbone | Conditioning routing | Class | File |
| --- | --- | --- | --- | --- |
| **CAMDM** | `trans_enc` | per-source tokens *and* per-frame traj as separate tokens, all in one self-attended sequence | `MotionDiffusionEnv` | `network/models_env2d.py` |
| **DIPTRAJ** | `trans_dec` | per-frame traj tokens kept as cross-attention memory (CAMDM-style routing through a DiP-style backbone) | `MotionDiffusionDipTraj` | `network/models_dip2d_traj.py` |
| **DIPCMD** | `trans_dec` | single body-frame twist `(vx, vy, ω)` token in memory; **no** per-frame traj | `MotionDiffusionDipCmd` | `network/models_dip2d_cmd.py` |

CAMDM's per-frame XY+heading is what gives it strong path-following.
DIPTRAJ keeps that per-frame steering signal but routes it through
DiP's cross-attention so it stays read-only. DIPCMD asks "do we need
per-frame steering at all, or is one global twist enough?"


## 1. `interface(x, timesteps, y)` contract

The training portal sees three slightly different `y` dicts:

| key | shape | CAMDM | DIPTRAJ | DIPCMD |
| --- | --- | :-: | :-: | :-: |
| `past_motion` | (bs, J, F, TP) | ✓ | ✓ | ✓ |
| `traj_pose`   | (bs, 6, TF)    | ✓ | ✓ |   |
| `traj_trans`  | (bs, 2, TF)    | ✓ | ✓ |   |
| `command`     | (bs, 3)        |   |   | ✓ |
| `style_idx`   | (bs,)          | ✓ | ✓ | ✓ |
| `sensor`      | (bs, 226)      | ✓ | ✓ | ✓ |
| `mask`        | (bs, TF)       | ✓ | ✓ | ✓ |

Returns `(bs, J, F, TF)` denoised future motion in all three cases.
DIPCMD swaps per-frame `traj_*` for a single `command` produced by the
companion dataset (`network/dataset_g1_env2d_cmd.py`):

```
vx = traj_trans_per_frame[current, -1, 0] / (TF * dt)
vy = traj_trans_per_frame[current, -1, 1] / (TF * dt)
ω  = yaw_of(traj_pose_per_frame[current, -1]) / (TF * dt)
```


## 2. CAMDM — `trans_enc`, separate per-source tokens

```
 pos:   0      1       2..46          47..91          92..93       94..138
        ┌────┐ ┌─────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────┐
xseq:   │time│ │style│ │ traj_trans │ │ traj_pose  │ │  past    │ │ noisy future   │
        │ (1)│ │ (1) │ │   (TF=45)  │ │   (TF=45)  │ │ motion(2)│ │  motion (TF=45)│
        └────┘ └─────┘ └────────────┘ └────────────┘ └──────────┘ └────────────────┘
                                                                  └── output sliced
```

Total tokens for `past_frame=2`: **2 + 2*45 + 2 + 45 = 139** (plus +1 if
the env-sensor variant adds a sensor token). All in one tensor; one
self-attention pass per layer. Every motion frame attends to every
other token, including the per-frame traj at its own index.


## 3. DIPTRAJ — `trans_dec`, per-frame traj in memory

```
 memory  (cross-attn keys/values, read-only):
        ┌────┐ ┌─────┐ ┌──────┐ ┌────────────┐ ┌────────────┐
mem:    │time│ │style│ │sensor│ │ traj_trans │ │ traj_pose  │    (3 + 2*TF, bs, L)
        │(1) │ │ (1) │ │ (1)  │ │   (TF=45)  │ │   (TF=45)  │
        └────┘ └─────┘ └──────┘ └────────────┘ └────────────┘

 target  (self-attn within these; queries memory):
        ┌──────────┐ ┌────────────────┐
tgt:    │  past    │ │  noisy future  │                          (TP + TF, bs, L)
        │ motion(2)│ │   motion (45)  │
        └──────────┘ └────────────────┘
                     └── output sliced (last TF)
```

For `past_frame=2`: **93 memory + 47 target = 140 tokens** total but
split across two attention paths.

Each future-motion frame computes a cross-attention softmax over 93
memory slots — it can dynamically attend to *its specific*
`traj_trans_k` token vs the global cues, and the memory never gets
mutated by the motion (read-only).


## 4. DIPCMD — `trans_dec`, single command token in memory

```
 memory  (cross-attn keys/values, read-only):
        ┌────┐ ┌─────┐ ┌──────┐ ┌─────┐
mem:    │time│ │style│ │sensor│ │ cmd │                          (4, bs, L)
        │(1) │ │ (1) │ │ (1)  │ │ (1) │
        └────┘ └─────┘ └──────┘ └─────┘

 target:
        ┌──────────┐ ┌────────────────┐
tgt:    │  past    │ │  noisy future  │                          (TP + TF, bs, L)
        │ motion(2)│ │   motion (45)  │
        └──────────┘ └────────────────┘
                     └── output sliced
```

For `past_frame=2`: **4 memory + 47 target = 51 tokens** total. The
steering reduces from a 45-frame XY+heading trajectory (≈ 360 numbers)
to a single 3-D vector (vx, vy, ω). Conditioning capacity is the
absolute minimum here; this variant tests whether the model can plan
its own path from just a desired twist + the obstacle sensor.


## 5. Reference: DiP (CLoSD's released checkpoint)

For context, vanilla DiP uses:

```
 memory:  time_emb (1) + BERT text tokens (N ≈ 20–30)             (1+N, bs, 512)
 tgt:     prefix (clean, 20) + noisy future (40)                  (60,   bs, 512)
```

`arch=trans_dec, emb_policy=concat, latent_dim=512, num_layers=8,
context_len=20, pred_len=40, diffusion_steps=10`.

DIPTRAJ and DIPCMD pick up DiP's `trans_dec + concat` backbone choice
but substitute the data-relevant inputs (sensor + traj or sensor + cmd)
for DiP's text tokens.


## 6. Side-by-side summary

| Aspect | CAMDM | DIPTRAJ | DIPCMD |
| --- | --- | --- | --- |
| Backbone | trans_enc (self-attn only) | trans_dec (self + cross) | trans_dec (self + cross) |
| Conditioning routing | flat concat, self-attn over all | cross-attn memory (read-only) | cross-attn memory (read-only) |
| Per-frame steering tokens | yes (TF + TF separate) | yes (TF + TF separate, in memory) | no |
| Global steering token | no | no | yes (single 3-D cmd) |
| Total tokens at transformer | ~139 (1 tensor) | ~93 mem + 47 tgt (2 tensors) | 4 mem + 47 tgt (2 tensors) |
| `interface(x, t, y)` shape contract | unchanged | unchanged | `command` replaces `traj_*` |


## 7. Files at a glance

```
# CAMDM (untouched original)
PyTorch/network/models_env2d.py                  # MotionDiffusionEnv
PyTorch/train_g1_env2d.py
PyTorch/config/default_g1_env_geo.json
PyTorch/visualize/step6_demo_control.py

# DIPTRAJ (per-frame traj memory, trans_dec)
PyTorch/network/models_dip2d_traj.py             # MotionDiffusionDipTraj
PyTorch/train_g1_env2d_dip_traj.py
PyTorch/config/default_g1_env_geo_dip_traj.json
PyTorch/visualize/step6_demo_control_dip_traj.py

# DIPCMD (single twist memory, trans_dec)
PyTorch/network/models_dip2d_cmd.py              # MotionDiffusionDipCmd
PyTorch/network/dataset_g1_env2d_cmd.py          # derives (vx, vy, omega) per clip
PyTorch/train_g1_env2d_dip_cmd.py
PyTorch/config/default_g1_env_geo_dip_cmd.json
PyTorch/visualize/step6_demo_control_cmd.py
```


## 8. Training invocations

All three use `past_frame=2`, the `lafan1_g1_past2_motion30_env2d_yellow_circle_*_none.pkl`
dataset, lr=3e-4, batch=512, epoch=3000, geo losses on, wandb on.

```bash
# CAMDM
python train_g1_env2d.py \
    -n camdm_yc_past2_ep3k \
    -c config/default_g1_env_geo.json \
    -i data/pkls/lafan1_g1_past2_motion30_env2d_yellow_circle_detour_only_detour_sparse_detour_dense_detour_packed_dense_none.pkl \
    --past_frame 2 \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group camdm

# DIPTRAJ
python train_g1_env2d_dip_traj.py \
    -n dip_traj_yc_past2_ep3k \
    -c config/default_g1_env_geo_dip_traj.json \
    -i data/pkls/lafan1_g1_past2_motion30_env2d_yellow_circle_detour_only_detour_sparse_detour_dense_detour_packed_dense_none.pkl \
    --past_frame 2 \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group dip_traj

# DIPCMD
python train_g1_env2d_dip_cmd.py \
    -n dip_cmd_yc_past2_ep3k \
    -c config/default_g1_env_geo_dip_cmd.json \
    -i data/pkls/lafan1_g1_past2_motion30_env2d_yellow_circle_detour_only_detour_sparse_detour_dense_detour_packed_dense_none.pkl \
    --past_frame 2 \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group dip_cmd
```

On sky1 the matching sbatch scripts live at `~/flash/slurm/CAMDM/`:
`run_camdm_yc.sh`, `run_dip_traj.sh`, `run_dip_cmd.sh`.
