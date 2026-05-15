# CAMDM project notes for Claude

This file captures the workflow for running CAMDM jobs on the **sky1**
SLURM cluster at Georgia Tech, plus the file layout for the
multi-variant DiP work that lives on the `*_avoid2d` branches.


## Skynet workflow (run training there)

### 1. Connect

```bash
ssh sky1                    # → sky1.cc.gatech.edu, login node
```

**Paths on sky1** (note that `~/flash` is a per-session symlink — the
sbatch scripts re-create it on each compute node):

| Local path                         | sky1 path                                        |
| ---------------------------------- | ------------------------------------------------ |
| `~/Projects/CAMDM`                 | `~/flash/Projects/CAMDM`                         |
| `~/Projects/CAMDM/PyTorch`         | `~/flash/Projects/CAMDM/PyTorch`                 |
| (slurm jobscripts)                 | `~/flash/slurm/CAMDM`                            |
| (real flash storage)               | `/coc/flash2/jkim3662`                           |

### 2. SLURM binaries are not on the default non-interactive PATH

When ssh-ing in via a script (`ssh sky1 'sbatch ...'`), the login shell
isn't sourced, so `sbatch` / `squeue` / `gpu_usage` aren't on PATH.
Two options:

- **From a Bash tool**: use full paths.
  ```bash
  /opt/slurm/Ubuntu-20.04/current/bin/sbatch  run_X.sh
  /opt/slurm/Ubuntu-20.04/current/bin/squeue  --user jkim3662
  /coc/testnvme/admin/tools/skynet-utilities/gpu_usage -l
  ```
- **Interactive use**: `ssh sky1 -t 'bash -lc "sbatch run_X.sh"'`
  (force a login shell to pick up `~/.bash_aliases`).

The user-facing aliases that exist after login:
- `sq` — formatted squeue for the current user
- `gpu_usage -l` — GPU availability per lab account

### 3. Pull the latest code

```bash
ssh sky1 'cd ~/flash/Projects/CAMDM && git fetch origin && git checkout <branch> && git pull'
```

The repo's remote is `git@github.com:whitealex95/CAMDM_clone.git`.

### 4. SLURM script template

`~/flash/slurm/CAMDM/run_camdm.sh` (and the variant scripts) follow this
shape — note especially the `ln -snf` line; without it the compute node
won't be able to resolve `$HOME/flash`:

```bash
#!/bin/bash
#SBATCH --job-name=<short_name>
#SBATCH --output=logs/%j_<short_name>.out
#SBATCH --error=logs/%j_<short_name>.err
#SBATCH --partition=ha-lab
#SBATCH -c 8
#SBATCH --qos short
#SBATCH --gres=gpu:a40:1

mkdir -p logs

# Re-create the per-node ~/flash symlink to the real flash storage.
REAL_FLASH="/coc/flash2/jkim3662"
ln -snf $REAL_FLASH $HOME/flash

cd /nethome/jkim3662/flash/Projects/CAMDM/PyTorch
source /nethome/jkim3662/flash/miniconda3/etc/profile.d/conda.sh
conda activate camdm

python <train_script>.py \
    -n <run_name> \
    -c <config.json> \
    -i data/pkls/<dataset>.pkl \
    --epoch 3000 --lr 0.0003 --batch_size 512 --workers 8 \
    --wandb --wandb_project CAMDM --wandb_group <group>
```

### 5. Submit and monitor

```bash
# Submit
ssh sky1 'cd ~/flash/slurm/CAMDM && /opt/slurm/Ubuntu-20.04/current/bin/sbatch run_X.sh'

# Watch the queue (locally — DO NOT poll from a Claude session; one check is enough)
ssh sky1 -t 'bash -lc "watch -n 2 squeue --user jkim3662"'

# Or single-shot
ssh sky1 '/opt/slurm/Ubuntu-20.04/current/bin/squeue --user jkim3662'

# Why is my job pending?
#   - "(QOSGrpGRES)"  → too many of your jobs are already using GPU/CPU
#   - "(Priority)"    → waiting in queue behind other users' work
#   - "(Resources)"   → no eligible node free yet
ssh sky1 -t 'bash -lc "gpu_usage -l"'   # check ha-lab row
```

ha-lab has **12 a40 GPUs**; CPU is **192**, which is the bottleneck more
often than the GPU.

### 6. Tail a running job

```bash
ssh sky1 'tail -f ~/flash/slurm/CAMDM/logs/<job_id>_<short_name>.out'
```


## Active SLURM jobscripts for this work

Located at `~/flash/slurm/CAMDM/` on sky1:

| Script | Trains | Class | Config |
| --- | --- | --- | --- |
| `run_camdm_yc.sh` | CAMDM (trans_enc, separate per-frame traj tokens) | `MotionDiffusionEnv` | `default_g1_env_geo.json` |
| `run_dip_traj.sh` | DIPTRAJ (trans_dec, per-frame traj memory) | `MotionDiffusionDipTraj` | `default_g1_env_geo_dip_traj.json` |
| `run_dip_cmd.sh`  | DIPCMD (trans_dec, single twist memory) | `MotionDiffusionDipCmd` | `default_g1_env_geo_dip_cmd.json` |
| `run_camdm.sh` | (legacy band_yellow CAMDM run, do not use) | `MotionDiffusionEnv` | `default_g1_env_geo.json` |

All three current scripts:
- Live on branch `dip_traj_dec_avoid2d` (file layout:
  `models_env2d.py` for CAMDM, `models_dip2d_traj.py` for DIPTRAJ,
  `models_dip2d_cmd.py` for DIPCMD).
- Train on `data/pkls/lafan1_g1_past2_motion30_env2d_yellow_circle_detour_only_detour_sparse_detour_dense_detour_packed_dense_none.pkl`
  with `--past_frame 2` (matching the local vEnvGeoV1 reference run).
- 3000 epochs, lr=3e-4, batch=512, workers=8, geo losses on, wandb on.

To re-run any of them after a code change:

```bash
# 1. Push from local
git push origin dip_traj_dec_avoid2d

# 2. Pull on sky1
ssh sky1 'cd ~/flash/Projects/CAMDM && git pull'

# 3. Resubmit
ssh sky1 'cd ~/flash/slurm/CAMDM && /opt/slurm/Ubuntu-20.04/current/bin/sbatch run_dip_traj_dec.sh'
```


## Where things land

- **Checkpoints / logs**: `~/flash/Projects/CAMDM/PyTorch/save/<run_name>/`
  - `best.pt` — lowest training loss seen
  - `log.txt` — full training log
  - `wandb/` — wandb local cache
- **SLURM stdout/stderr**: `~/flash/slurm/CAMDM/logs/<job_id>_<short_name>.{out,err}`
- **wandb runs**: project `CAMDM`, entity `jkim3662-gt`
  - https://wandb.ai/jkim3662-gt/CAMDM


## DiP variant cheat sheet (for code navigation)

Three primary architectural choices:

| Tag | Backbone | Conditioning routing | Class | File |
| --- | --- | --- | --- | --- |
| **CAMDM** | `trans_enc` | per-frame traj as separate tokens in one self-attended sequence | `MotionDiffusionEnv` | `network/models_env2d.py` |
| **DIPTRAJ** | `trans_dec` | per-frame traj as cross-attention memory tokens | `MotionDiffusionDipTraj` | `network/models_dip2d_traj.py` |
| **DIPCMD** | `trans_dec` | single `(vx, vy, ω)` cross-attention memory token | `MotionDiffusionDipCmd` | `network/models_dip2d_cmd.py` |

Full layout/ASCII diagrams live in `PyTorch/README_dip_variants.md`.


## Run-log convention (gitignored)

Every time you submit a skynet job from a Claude session, **append an
entry to `skynet_runs.md`** at the repo root. That file is gitignored
(personal tracking only) and serves as a durable record of what was
running on sky1 at any given commit.

Each entry should include:
- timestamp
- branch and short commit hash (the one on sky1 when the job was submitted)
- slurm job IDs and the matching script names
- the exact submission command used locally
- per-job: run name, wandb group, log path
- a "status at submission time" line, and a "status (fill in later)" stub

Use the template comment at the bottom of `skynet_runs.md` to start a new
entry. Newest entries go at the top.

When a run completes (or is killed), update the "Status (fill in later)"
line of its entry with the outcome (best_loss, total runtime, anything
notable).


## Things NOT to do

- Don't `ssh sky1 'sbatch …'` without the full path — it'll silently
  fail with `sbatch: command not found`.
- Don't `watch -n 2 squeue …` from a Claude Bash call — it never exits.
  Run one-shot `squeue` instead, or rely on tailing the slurm log file.
- Don't push to `main`; the active branches for this work are
  `dip_avoid2d`, `cmd_dip_avoid2d`, `dip_traj_dec_avoid2d`.
