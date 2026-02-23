# G1 Single Object Quickstart

This README is for G1 single-object pipeline only.

## Setup
```bash
conda env create -f environment.yml
conda activate camdm
```

Optional W&B:
```bash
pip install wandb
wandb login
```

## Stage 1: Build Merged Dataset
```bash
python make_merged_single_object_pkl.py --output data/pkls/merged_object_motion.pkl
```

## Stage 1: Visual Check
```bash
python visualize/step2_visualize_data_object_pkl.py --dataset data/pkls/merged_object_motion.pkl
```

## Stage 2: Train Single-Object Model
```bash
python train_g1_object.py -n camdm_g1_object_test -c ./config/single_object_g1.json --epoch 10000 --batch_size 512 --data data/pkls/merged_object_motion.pkl
```

Explicit 8-step variant:
```bash
python train_g1_object.py -n camdm_g1_object_test -c ./config/single_object_g1.json --epoch 10000 --batch_size 512 --diffusion_steps 8 --data data/pkls/merged_object_motion.pkl
```

## W&B Training Command
```bash
python train_g1_object.py -n camdm_g1_object_wandb -c ./config/single_object_g1.json --data data/pkls/merged_object_motion.pkl --wandb --wandb_project CAMDM --wandb_group g1_object
```

## Stage 3: Interactive Demo
```bash
python visualize/step3_demo_object.py --checkpoint <path_to_checkpoint.pt> --dataset data/pkls/merged_object_motion.pkl
```

Controls: `P=pick`, `O=drop`, `Space=pause`, `S=status`

## Tiny Troubleshooting
- `W&B requested but wandb is not installed`:
  - `pip install wandb`
  - `wandb login`
