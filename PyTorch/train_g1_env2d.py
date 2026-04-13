"""
Train G1 Humanoid Motion Diffusion with Environment-Sensor Conditioning
------------------------------------------------------------------------
Identical to train_g1.py but uses:
  * HumanoidEnvMotionDataset  (network/dataset_g1_env2d.py)
  * MotionDiffusionEnv        (network/models_env2d.py)

The training dataset must have been created with:
    python visualize/step2_visualize_data_env2d.py --create-dataset [--mode sparse|dense|packed]

Usage
-----
    python train_g1_env2d.py -n my_env2d_run \\
        -c config/default_g1_env.json \\
        -i data/pkls/lafan1_g1_env2d_sparse.pkl
"""

import os
import time
import torch
import shutil
import argparse
import utils.common as common

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import Logger
from network.models_env2d import MotionDiffusionEnv
from network.training import HumanoidTrainingPortal
from network.dataset_g1_env2d import HumanoidEnvMotionDataset

from diffusion.create_diffusion import create_gaussian_diffusion
from config.option import add_model_args, add_train_args, add_diffusion_args, config_parse


def train(config, resume, logger, tb_writer):

    common.fixseed(1024)
    np_dtype = common.select_platform(32)

    print("Loading dataset …")
    train_data = HumanoidEnvMotionDataset(
        config.data,
        config.arch.rot_req,
        config.arch.offset_frame,
        config.arch.past_frame,
        config.arch.future_frame,
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
        min_start_velocity=config.trainer.min_start_velocity,
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
    )
    logger.info(
        "\nTraining Dataset: %d clips, %d frames per clip"
        % (len(train_data), config.arch.clip_len)
    )

    diffusion = create_gaussian_diffusion(config)

    # env_sensor_dim from config, falling back to the value stored in the dataset
    env_sensor_dim = getattr(config.arch, "env_sensor_dim", train_data.env_sensor_dim)

    input_feats = (train_data.joint_num + 1) * train_data.per_rot_feat

    model = MotionDiffusionEnv(
        input_feats,
        len(train_data.style_set),
        train_data.joint_num + 1,
        train_data.per_rot_feat,
        config.arch.rot_req,
        config.arch.clip_len,
        env_sensor_dim=env_sensor_dim,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=config.device,
    ).to(config.device)

    logger.info(f"MotionDiffusionEnv: env_sensor_dim={env_sensor_dim}")

    # HumanoidTrainingPortal handles the training loop and geometric losses.
    # It reads 'sensor' from cond dict and permutes it before passing to the model.
    # We subclass it here to add the sensor permutation.
    trainer = _EnvTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer)

    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
        except FileNotFoundError:
            print(f"No checkpoint found at {resume}")
            exit()

    trainer.run_loop()


# ---------------------------------------------------------------------------
# Thin wrapper: adds sensor permutation to the training step
# ---------------------------------------------------------------------------

class _EnvTrainingPortal(HumanoidTrainingPortal):
    """
    Extends HumanoidTrainingPortal to handle the 'sensor' condition.

    The sensor tensor from the DataLoader is already (bs, env_sensor_dim)
    and requires no permutation before being passed to model.interface().
    """

    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="G1 Env-Sensor Motion Diffusion Training")
    parser.add_argument("-n", "--name",   default="debug_env", type=str)
    parser.add_argument("-c", "--config", default="./config/default_g1_env.json", type=str)
    parser.add_argument("-i", "--data",   default="data/pkls/lafan1_g1_env_sparse.pkl", type=str)
    parser.add_argument("-r", "--resume", default=None, type=str)
    parser.add_argument("-s", "--save",   default="./save", type=str)
    parser.add_argument("--cluster",      action="store_true")
    add_model_args(parser)
    add_diffusion_args(parser)
    add_train_args(parser)

    args = parser.parse_args()

    if args.cluster:
        args.data = "xxxxxxx/pkls/" + args.data.split("/")[-1]
        args.save = "xxxxx"

    if args.config:
        config = config_parse(args)
    else:
        raise AssertionError("Configuration file must be specified (-c config.json).")

    if "debug" in args.name:
        config.arch.offset_frame = config.arch.clip_len
        config.trainer.workers = 1
        config.trainer.load_num = None
        config.trainer.batch_size = 256

    if not args.cluster:
        if os.path.exists(config.save) and "debug" not in args.name and args.resume is None:
            allow = input(f"Model dir ({config.save}) exists. Overwrite? (Y/N): ").lower()
            if allow == "n":
                exit()
            else:
                shutil.rmtree(config.save, ignore_errors=True)
    else:
        if os.path.exists(config.save):
            if os.path.exists(f"{config.save}/best.pt"):
                args.resume = f"{config.save}/best.pt"
            else:
                existing = [v for v in os.listdir(config.save) if "weights_" in v]
                if existing:
                    epochs = [int(f.split("_")[1].split(".")[0]) for f in existing]
                    args.resume = f"{config.save}/weights_{max(epochs)}.pt"

    os.makedirs(config.save, exist_ok=True)

    logger = Logger(f"{config.save}/log.txt")
    tb_writer = SummaryWriter(log_dir=f"{config.save}/runtime")

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{config.save}/config.json", "w") as f:
        f.write(str(config))

    logger.info(f"\nEnv-Sensor motion training with config:\n{config}")
    train(config, args.resume, logger, tb_writer)
    logger.info(f"\nTotal training time: {(time.time() - start_time) / 60:.1f} mins")
