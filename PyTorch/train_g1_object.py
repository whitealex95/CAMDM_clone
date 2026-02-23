import os
import time
import torch
import shutil
import argparse
import utils.common as common

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import Logger
from network.models_object import MotionDiffusionObject
from network.training_object import HumanoidSingleObjectTrainingPortal
from network.dataset_g1_object import SingleObjectMotionDataset

from diffusion.create_diffusion import create_gaussian_diffusion
from config.option import add_model_args, add_train_args, add_diffusion_args, config_parse


def train(config, resume, logger, tb_writer):
    common.fixseed(1024)
    np_dtype = common.select_platform(32)

    print("Loading single-object dataset..")
    train_data = SingleObjectMotionDataset(
        config.data,
        config.arch.rot_req,
        config.arch.offset_frame,
        config.arch.past_frame,
        config.arch.future_frame,
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
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
        "\nTraining Dataset including %d clips, with %d frame per clip;"
        % (len(train_data), config.arch.clip_len)
    )

    diffusion = create_gaussian_diffusion(config)

    input_feats = train_data.state_dim * 1
    model = MotionDiffusionObject(
        input_feats=input_feats,
        nstyles=len(train_data.style_set),
        njoints=train_data.state_dim,
        nfeats=1,
        rot_req=config.arch.rot_req,
        clip_len=config.arch.clip_len,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        traj_pose_feats=train_data.per_rot_feat,
        traj_trans_feats=2,
        traj_contact_feats=1,
        traj_obj_pose_feats=train_data.per_rot_feat,
        traj_obj_trans_feats=2,
        device=config.device,
    ).to(config.device)

    trainer = HumanoidSingleObjectTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer)

    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
        except FileNotFoundError:
            print("No checkpoint found at %s" % resume)
            exit()

    trainer.run_loop()


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="### Generative Locomotion Training (Single Object)")
    parser.add_argument("-n", "--name", default="camdm_g1_object_debug", type=str, help="The name of this training")
    parser.add_argument("-c", "--config", default="./config/single_object_g1.json", type=str, help="config file path")
    parser.add_argument("-i", "--data", default="data/pkls/merged_object_motion.pkl", type=str)
    parser.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint")
    parser.add_argument("-s", "--save", default="./save", type=str, help="save dir")
    parser.add_argument("--cluster", action="store_true", help="train with GPU cluster")
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
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if "debug" in args.name:
        config.arch.offset_frame = config.arch.clip_len
        config.trainer.workers = 1
        config.trainer.load_num = None
        config.trainer.batch_size = 64

    if not args.cluster:
        if os.path.exists(config.save) and "debug" not in args.name and args.resume is None:
            allow_cover = input(f"Model file({config.save}) detected, do you want to replace it? (Y/N)")
            allow_cover = allow_cover.lower()
            if allow_cover == "n":
                exit()
            shutil.rmtree(config.save, ignore_errors=True)
    else:
        if os.path.exists(config.save):
            if os.path.exists("%s/best.pt" % config.save):
                args.resume = "%s/best.pt" % config.save
            else:
                existing_pths = [val for val in os.listdir(config.save) if "weights_" in val]
                if len(existing_pths) > 0:
                    epoches = [int(filename.split("_")[1].split(".")[0]) for filename in existing_pths]
                    args.resume = "%s/%s" % (config.save, "weights_%s.pt" % max(epoches))

    os.makedirs(config.save, exist_ok=True)
    logger = Logger("%s/log.txt" % config.save)
    tb_writer = SummaryWriter(log_dir="%s/runtime" % config.save)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("%s/config.json" % config.save, "w") as f:
        f.write(str(config))

    logger.info("\nSingle-object generative locomotion training with config: \n%s" % config)
    train(config, args.resume, logger, tb_writer)
    logger.info("\nTotal training time: %s mins" % ((time.time() - start_time) / 60))
