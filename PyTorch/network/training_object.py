from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from diffusion.respace import SpacedDiffusion

import torch
from diffusion.gaussian_diffusion import ModelVarType, ModelMeanType, LossType
from network.training import BaseTrainingPortal
import utils.common as common
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import os


class HumanoidSingleObjectTrainingPortal(BaseTrainingPortal):
    """
    Training portal for single-object CAMDM.
    Denoises concatenated vector:
    [full body state, object pose (12), contact (1)].
    """

    def __init__(self, config, model, diffusion: "SpacedDiffusion", dataloader, logger, tb_writer, finetune_loader=None):
        super().__init__(config, model, diffusion, dataloader, logger, tb_writer, finetune_loader)

    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        batch_size, frame_num, state_dim, state_feat = x_start.shape
        x_start = x_start.permute(0, 2, 3, 1)  # [B, D, 1, T]

        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        cond["past_motion"] = cond["past_motion"].permute(0, 2, 3, 1)   # [B, D, 1, Tp]
        cond["traj_pose"] = cond["traj_pose"].permute(0, 2, 1)           # [B, 6, Tf]
        cond["traj_trans"] = cond["traj_trans"].permute(0, 2, 1)         # [B, 2, Tf]
        cond["traj_contact"] = cond["traj_contact"].permute(0, 2, 1)     # [B, 1, Tf]

        model_output = self.model.interface(x_t, self.diffusion._scale_timesteps(t), cond)

        if not return_loss:
            return model_output.permute(0, 3, 1, 2)

        loss_terms = {}
        if self.diffusion.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            loss_terms["vb"] = self.diffusion._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )["output"]
            if self.loss_type == LossType.RESCALED_MSE:
                loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0

        target = {
            ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.diffusion.model_mean_type]

        assert model_output.shape == target.shape == x_start.shape
        mask = cond["mask"].view(batch_size, 1, 1, -1)

        if self.config.trainer.use_loss_mse:
            loss_terms["loss_data"] = self.diffusion.masked_l2(target, model_output, mask)

        if self.config.trainer.use_loss_vel:
            model_output_vel = model_output[..., 1:] - model_output[..., :-1]
            target_vel = target[..., 1:] - target[..., :-1]
            loss_terms["loss_data_vel"] = self.diffusion.masked_l2(target_vel[:, :-1], model_output_vel[:, :-1], mask[..., 1:])

        loss_terms["loss"] = (
            loss_terms.get("vb", 0.0)
            + loss_terms.get("loss_data", 0.0)
            + loss_terms.get("loss_data_vel", 0.0)
        )
        return model_output.permute(0, 3, 1, 2), loss_terms

    def run_loop(self):
        sampling_num = 16
        sampling_idx = np.random.randint(0, len(self.dataloader.dataset), sampling_num)
        sampling_subset = DataLoader(Subset(self.dataloader.dataset, sampling_idx), batch_size=sampling_num)
        self.evaluate_sampling(sampling_subset, save_folder_name="init_samples")

        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f"Epoch {self.epoch}")
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx
            epoch_losses = {}

            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas["conditions"].items()}
                x_start = datas["data"]

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                total_loss = (losses["loss"] * weights).mean()
                total_loss.backward()
                self.opt.step()

                if self.config.trainer.ema:
                    self.ema.update()

                for key_name in losses.keys():
                    if "loss" in key_name:
                        if key_name not in epoch_losses:
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())

            loss_str = ""
            for key in epoch_losses.keys():
                loss_str += f"{key}: {np.mean(epoch_losses[key]):.6f}, "

            epoch_avg_loss = np.mean(epoch_losses["loss"])

            if self.epoch > 10 and epoch_avg_loss < self.best_loss:
                self.save_checkpoint(filename="best")

            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss

            epoch_process_bar.set_description(
                f"Epoch {epoch_idx}/{self.config.trainer.epoch} | "
                f"loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}"
            )
            self.logger.info(f"Epoch {epoch_idx}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}")

            if epoch_idx > 0 and epoch_idx % self.config.trainer.save_freq == 0:
                self.save_checkpoint(filename=f"weights_{epoch_idx}")
                self.evaluate_sampling(sampling_subset, save_folder_name="train_samples")

            self._log_epoch_metrics(epoch_losses, epoch_idx)

            self.scheduler.step()

        best_path = f"{self.config.save}/best.pt"
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)
            self.evaluate_sampling(sampling_subset, save_folder_name="best")
        else:
            self.logger.info(
                f"Skip loading best checkpoint because it does not exist yet: {best_path}. "
                "This is expected for very short runs (<=10 epochs)."
            )
        self._finish_wandb()

    def evaluate_sampling(self, dataloader, save_folder_name):
        self.model.eval()
        self.model.training = False
        save_path = f"{self.save_dir}/{save_folder_name}"
        common.mkdir(save_path)

        datas = next(iter(dataloader))
        datas = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in datas.items()}
        cond = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in datas["conditions"].items()}
        x_start = datas["data"]
        t, _ = self.schedule_sampler.sample(dataloader.batch_size, self.device)

        with torch.no_grad():
            model_output = self.diffuse(x_start, t, cond, noise=None, return_loss=False)

        torch.save(
            {
                "gt": x_start.cpu(),
                "pred": model_output.cpu(),
                "past_motion": cond["past_motion"].cpu(),
                "traj_contact": cond["traj_contact"].cpu(),
            },
            f"{save_path}/samples.pt",
        )
        self.logger.info(f"Sampling saved at epoch {self.epoch} -> {save_path}")
