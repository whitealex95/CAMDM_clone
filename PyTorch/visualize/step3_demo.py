"""
Step 2: Visualize Training Data
--------------------------------
Load motion clips from pickle files and play them in MuJoCo viewer.

This allows you to:
- Browse through different motion clips
- Select different styles
- Verify that data loads correctly and animations look natural
- Check coordinate systems and joint ranges

Controls:
- SPACE: Pause/Resume
- LEFT/RIGHT: Previous/Next frame
- UP/DOWN: Previous/Next motion clip
- R: Reset to first frame
- 1-9: Change playback speed
- ESC: Exit

Usage:
    python visualize/step2_visualize_data.py [--dataset lafan1_g1]
"""

import os
import sys
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models import MotionDiffusion
from diffusion.create_diffusion import create_gaussian_diffusion

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import draw_trajectory
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic, align_trajectory_to_pose
from visualize.utils.trajectory import match_future_horizon

import torch


class ModelWrapper(torch.nn.Module):
    """Wrapper to make the model compatible with diffusion sampler."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, timesteps, **kwargs):
        return self.model.forward(x, timesteps,
                                  kwargs.get('past_motion'),
                                  kwargs.get('traj_pose'),
                                  kwargs.get('traj_trans'),
                                  kwargs.get('style_idx'))
class MotionGenerator:
    """Autoregressive motion generator with Dataset Guidance."""
    
    def __init__(self, model, diffusion, config, device="cuda", sampler="ddpm", cfg_scale=1.0):
        self.model = ModelWrapper(model)
        self.diffusion = diffusion
        self.device = device
        self.sampler = sampler.lower()
        self.cfg_scale = float(cfg_scale)
        
        # Model parameters
        self.future_frames = config.arch.future_frame
        self.joint_num = 30 # 1 + 29
        self.rot_req = config.arch.rot_req
        self.per_rot_feat = 6 # 6D rotation representation
            
    def generate_motion(self, past_qpos, traj_trans, traj_pose, style_idx, cfg_scale=None):
        """
        Generate future motions given past motion and trajectory conditions.
        past_qpos: (past_frames, 36=7+29) numpy array
        traj_trans: (future_frames, 2) numpy array, XY in global frame
        traj_pose: (future_frames, 4) numpy array, orientation in global frame (wxyz)
        style_idx: int, style index
        Returns:
            generated_qpos: (future_frames, 36=7+29) numpy array
        """
        # Obtain current root position for centering
        curr_root_XY = past_qpos[-1, :2].copy()  # (2,)

        # A. Prepare Past Motion Conditions
        past_qpos_centered = past_qpos.copy()
        past_qpos_centered[:, :2] -= curr_root_XY[None, :]  # Center XY
        past_motion = qpos_to_model_format(past_qpos_centered)  # (past_frames, 31, feat)
        past_motion_tensor = torch.from_numpy(past_motion)\
            .float().unsqueeze(0).permute(0, 2, 3, 1).to(self.device)  # (1, 31, feat, past)

        # B. Prepare Trajectory Conditions 
        traj_trans_centered =\
            traj_trans - curr_root_XY[None, :] # (future, 2)
        traj_trans_tensor = torch.from_numpy(traj_trans_centered)\
            .float().unsqueeze(0).permute(0, 2, 1).to(self.device)  # (1, 2, future)
        
        traj_pose_repr = nn_transforms.get_rotation(
            torch.from_numpy(traj_pose).float(), self.rot_req
        ).numpy()
        traj_pose_tensor = torch.from_numpy(traj_pose_repr)\
            .float().unsqueeze(0).permute(0, 2, 1).to(self.device)  # (1, feat, future)
        
        style_idx_tensor = torch.tensor([style_idx]).to(self.device) # (1)
        
        # C. Generate
        model_kwargs = {
            'past_motion': past_motion_tensor, # (1, 31, feat, past)
            'traj_trans': traj_trans_tensor, # (1, 2, future)
            'traj_pose': traj_pose_tensor, # (1, feat, future)
            'style_idx': style_idx_tensor, # (1)
            'y': {}
        }
        uncond_model_kwargs = {
            'past_motion': torch.zeros_like(past_motion_tensor), # CAMDM uncond: empty past motion
            'traj_trans': traj_trans_tensor,
            'traj_pose': traj_pose_tensor,
            'style_idx': style_idx_tensor,
            'y': {}
        }
        guidance_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if guidance_scale == 1.0:
            sampling_model = self.model
            sampling_kwargs = model_kwargs
        else:
            class _CFGWrapper(torch.nn.Module):
                def __init__(self, cond_model, uncond_kwargs, scale):
                    super().__init__()
                    self.cond_model = cond_model
                    self.uncond_kwargs = uncond_kwargs
                    self.scale = scale

                def forward(self, x, timesteps, **kwargs):
                    pred_cond = self.cond_model(x, timesteps, **kwargs)
                    pred_uncond = self.cond_model(x, timesteps, **self.uncond_kwargs)
                    return pred_uncond + self.scale * (pred_cond - pred_uncond)

            sampling_model = _CFGWrapper(self.model, uncond_model_kwargs, guidance_scale)
            sampling_kwargs = model_kwargs
        
        shape = (1, self.joint_num + 1, self.per_rot_feat, self.future_frames)
        if self.sampler == "ddim":
            sample = self.diffusion.ddim_sample_loop(
                sampling_model, shape, clip_denoised=False, model_kwargs=sampling_kwargs,
                progress=False, eta=0.0, device=self.device
            )
        elif self.sampler == "ddpm":
            sample = self.diffusion.p_sample_loop(
                sampling_model, shape, clip_denoised=False, model_kwargs=sampling_kwargs,
                progress=False, device=self.device
            )
        else:
            raise ValueError(f"Unknown sampler '{self.sampler}'. Expected one of: ddpm, ddim")

        # Process Output
        sample = sample.squeeze(0).permute(2, 0, 1).cpu().numpy() # (future, 31, feat)
        generated_qpos = model_format_to_qpos(sample)
        generated_qpos[:, :2] += curr_root_XY[None, :] # Restore global position

        return generated_qpos  # (future, 36)

def qpos_to_model_format(qpos_seq):
    """
    Convert qpos sequence to model input format.
    qpos_seq: (T, 36=7+29) numpy array
    Returns: (T, 31=1+29+1, feat) numpy array
    """
    T = qpos_seq.shape[0]
    joint_num = 30  # 1 root + 29 joints
    rot_req = '6d'
    feat_dim = 6  # for rot_req='6d'
    model_input = np.zeros((T, joint_num + 1, feat_dim), dtype=np.float32)  # +1 for root pos

    for t in range(T):
        root_pos = qpos_seq[t, :3]  # (3,)
        root_quat = qpos_seq[t, 3:7]  # (4,) wxyz
        joint_angles = qpos_seq[t, 7:]  # (29,)

        # Root
        if rot_req == '6d':
            root_rot_repr = nn_transforms.quat2repr6d(torch.from_numpy(root_quat).float().unsqueeze(0)).numpy()[0]
        else:
            raise NotImplementedError(f"Rotation representation '{rot_req}' not implemented.")
        
        model_input[t, 0, :] = root_rot_repr
        model_input[t, 1:30, 0] = joint_angles
        model_input[t, 30, :3] = root_pos

    return model_input  # (T, 31, feat)

def model_format_to_qpos(model_output):
    """
    Convert model output format back to qpos sequence.
    model_output: (T, 31=1+29+1, feat) numpy array
    Returns: (T, 36=7+29) numpy array
    """
    T = model_output.shape[0]
    qpos_seq = np.zeros((T, 36), dtype=np.float32)

    for t in range(T):
        root_rot_repr = model_output[t, 0, :]  # (feat,)
        joint_angles = model_output[t, 1:30, 0]  # (29, feat)
        root_pos = model_output[t, 30, :3]  # (3,)

        qpos_seq[t, :3] = root_pos
        root_quat = nn_transforms.repr6d2quat(torch.from_numpy(root_rot_repr).float().unsqueeze(0)).numpy()[0]
        qpos_seq[t, 3:7] = root_quat
        qpos_seq[t, 7:] = joint_angles
    return qpos_seq  # (T, 36)        

class DemoPlayer:
    def __init__(self, model, data, dataset, motion_generator: MotionGenerator,
                 show_trajectory=True, past_frames=10, future_frames=45,
                 traj_bias_pos=0.4, traj_bias_rot=2.2,
                 cfg_count=2, applyframes=15,
                 inertialize=True,
                 inertialization_mode="camdm",
                 blendtime_rotation=0.2, blendtime_position=0.2,
                 spring_halflife_position=0.12, spring_halflife_rotation=0.12,
                 inertial_quat_start=3, inertial_quat_end=7):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.motion_generator = motion_generator
        self.show_trajectory = show_trajectory
        self.camera_follow = True
        self.past_frames = past_frames
        self.future_frames = future_frames
        
        # Playback state
        self.current_motion_idx = 0
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()
        
        # Frame rate (from make_pose_data_g1.py)
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        # CAMDM uses "applyframes" to control how many predicted frames are applied
        # before running a new diffusion pass.
        self.apply_generated_frames = int(applyframes)
        self.generated_frame_idx = 0

        # Create a queue to store qpos history
        from collections import deque 
        self.qpos_history = deque(maxlen=self.past_frames)
                

        # Generated future poses
        self.generated_qpos = None
        self.generated_future_traj = None
        self.generated_future_orient = None
        # CAMDM baseline trajectory blending coefficients:
        # scale = 1 - (1 - w)^bias, w = (t+1)/T.
        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)
        # CAMDM CFG burst count: use cfg_scale for first N regenerations,
        # then fallback to scale=1.0 until style changes.
        self.cfg_count_cache = int(cfg_count)
        self.cfg_count = int(cfg_count)
        self.prev_style_idx = None

        self.inertialize = bool(inertialize)
        self.inertialization_mode = str(inertialization_mode).lower()
        # CAMDM inertialization parameters.
        self.blendtime_rotation = float(blendtime_rotation)
        self.blendtime_position = float(blendtime_position)
        self.spring_halflife_position = float(spring_halflife_position)
        self.spring_halflife_rotation = float(spring_halflife_rotation)
        # Quaternion segment in a single qpos vector (shape (4,), wxyz; no batch here).
        quat_slice = slice(int(inertial_quat_start), int(inertial_quat_end))
        self.quat_slice = quat_slice
        self.transition_manager = None
        
        # Load first motion
        self.load_motion(motion_idx=27)
        
        

    def load_motion(self, motion_idx):
        """Load a specific motion clip."""
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion_data = self.dataset[self.current_motion_idx]
        self.current_frame = 0
        if self.prev_style_idx is None or self.current_motion_data.style_idx != self.prev_style_idx:
            self.cfg_count = self.cfg_count_cache
        self.prev_style_idx = self.current_motion_data.style_idx
        
        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Style: {self.current_motion_data.style}")
        print(f"Frames: {self.current_motion_data.num_frames}")
        print(f"Duration: {self.current_motion_data.num_frames / self.fps:.2f}s")
        print(f"{'='*60}\n")
        
        self.init_pose()
        self.update_past_trajectory()
        self.update_future_trajectory()
        
    def init_pose(self):
        """Load current frame pose into MuJoCo data."""
        qpos = self.current_motion_data.get_qpos(self.current_frame)
        self.init_qpos_history(qpos.copy())
        # Set the pose
        self.data.qpos[:] = qpos
        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)
    
    def load_past_qpos(self):
        past_qpos_dataset = self.current_motion_data.get_past_qpos(self.current_frame)
        return past_qpos_dataset

    def generate_motion(self):
        """Generate future poses using the motion generator."""
        past_qpos_dataset = self.load_past_qpos()
        if self.current_frame > 0:
            past_qpos = np.array(self.qpos_history)  # (past_frames, 36)
        else:
            past_qpos = past_qpos_dataset
        style_idx = self.current_motion_data.style_idx
        effective_cfg_scale = self.motion_generator.cfg_scale if self.cfg_count > 0 else 1.0

        generated_qpos = self.motion_generator.generate_motion(
            past_qpos, self.future_traj, self.future_orient, style_idx, cfg_scale=effective_cfg_scale
        )
        if self.cfg_count > 0:
            self.cfg_count -= 1
        return generated_qpos # (future_frames, 36)

    def update_pose(self):
        if self.inertialize:
            self.update_pose_inertialized()
        else:
            self.update_pose_raw()

    def update_pose_raw(self):
        """Update MuJoCo model with current frame pose."""
        if self.generated_frame_idx == 0:
            self.generated_qpos = self.generate_motion()

        qpos = self.generated_qpos[self.generated_frame_idx]

        self.update_qpos_history(qpos.copy())
        self.update_past_trajectory()
        self.update_future_trajectory()
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        self.generated_frame_idx = (self.generated_frame_idx + 1) % self.apply_generated_frames

    def update_pose_inertialized(self):
        """Update MuJoCo model with current frame pose using Inertialization."""
        if self.transition_manager is None:
            self.transition_manager = create_transition_manager(
                mode=self.inertialization_mode,
                frame_dt=self.frame_dt,
                quat_slice=self.quat_slice,
                blend_time_rotation=self.blendtime_rotation,
                blend_time_position=self.blendtime_position,
                halflife_position=self.spring_halflife_position,
                halflife_rotation=self.spring_halflife_rotation,
            )

        # Regenerate chunk and trigger transition at chunk boundary.
        if self.generated_frame_idx == 0:
            self.generated_qpos = self.generate_motion()
            # Initialize transition state from previous/current pose to new chunk start.
            self.transition_manager.start_transition(
                self.qpos_history, self.data.qpos.copy(), self.generated_qpos
            )

        raw_target_qpos = self.generated_qpos[self.generated_frame_idx]
        # Apply one-frame inertialization toward the raw target pose.
        final_qpos = self.transition_manager.apply(raw_target_qpos)

        self.update_qpos_history(final_qpos.copy())  # Store the SMOOTHED pose
        self.update_past_trajectory()
        self.update_future_trajectory()

        self.data.qpos[:] = final_qpos
        mujoco.mj_forward(self.model, self.data)

        self.generated_frame_idx = (self.generated_frame_idx + 1) % self.apply_generated_frames

    def init_qpos_history(self, qpos):
        """Initialize qpos history deque."""
        for _ in range(self.past_frames):
            self.qpos_history.append(qpos)

    def update_qpos_history(self, qpos):
        """Update qpos history deque."""
        self.qpos_history.append(qpos) # most recent at the end

    def update_past_trajectory(self):
        """Update past trajectory based on qpos history.
        The resulting past trajectory is fed into the motion generator.
        """
        qpos_history = np.array(self.qpos_history)
        past_xyz = qpos_history[-self.past_frames:, :3]
        past_quat = qpos_history[-self.past_frames:, 3:7] # wxyz
        self.past_traj = past_xyz
        self.past_orient = past_quat
    
    def load_future_trajectory(self, align_to_robot=True):
        # Load raw global trajectory from dataset
        _, future_traj_dataset, _, future_orient_dataset = self._load_trajectory_from_dataset()
        
        if align_to_robot:
            # Get Reference (Dataset) and Current (Robot) states
            ref_qpos = self.current_motion_data.get_qpos(self.current_frame)
            curr_qpos = self.data.qpos.copy()

            # Align trajectory to match the robot's current specific state
            aligned_traj, aligned_orient = align_trajectory_to_pose(
                future_traj_dataset, 
                future_orient_dataset, 
                ref_qpos, 
                curr_qpos
            )
            self.future_traj_dataset, self.future_orient_dataset = match_future_horizon(
                aligned_traj, aligned_orient, self.future_frames
            )
        else:
            # Use raw global data directly
            self.future_traj_dataset, self.future_orient_dataset = match_future_horizon(
                future_traj_dataset, future_orient_dataset, self.future_frames
            )

    def update_future_trajectory(self):
        """Update future trajectory based on target future and predicted future
        target future: self.future_traj_dataset and self.future_orient_dataset
        predicted future: self.generated_qpos
        """
        # Load future trajectory from dataset
        self.load_future_trajectory()

        if self.generated_qpos is not None:
            self.generated_future_traj = self.generated_qpos[:, :3] # XYZ
            self.generated_future_orient = self.generated_qpos[:, 3:7] # wxyz
            t_cur = self.generated_frame_idx
            t_total = self.future_frames
            model_pred_future_traj = self.generated_future_traj[t_cur+1:, :2] # XY only
            model_pred_future_orient = self.generated_future_orient[t_cur+1:]
            extended_future_traj, extended_future_orient = extend_future_traj_heusristic(
                model_pred_future_traj, model_pred_future_orient, t_total, K=1)
            
            blended_future_traj, blended_future_orient = blend_trajectory(extended_future_traj, extended_future_orient,
                                                                          self.future_traj_dataset, self.future_orient_dataset,
                                                                          blend=self.traj_bias_pos, blend_rot=self.traj_bias_rot)
            self.future_traj = blended_future_traj # XY only
            self.future_orient = blended_future_orient # wxyz
        else:
            self.future_traj = self.future_traj_dataset
            self.future_orient = self.future_orient_dataset

    def step(self):
        """Step the animation forward."""
        if not self.playing:
            return
        
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Check if enough time has passed for next frame
        if dt >= self.frame_dt / self.playback_speed:
            self.current_frame += 1
            
            # Loop back to start when reaching end
            if self.current_frame >= self.current_motion_data.num_frames:
                self.current_frame = 0
            
            self.update_pose()
            self.last_update_time = current_time
            
    def next_motion(self):
        """Load next motion clip."""
        self.load_motion(self.current_motion_idx + 1)
    
    def prev_motion(self):
        """Load previous motion clip."""
        self.load_motion(self.current_motion_idx - 1)
    
    def toggle_pause(self):
        """Toggle play/pause."""
        self.playing = not self.playing
        print(f"{'Playing' if self.playing else 'Paused'}")
    
    def reset(self):
        """Reset to first frame."""
        self.current_frame = 0
        self.load_motion(self.current_motion_idx)
        self.update_pose()
        print("Reset to first frame")
    
    def set_speed(self, speed):
        """Set playback speed multiplier."""
        self.playback_speed = speed
        print(f"Playback speed: {speed}x")
    
    def _load_trajectory_from_dataset(self):
        """Get trajectory data for current frame.
        Returns:
            past_traj_dataset: (past_frames, 2) numpy array
            future_traj_dataset: (future_frames, 2) numpy array
            past_orient_dataset: (past_frames, 4) numpy array
            future_orient_dataset: (future_frames, 4) numpy array
        """
        past_traj_dataset, future_traj_dataset, past_orient_dataset, future_orient_dataset = \
            self.current_motion_data.get_trajectory(
                self.current_frame, 
                self.past_frames, 
                self.future_frames,
                kernel_idx=0  # Use first smoothing kernel, choose from 0 or 1
            )
        
        return past_traj_dataset[:, :2], future_traj_dataset[:, :2], past_orient_dataset, future_orient_dataset
    
    def render_trajectory(self, scene):
        if not self.show_trajectory:
            return
        assert hasattr(self, 'past_traj') and hasattr(self, 'future_traj'), \
            "Trajectory data not available for visualization."
        # --- DRAW PAST (Blue) ---
        if len(self.past_traj) > 0:
            draw_trajectory(scene, self.past_traj, self.past_orient, color=[0.2, 0.5, 1.0, 1.0])
        # --- DRAW FUTURE (Red/Green/Gray) ---
        if len(self.future_traj_dataset) > 0:
            draw_trajectory(scene, self.future_traj_dataset, self.future_orient_dataset, color=[1.0, 0.2, 0.2, 1.0])
            draw_trajectory(scene, self.future_traj, self.future_orient, color=[0.2, 1.0, 0.2, 1.0])
            if self.generated_future_traj is not None:
                draw_trajectory(scene, self.generated_future_traj, self.generated_future_orient, color=[0.2, 0.2, 0.2, 0.5])

    def toggle_trajectory(self):
        """Toggle trajectory visualization."""
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")
    
    def toggle_camera_follow(self):
        """Toggle camera follow mode."""
        self.camera_follow = not self.camera_follow
        print(f"Camera follow mode: {'ON' if self.camera_follow else 'OFF'}")

    def print_status(self):
        """Print current status."""
        status = (
            f"Motion: {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"Frame: {self.current_frame}/{self.current_motion_data.num_frames} | "
            f"Style: {self.current_motion_data.style} | "
            f"{'Playing' if self.playing else 'Paused'} "
            f"({self.playback_speed}x) | "
            f"ApplyFrames: {self.apply_generated_frames} | "
            f"Traj: {'ON' if self.show_trajectory else 'OFF'}"
        )
        print(status)

def get_args():
    parser = argparse.ArgumentParser(description="Visualize training motion data")
    # Data/model setup
    parser.add_argument("--dataset", type=str, default="lafan1_g1", help="Dataset name (lafan1_g1 or 100style)")
    parser.add_argument("--checkpoint", type=str, default="save/camdm_g1_lafan1_g1_epoch500_diff8/best.pt")

    # Trajectory conditioning
    parser.add_argument("--traj-bias-pos", type=float, default=0.4, help="CAMDM trajectory position blend bias (bias_HFTE)")
    parser.add_argument("--traj-bias-rot", type=float, default=2.2, help="CAMDM trajectory rotation blend bias (bias_dir)")
    parser.add_argument("--past-frames", type=int, default=10, help="Number of past trajectory frames to visualize (default: 10)")
    parser.add_argument("--future-frames", type=int, default=45, help="Number of future trajectory frames to visualize (default: 45)")

    # Diffusion/CFG
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Diffusion sampler (ddpm matches CAMDM inference path more closely)")
    parser.add_argument("--cfg-scale", type=float, default=0.5, help="CAMDM CFG weight used during short style-switch/startup burst")
    parser.add_argument("--cfg-count", type=int, default=2, help="CAMDM CFG burst length in regeneration cycles; 0 disables burst scheduling")
    parser.add_argument("--applyframes", type=int, default=15, help="CAMDM applyframes: number of generated frames to apply before next inference (must be <= future_frames)")

    # Inertialization
    parser.add_argument("--inertialize", type=str, default="on", choices=["on", "off"], help="Enable or disable inertialization")
    parser.add_argument("--inertialization-mode", type=str, default="camdm", choices=["camdm", "spring"], help="Inertialization backend")
    parser.add_argument("--blendtime-rotation", type=float, default=0.2, help="CAMDM inertialization blend time for rotations (seconds)")
    parser.add_argument("--blendtime-position", type=float, default=0.2, help="CAMDM inertialization blend time for root position (seconds)")
    parser.add_argument("--spring-halflife-position", type=float, default=0.12, help="Spring inertialization half-life for root position (seconds)")
    parser.add_argument("--spring-halflife-rotation", type=float, default=0.12, help="Spring inertialization half-life for root rotation/joint scalars (seconds)")
    parser.add_argument("--inertial-quat-start", type=int, default=3, help="Start index (inclusive) of quaternion slice in qpos")
    parser.add_argument("--inertial-quat-end", type=int, default=7, help="End index (exclusive) of quaternion slice in qpos")

    # Playback/navigation
    parser.add_argument("--motion", type=int, default=0, help="Starting motion index")
    args = parser.parse_args()
    return args

def print_instruction():
    print("\n" + "=" * 60)
    print("Controls:")
    print("-" * 60)
    print("  SPACE       : Pause/Resume")
    print("  LEFT/RIGHT  : Previous/Next frame (when paused)")
    print("  UP/DOWN     : Previous/Next motion clip")
    print("  R           : Reset to first frame")
    print("  T           : Toggle trajectory visualization")
    print("  1-9         : Set playback speed (1=0.25x, 5=1x, 9=2x)")
    print("  S           : Print status")
    print("  ESC         : Exit")
    print("=" * 60 + "\n")
    

# Keyboard handler
def key_callback(player: DemoPlayer, keycode):
    if keycode == 32:  # SPACE
        player.toggle_pause()
    elif keycode == 265:  # UP
        player.next_motion()
    elif keycode == 264:  # DOWN
        player.prev_motion()
    elif keycode == ord('r') or keycode == ord('R'):
        player.reset()
    elif keycode == ord('t') or keycode == ord('T'):
        player.toggle_trajectory()
    elif keycode == ord('c') or keycode == ord('C'):
        player.toggle_camera_follow()
    elif keycode == ord('s') or keycode == ord('S'):
        player.print_status()
    elif ord('1') <= keycode <= ord('9'):
        # Speed: 1=0.25x, 5=1x, 9=2x
        speed_map = {
            ord('1'): 0.25,
            ord('2'): 0.5,
            ord('3'): 0.75,
            ord('4'): 0.9,
            ord('5'): 1.0,
            ord('6'): 1.25,
            ord('7'): 1.5,
            ord('8'): 1.75,
            ord('9'): 2.0,
        }
        player.set_speed(speed_map[keycode])



def main():
    args = get_args()    
    # Paths
    scene_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "scene.xml"
    )
    dataset_path = f"data/pkls/{args.dataset}.pkl"
    
    print("=" * 60)
    print("Step 2: Visualizing Training Data")
    print("=" * 60)
    
    # Load MuJoCo model
    print(f"\nLoading MuJoCo scene: {scene_path}")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Load motion dataset
    print(f"\nLoading motion dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found: {dataset_path}")
        print("\nAvailable datasets:")
        pkl_dir = "data/pkls"
        for f in os.listdir(pkl_dir):
            if f.endswith('.pkl'):
                print(f"  - {f[:-4]}")
        return
    
    dataset = MotionDataset(dataset_path)
    dataset.print_summary()

    # 1. Setup
    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model & Config
    print(f"Loading {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    # Init Modules
    diffusion = create_gaussian_diffusion(config)
    train_data_joint_num = 30 # 1 + 29
    train_data_per_rot_feat = 6
    train_data_style_set = dataset.styles
    input_feats = (1+29+1) * 6

    diffusion_model = MotionDiffusion(
        input_feats, len(train_data_style_set), train_data_joint_num + 1, 
        train_data_per_rot_feat, config.arch.rot_req, config.arch.clip_len,
        config.arch.latent_dim, config.arch.ff_size, config.arch.num_layers, 
        config.arch.num_heads, arch=config.arch.decoder, 
        cond_mask_prob=config.trainer.cond_mask_prob, device=device
    ).to(device)
    diffusion_model.load_state_dict(checkpoint['state_dict'])
    diffusion_model.eval()

    # Create motion generator
    generator = MotionGenerator(
        diffusion_model, diffusion, config, device,
        sampler=args.sampler, cfg_scale=args.cfg_scale
    )
    
    # Create motion player
    player = DemoPlayer(
        mj_model, mj_data, dataset, generator,
        show_trajectory=True,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        traj_bias_pos=args.traj_bias_pos,
        traj_bias_rot=args.traj_bias_rot,
        cfg_count=args.cfg_count,
        applyframes=args.applyframes,
        inertialize=(args.inertialize == "on"),
        inertialization_mode=args.inertialization_mode,
        blendtime_rotation=args.blendtime_rotation,
        blendtime_position=args.blendtime_position,
        spring_halflife_position=args.spring_halflife_position,
        spring_halflife_rotation=args.spring_halflife_rotation,
        inertial_quat_start=args.inertial_quat_start,
        inertial_quat_end=args.inertial_quat_end,
    )
    
    # Start from specified motion
    if args.motion > 0:
        player.load_motion(args.motion)
        
    import imageio.v2 as imageio


    print_instruction()
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=lambda keycode: key_callback(player, keycode)) as viewer:
        W, H = 640, 320
        OUT = f"videos/demo_{time.strftime('%m%d_%H%M')}.mp4"
        FPS = 30
        writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", pixelformat="yuv420p")
        renderer = mujoco.Renderer(mj_model, height=H, width=W)
        frame_last_time = -np.inf

        viewer.sync()
        try:
            while viewer.is_running():

                player.step()
                viewer.user_scn.ngeom = 0
                player.render_trajectory(viewer.user_scn)
                if player.camera_follow:
                    viewer.cam.lookat[:] = mj_data.qpos[:3]
                viewer.sync()

                # --- Render frame to video---
                if time.time() - frame_last_time > 1/FPS:
                    renderer.update_scene(
                        mj_data, 
                        camera=viewer.cam # Use the viewer's active camera
                    )
                    player.render_trajectory(renderer.scene)
                    frame = renderer.render()
                    writer.append_data(frame)
                    frame_last_time = time.time()
                else:
                    continue
        except KeyboardInterrupt:
            writer.close()
            print("Video saved to", OUT)
        finally:
            print("Exiting viewer...")
if __name__ == "__main__":
    main()
