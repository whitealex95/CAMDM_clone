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
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models import MotionDiffusion
from diffusion.create_diffusion import create_gaussian_diffusion

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import draw_trajectory
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic

import torch
from visualize.utils.rotations import rot_from_wxyz


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
    
    def __init__(self, model, diffusion, config, device="cuda"):
        self.model = ModelWrapper(model)
        self.diffusion = diffusion
        self.config = config
        self.device = device
        
        # Model parameters
        self.past_frames = config.arch.past_frame
        self.future_frames = config.arch.future_frame
        self.joint_num = 30 # 1 + 29
        self.rot_req = config.arch.rot_req
        self.per_rot_feat = 6 # 6D rotation representation
            
    def generate_motion(self, past_qpos, traj_trans, traj_pose, style_idx):
        """
        Generate future motions given past motion and trajectory conditions.
        past_qpos: (past_frames, 36=7+29) numpy array
        traj_trans: (future_frames, 2) numpy array, XY in global frame
        traj_pose: (future_frames, 4) numpy array, orientation in global frame (wxyz)
        style_idx: int, style index
        Returns:
            generated_qpos: (future_frames, 36=7+29) numpy array
        """
        # Obtain current root position and orientation for centering
        curr_root_XY = past_qpos[-1, :2].copy()  # (2,)
        curr_root_quat = past_qpos[-1, 3:7]

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
        
        shape = (1, self.joint_num + 1, self.per_rot_feat, self.future_frames)
        sample = self.diffusion.ddim_sample_loop(
            self.model, shape, clip_denoised=False, model_kwargs=model_kwargs,
            progress=False, eta=0.0, device=self.device
        )

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
                 show_trajectory=True, past_frames=10, future_frames=45, blend=0.5):
        self.model = model
        self.data = data
        self.dataset = dataset
        self.motion_generator = motion_generator
        self.show_trajectory = show_trajectory
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
        self.apply_generated_frames = 15 # frames to apply per generation step (default: 15, meaning 2Hz generation)
        self.generated_frame_idx = 0

        # Create a queue to store qpos history
        from collections import deque 
        self.qpos_history = deque(maxlen=self.past_frames)
                

        # Generated future poses
        self.generated_qpos = None
        self.generated_future_traj = None
        self.generated_future_orient = None
        # Blending factor (1-t^blend) predicted + t^blend target, t ∈ [0, 1]
        self.blend = 0.5 # (0: pure target, not using generated prediction)
        
        # Load first motion
        self.load_motion(0)
        
        

    def load_motion(self, motion_idx):
        """Load a specific motion clip."""
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion_data = self.dataset[self.current_motion_idx]
        self.current_frame = 0
        
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

    def load_future_qpos(self):
        future_qpos_dataset = self.current_motion_data.get_future_qpos(self.current_frame)
        return future_qpos_dataset

    def generate_motion(self):
        """Generate future poses using the motion generator."""
        past_qpos_dataset = self.load_past_qpos()
        if self.current_frame > 0:
            past_qpos = np.array(self.qpos_history)  # (past_frames, 36)
        else:
            past_qpos = past_qpos_dataset
        style_idx = self.current_motion_data.style_idx

        # generated_qpos = self.motion_generator.generate_motion(
        #     past_qpos, self.future_traj_dataset[:, :2], self.future_orient_dataset, style_idx)
        self.raw_generated_qpos = self.motion_generator.generate_motion(past_qpos, self.future_traj, self.future_orient, style_idx)
        generated_qpos = self.raw_generated_qpos.copy()
        if True: # override with future frames
            future_qpos_dataset = self.load_future_qpos()
            # TODO: figure out why this is required. Shouldn't this be easily learned by the model?
            # generated_qpos[:, :2] = self.future_traj[:, :2]
        return generated_qpos # (future_frames, 36)

    def update_pose(self):
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
    
        
    def load_future_trajectory(self):
        # Load future trajectory from dataset
        _, future_traj_dataset, _, future_orient_dataset = self._load_trajectory_from_dataset()
        self.future_traj_dataset = future_traj_dataset
        self.future_orient_dataset = future_orient_dataset


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
                                                                          self.future_traj_dataset, self.future_orient_dataset, self.blend)
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
    
    def next_frame(self):
        """Advance one frame."""
        self.current_frame = (self.current_frame + 1) % self.current_motion_data.num_frames
        self.update_pose()
    
    def prev_frame(self):
        """Go back one frame."""
        self.current_frame = (self.current_frame - 1) % self.current_motion_data.num_frames
        self.update_pose()
    
    def toggle_pause(self):
        """Toggle play/pause."""
        self.playing = not self.playing
        print(f"{'Playing' if self.playing else 'Paused'}")
    
    def reset(self):
        """Reset to first frame."""
        self.current_frame = 0
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
    
    def render_trajectory(self, viewer: mujoco.viewer.Handle):
        if not self.show_trajectory:
            return
        assert hasattr(self, 'past_traj') and hasattr(self, 'future_traj'), \
            "Trajectory data not available for visualization."
        if True:
            viewer.user_scn.ngeom = 0
            # --- DRAW PAST (Blue) ---
            if len(self.past_traj) > 0:
                draw_trajectory(viewer, self.past_traj, self.past_orient, color=[0.2, 0.5, 1.0, 1.0])
            # --- DRAW FUTURE (Red) ---
            if len(self.future_traj_dataset) > 0:
                draw_trajectory(viewer, self.future_traj_dataset, self.future_orient_dataset, color=[1.0, 0.2, 0.2, 1.0])
                draw_trajectory(viewer, self.future_traj, self.future_orient, color=[0.2, 1.0, 0.2, 1.0])
                if self.generated_future_traj is not None:
                    draw_trajectory(viewer, self.generated_future_traj, self.generated_future_orient, color=[0.2, 0.2, 0.2, 0.5])

    def toggle_trajectory(self):
        """Toggle trajectory visualization."""
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")
    
    def print_status(self):
        """Print current status."""
        status = (
            f"Motion: {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"Frame: {self.current_frame}/{self.current_motion_data.num_frames} | "
            f"Style: {self.current_motion_data.style} | "
            f"{'Playing' if self.playing else 'Paused'} "
            f"({self.playback_speed}x) | "
            f"Traj: {'ON' if self.show_trajectory else 'OFF'}"
        )
        print(status)

def get_args():
    parser = argparse.ArgumentParser(description="Visualize training motion data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lafan1_g1",
        help="Dataset name (lafan1_g1 or 100style)"
    )
    parser.add_argument("--checkpoint", type=str, default="save/camdm_g1_lafan1_g1/best.pt")
    parser.add_argument("--blend", type=float, default=0.3, help="0.0 = Pure AI, 1.0 = Pure GT Trajectory")    

    parser.add_argument(
        "--motion",
        type=int,
        default=0,
        help="Starting motion index"
    )
    parser.add_argument(
        "--past-frames",
        type=int,
        default=10,
        help="Number of past trajectory frames to visualize (default: 10)"
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=45,
        help="Number of future trajectory frames to visualize (default: 45)"
    )
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
    elif keycode == 263:  # LEFT
        player.prev_frame()
    elif keycode == 262:  # RIGHT
        player.next_frame()
    elif keycode == ord('r') or keycode == ord('R'):
        player.reset()
    elif keycode == ord('t') or keycode == ord('T'):
        player.toggle_trajectory()
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
    generator = MotionGenerator(diffusion_model, diffusion, config, device)   
    
    # Create motion player
    player = DemoPlayer(
        mj_model, mj_data, dataset, generator,
        show_trajectory=True,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        blend=args.blend
    )
    
    # Start from specified motion
    if args.motion > 0:
        player.load_motion(args.motion)
    
    print_instruction()
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=lambda keycode: key_callback(player, keycode)) as viewer:
        viewer.sync()
        while viewer.is_running():
            player.step()
            # with viewer.lock():
            if True:
                viewer.user_scn.ngeom = 0
                player.render_trajectory(viewer)
                # viewer.cam.lookat[:] = mj_data.qpos[:3]
                viewer.sync()
            time.sleep(0.001)
if __name__ == "__main__":
    main()
