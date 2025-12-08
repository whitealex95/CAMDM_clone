"""
Step 3: Visualize Generated Motion (Autoregressive + Dataset Guidance)
---------------------------------------------------
Features:
1. Loads initial pose and style from a real dataset sequence.
2. Generates motion autoregressively (output of T becomes input of T+1).
3. Blends the generated root trajectory with the Ground Truth dataset trajectory
   to keep the robot on track while synthesizing new joint details.

Controls:
- SPACE: Pause/Resume
- LEFT/RIGHT: Previous/Next frame
- G: Generate new motion sequence (picks a new random dataset sequence)
- V: Toggle trajectory visualization (Blue=Generated, Red=Target/GT)
- 1-9: Change playback speed
- ESC: Exit
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models import MotionDiffusion
from network.dataset_g1 import HumanoidMotionDataset
from diffusion.create_diffusion import create_gaussian_diffusion
from visualize.utils.geometry import draw_trajectory, init_geom

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

def get_rotation_matrix(quat):
    """Convert Quaternion [w, x, y, z] to 3x3 Rotation Matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

class MotionGenerator:
    """Autoregressive motion generator with Dataset Guidance."""
    
    def __init__(self, model, diffusion, config, train_dataset, device='cuda'):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.train_dataset = train_dataset
        self.device = device
        
        # Model parameters
        self.past_frames = config.arch.past_frame
        self.future_frames = config.arch.future_frame
        self.joint_num = train_dataset.joint_num
        self.per_rot_feat = train_dataset.per_rot_feat
        self.rot_req = config.arch.rot_req
        
        # State
        self.generated_motion = None
        self.trajectory_data = []
        self.current_style_idx = 0
        self.current_style = "Unknown"
        
        # Current Sequence Info
        self.current_seq_start_idx = 0
        
        print(f"\n{'='*60}")
        print("Motion Generator Initialized")
        print(f"Window: Past {self.past_frames} / Future {self.future_frames}")
        print(f"{'='*60}\n")

    def get_full_qpos_from_sample(self, dataset_idx):
        """Extracts qpos (36,) from a dataset sample index."""
        sample = self.train_dataset[dataset_idx]
        
        # The dataset returns windows. We need to reconstruct the qpos from the 'conditions' (past)
        # and 'data' (future) to get a full window qpos.
        
        # 1. Parse Past Motion
        past_data = sample['conditions']['past_motion'].numpy() # (past, 31, feat)
        
        # 2. Parse Future Motion (Ground Truth from dataset)
        future_data = sample['data'].numpy() # (future, 31, feat)
        # Combine
        full_window_data = np.concatenate([past_data, future_data], axis=0)
        
        qpos_list = []
        for t in range(len(full_window_data)):
            root_repr = full_window_data[t, 0]
            joint_repr = full_window_data[t, 1:30]
            root_pos = full_window_data[t, 30, :3]
            
            if self.rot_req == '6d':
                root_quat = nn_transforms.repr6d2quat(torch.from_numpy(root_repr).float().unsqueeze(0)).numpy()[0]
            else:
                root_quat = root_repr[:4]
            
            joint_angles = joint_repr[:, 0]
            qpos = np.concatenate([root_pos, root_quat, joint_angles])
            qpos_list.append(qpos)
            
        return np.array(qpos_list), sample['conditions']['style_idx']

    def get_aligned_target_trajectory(self, current_root_pos, current_root_quat, dataset_idx):
        """
        Fetches the GT trajectory from the dataset starting at dataset_idx.
        Aligns this GT trajectory to start at the robot's current position/heading.
        """
        # Get the GT motion for the next window
        # Note: In a sliding window dataset, index i+1 is usually 1 frame later.
        # We need the window that represents the future relative to our current progress.
        
        # Extract GT from the dataset sample
        sample = self.train_dataset[dataset_idx]
        
        # Extract GT trajectory (XY pos and Orientation)
        # sample['traj_trans'] is (future, 2)
        # sample['traj_pose'] is (future, 4) or 6d
        gt_trans_local = sample['conditions']['traj_trans'].numpy() 
        gt_pose_local = sample['conditions']['traj_pose'].numpy()
        
        # The dataset trajectory is usually relative to the frame at t=0 of the window.
        # We need to transform this relative path to the World frame based on where the robot CURRENTLY is.
        
        # Current Robot Transform
        curr_rot = R.from_quat([current_root_quat[1], current_root_quat[2], current_root_quat[3], current_root_quat[0]]) # Scalar last for Scipy
        curr_pos = current_root_pos
        
        # 1. Convert GT Local -> GT World Delta -> Applied to Current
        # Assuming dataset traj is relative offsets:
        # We need to rotate the GT offsets by the current robot heading and add to current robot pos.
        
        aligned_trans = []
        aligned_pose = []
        
        # If the dataset stores absolute coords relative to window start:
        # We assume the model expects the trajectory condition to be relative to the *start* of the generation window.
        # So we actually don't need to do complex world alignment for the *Input Condition* if the model is local.
        # However, for the *Blending* (Target) step, we need it in World Frame.
        
        # Let's reconstruct the Absolute World Trajectory of the GT target aligned to current robot
        for t in range(len(gt_trans_local)):
            # Local offset from start of window
            rel_pos_2d = gt_trans_local[t] # x, y
            rel_pos_3d = np.array([rel_pos_2d[0], rel_pos_2d[1], 0.0])
            
            # Rotate offset by current robot heading
            world_offset = curr_rot.apply(rel_pos_3d)
            target_pos = curr_pos + world_offset
            aligned_trans.append(target_pos[:2]) # Keep 2D for model input
            
            # Orientation
            # Convert GT pose (usually 6d or quat) to rotation
            if gt_pose_local.shape[-1] == 6:
                gt_rot_mat = nn_transforms.rot6d_to_rotmat(torch.from_numpy(gt_pose_local[t]).unsqueeze(0)).numpy()[0]
                gt_rel_rot = R.from_matrix(gt_rot_mat)
            else:
                # Assuming Quat
                # GT usually stores absolute orientation relative to world? Or relative to root?
                # Standard practice: Relative to root at t=0
                gt_rel_rot = R.from_quat(gt_pose_local[t]) # Check wxyz vs xyzw order for dataset
                
            # Compose: New Target Rot = Current Rot * GT Relative Rot
            # Note: This depends heavily on dataset encoding. Assuming standard relative.
            target_rot = curr_rot * gt_rel_rot
            target_quat = target_rot.as_quat() # xyzw
            # Convert back to WXYZ for internal use
            aligned_pose.append(np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]]))
            
        return np.array(aligned_trans), np.array(aligned_pose)

    def blend_trajectories(self, gen_qpos, target_trans, target_pose, alpha=0.5):
        """
        Blends the Root of the Generated motion with the Aligned Target trajectory.
        alpha: 0.0 = Use Generated, 1.0 = Use Target
        """
        blended_qpos = gen_qpos.copy()
        T = len(gen_qpos)
        
        for t in range(T):
            # 1. Position Blend (XY only, preserve Z from generation (physics))
            gen_pos = gen_qpos[t, :3]
            tgt_pos_2d = target_trans[t]
            
            # Linear Interpolate XY
            new_x = (1 - alpha) * gen_pos[0] + alpha * tgt_pos_2d[0]
            new_y = (1 - alpha) * gen_pos[1] + alpha * tgt_pos_2d[1]
            blended_qpos[t, 0] = new_x
            blended_qpos[t, 1] = new_y
            
            # 2. Rotation Blend (Slerp)
            gen_quat = gen_qpos[t, 3:7] # wxyz
            tgt_quat = target_pose[t]   # wxyz
            
            # Scipy uses xyzw
            r_gen = R.from_quat([gen_quat[1], gen_quat[2], gen_quat[3], gen_quat[0]])
            r_tgt = R.from_quat([tgt_quat[1], tgt_quat[2], tgt_quat[3], tgt_quat[0]])
            
            # Slerp
            slerp = Slerp([0, 1], R.from_matrix(np.stack([r_gen.as_matrix(), r_tgt.as_matrix()])))
            r_blended = slerp([alpha])[0]
            
            # Back to wxyz
            b_q = r_blended.as_quat()
            blended_qpos[t, 3] = b_q[3] # w
            blended_qpos[t, 4] = b_q[0] # x
            blended_qpos[t, 5] = b_q[1] # y
            blended_qpos[t, 6] = b_q[2] # z
            
        return blended_qpos

    def generate_motion(self, num_segments=4, trajectory_blend=0.5):
        """
        Generates motion autoregressively.
        1. Pick a random start from dataset.
        2. Loop segments.
        3. Feed output of T to input of T+1.
        4. Blend output root with GT dataset trajectory.
        """
        # 1. Pick Random Sequence Start
        # Ensure we have enough room in the dataset
        max_idx = len(self.train_dataset) - (num_segments * self.future_frames)
        start_idx = np.random.randint(0, max_idx)
        self.current_seq_start_idx = start_idx
        
        # Get Initial Data (Pose + Style)
        # We need the PAST context for the very first frame.
        # We take the first 'past_frames' from this window.
        initial_full_qpos, style_idx_tensor = self.get_full_qpos_from_sample(start_idx)
        
        self.current_style_idx = int(style_idx_tensor)
        self.current_style = self.train_dataset.style_set[self.current_style_idx]
        
        # Initial Past Motion (Context)
        past_qpos = initial_full_qpos[:self.past_frames] 
        
        print(f"\n{'='*60}")
        print(f"Generating Sequence from Dataset Index {start_idx}")
        print(f"Style: {self.current_style}")
        print(f"Blending Factor: {trajectory_blend} (0=Pure Gen, 1=Pure GT)")
        print(f"{'='*60}")
        
        # Storage
        all_generated_qpos = list(past_qpos)
        self.trajectory_data = [] # Store for viz
        
        # Autoregressive Loop
        curr_dataset_idx = start_idx
        
        with torch.no_grad():
            for seg_idx in range(num_segments):
                print(f"Segment {seg_idx+1}/{num_segments} | Dataset Idx: {curr_dataset_idx}")
                
                # A. Prepare Input Conditions from Previous Output
                past_motion = self.qpos_to_model_format(past_qpos)
                past_motion_tensor = torch.from_numpy(past_motion).float().unsqueeze(0).to(self.device)
                past_motion_tensor = past_motion_tensor.permute(0, 2, 3, 1) # (1, 31, feat, past)
                
                # B. Prepare Target Trajectory (Conditions)
                # We want the trajectory relative to the robot's current position (last frame of past_qpos)
                curr_root_pos = past_qpos[-1, :3]
                curr_root_quat = past_qpos[-1, 3:7]
                
                # Fetch GT future trajectory aligned to current robot
                traj_trans_aligned, traj_pose_aligned = self.get_aligned_target_trajectory(
                    curr_root_pos, curr_root_quat, curr_dataset_idx
                )
                
                # Convert to Model Tensor
                traj_pose_repr = nn_transforms.get_rotation(
                    torch.from_numpy(traj_pose_aligned).float(), self.rot_req
                ).numpy()
                
                traj_trans_tensor = torch.from_numpy(traj_trans_aligned).float().unsqueeze(0).to(self.device).permute(0, 2, 1)
                traj_pose_tensor = torch.from_numpy(traj_pose_repr).float().unsqueeze(0).to(self.device).permute(0, 2, 1)
                style_idx = torch.tensor([float(self.current_style_idx)]).to(self.device)
                
                # C. Generate
                model_kwargs = {
                    'past_motion': past_motion_tensor, # (1, 31, feat, past)
                    'traj_pose': traj_pose_tensor, # (1, feat, future)
                    'traj_trans': traj_trans_tensor, # (1, 2, future)
                    'style_idx': style_idx, # (1)
                    'y': {}
                }
                
                shape = (1, self.joint_num + 1, self.per_rot_feat, self.future_frames)
                wrapped_model = ModelWrapper(self.model)
                
                sample = self.diffusion.ddim_sample_loop(
                    wrapped_model, shape, clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, eta=0.0, device=self.device
                )
                
                # Process Output
                sample = sample.squeeze(0).permute(2, 0, 1).cpu().numpy() # (future, 31, feat)
                generated_qpos = self.model_format_to_qpos(sample)
                
                # D. BLEND OUTPUT
                # Blend the generated root motion with the aligned GT trajectory
                # This keeps the robot on the specific path while generating joint animations
                if trajectory_blend > 0:
                    generated_qpos = self.blend_trajectories(
                        generated_qpos, traj_trans_aligned, traj_pose_aligned, alpha=trajectory_blend
                    )
                
                # E. Store & Update State
                all_generated_qpos.extend(generated_qpos)
                
                # Save trajectories for visualization
                self.trajectory_data.append({
                    'start_frame': len(all_generated_qpos) - len(generated_qpos),
                    'gen_trans': generated_qpos[:, :3], # The result (Blue)
                    'gt_trans': traj_trans_aligned,     # The target (Red)
                    'gt_pose': traj_pose_aligned
                })
                
                # Update for next loop
                past_qpos = np.array(all_generated_qpos[-self.past_frames:])
                
                # Advance Dataset Index (Usually stride is 1, so we jump by future_frames)
                # However, HumanoidMotionDataset might have overlaps. Assuming continuous indexing maps to continuous frames (mostly true for standard loaders)
                curr_dataset_idx += self.future_frames
        
        self.generated_motion = np.array(all_generated_qpos)
        print("Generation Complete.")

    def qpos_to_model_format(self, qpos_seq):
        """Convert qpos sequence to model input format."""
        T = len(qpos_seq)
        motion = np.zeros((T, self.joint_num + 1, self.per_rot_feat))
        for t in range(T):
            qpos = qpos_seq[t]
            root_pos = qpos[:3]
            root_quat = qpos[3:7] # WXYZ
            
            root_repr = nn_transforms.get_rotation(
                torch.from_numpy(root_quat).float().unsqueeze(0), self.rot_req
            ).numpy()[0]
            
            joint_angles = qpos[7:]
            joint_repr = np.zeros((29, self.per_rot_feat))
            joint_repr[:, 0] = joint_angles
            
            motion[t, 0] = root_repr
            motion[t, 1:30] = joint_repr
            motion[t, 30, :3] = root_pos
        return motion
    
    def model_format_to_qpos(self, motion):
        """Convert model output back to qpos."""
        T = motion.shape[0]
        qpos_seq = []
        for t in range(T):
            root_repr = motion[t, 0]
            if self.rot_req == '6d':
                root_quat = nn_transforms.repr6d2quat(torch.from_numpy(root_repr).float().unsqueeze(0)).numpy()[0]
            elif self.rot_req == 'q':
                root_quat = root_repr
            
            joint_angles = motion[t, 1:30, 0]
            root_pos = motion[t, 30, :3]
            qpos = np.concatenate([root_pos, root_quat, joint_angles])
            qpos_seq.append(qpos)
        return np.array(qpos_seq)
    
    def get_viz_paths(self, frame_idx):
        """Returns points for Blue (Generated) and Red (GT) lines for current segment."""
        for seg in self.trajectory_data:
            s = seg['start_frame']
            e = s + len(seg['gen_trans'])
            if s <= frame_idx < e:
                # Found the segment
                return seg['gen_trans'], seg['gt_trans']
        return None, None

class GeneratedMotionPlayer:
    """Interactive player."""
    def __init__(self, model, data, generator):
        self.model = model
        self.data = data
        self.generator = generator
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.show_traj = True
        self.last_update = time.time()
        
    def step(self):
        if not self.playing or self.generator.generated_motion is None: return
        
        now = time.time()
        if now - self.last_update > (1/30) / self.playback_speed:
            self.current_frame = (self.current_frame + 1) % len(self.generator.generated_motion)
            self.update_pose()
            self.last_update = now

    def update_pose(self):
        qpos = self.generator.generated_motion[self.current_frame]
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="save/camdm_g1_lafan1_g1/best.pt")
    parser.add_argument("--blend", type=float, default=0.3, help="0.0 = Pure AI, 1.0 = Pure GT Trajectory")
    args = parser.parse_args()
    
    # 1. Setup
    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model & Config
    print(f"Loading {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # 3. Load Dataset
    print(f"Loading Dataset {config.data}...")
    train_data = HumanoidMotionDataset(
        config.data, config.arch.rot_req, config.arch.offset_frame,  
        config.arch.past_frame, config.arch.future_frame, 
        dtype=common.select_platform(32), limited_num=-1
    )
    
    # 4. Init Modules
    diffusion = create_gaussian_diffusion(config)
    input_feats = (train_data.joint_num + 1) * train_data.per_rot_feat
    model = MotionDiffusion(
        input_feats, len(train_data.style_set), train_data.joint_num + 1, 
        train_data.per_rot_feat, config.arch.rot_req, config.arch.clip_len,
        config.arch.latent_dim, config.arch.ff_size, config.arch.num_layers, 
        config.arch.num_heads, arch=config.arch.decoder, 
        cond_mask_prob=config.trainer.cond_mask_prob, device=device
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 5. Mujoco
    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)
    
    # 6. Generator & Player
    generator = MotionGenerator(model, diffusion, config, train_data, device)
    player = GeneratedMotionPlayer(mj_model, mj_data, generator)
    
    print("\nInitial Generation...")
    generator.generate_motion(num_segments=4, trajectory_blend=args.blend)
    player.update_pose()
    
    # 7. Viz Loop
    def key_callback(keycode):
        if keycode == 32: player.playing = not player.playing # Space
        elif keycode == ord('g'): 
            generator.generate_motion(num_segments=4, trajectory_blend=args.blend)
            player.current_frame = 0
        elif keycode == ord('v'): player.show_traj = not player.show_traj
        elif ord('1') <= keycode <= ord('9'): player.playback_speed = (keycode - ord('0')) * 0.25

    print("\nrunning viewer...")
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            player.step()
            viewer.cam.lookat[:] = mj_data.qpos[:3]
            
            if player.show_traj:
                gen_path, gt_path = generator.get_viz_paths(player.current_frame)
                viewer.user_scn.ngeom = 0
                if gen_path is not None:
                    # Blue = Actual Generated Path
                    # Red = The Target Path (GT) we tried to blend toward
                    for i in range(len(gen_path)-1):
                        p1, p2 = gen_path[i], gen_path[i+1]
                        init_geom(viewer.user_scn.geoms[viewer.user_scn.ngeom], color=[0.2, 0.2, 1.0, 1])
                        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 0.02, 
                                             [p1[0], p1[1], 0.05], [p2[0], p2[1], 0.05])
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0.2, 0.2, 1.0, 1]
                        viewer.user_scn.ngeom += 1
                        
                        g1, g2 = gt_path[i], gt_path[i+1] # 2D
                        init_geom(viewer.user_scn.geoms[viewer.user_scn.ngeom], color=[1.0, 0.2, 0.2, 1])
                        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 0.02, 
                                             [g1[0], g1[1], 0.05], [g2[0], g2[1], 0.05])
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1.0, 0.2, 0.2, 1]
                        viewer.user_scn.ngeom += 1
                        
            viewer.sync()
            time.sleep(0.002)

if __name__ == "__main__":
    main()