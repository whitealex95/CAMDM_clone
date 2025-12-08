"""
Motion Data Loader for CAMDM
-----------------------------
Handles loading and converting motion data from pickle files to MuJoCo format.

Data format (from make_pose_data_g1.py):
- root: XYZ position (3) + XYZW quaternion (4) 
- joints: 29 joint angles (1 DOF each)
- Total: 36 DOF matching MuJoCo qpos

The pickle file contains:
{
    "parents": None,
    "offsets": None, 
    "names": None,
    "motions": [
        {
            "filepath": str,
            "local_joint_rotations": (T, 30, 4),  # frame, joint, [angle, 0, 0, 0]
            "global_root_positions": (T, 3),       # frame, xyz
            "traj": list of trajectory translations,
            "traj_pose": list of trajectory poses,
            "style": str,
            "text": str
        },
        ...
    ]
}
"""

import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


class MotionData:
    """Container for a single motion clip."""
    
    def __init__(self, motion_dict):
        """
        Args:
            motion_dict: Dictionary from pickle file containing motion data
        """
        self.filepath = motion_dict["filepath"]
        self.style = motion_dict["style"]
        self.text = motion_dict.get("text", self.style)
        
        # Shape: (T, 30, 4) where joint 0 is root, joints 1-29 are body joints
        # For root: all 4 dims used (quaternion wxyz)
        # For body joints: only dim 0 used (1D rotation angle)
        self.local_joint_rotations = motion_dict["local_joint_rotations"]
        
        # Shape: (T, 3) - root position over time
        self.global_root_positions = motion_dict["global_root_positions"]
        
        # Trajectory data: list of 2 arrays (different smoothing kernels [5, 10])
        # traj: [array(T, 2), array(T, 2)] - XY trajectory positions
        # traj_pose: [array(T, 4), array(T, 4)] - trajectory orientations (wxyz quaternions)
        self.traj = motion_dict.get("traj", None)
        self.traj_pose = motion_dict.get("traj_pose", None)
        
        self.num_frames = len(self.global_root_positions)
        
    def get_qpos(self, frame_idx):
        """
        Extract qpos for a specific frame in MuJoCo format.
        
        Returns:
            qpos: (36,) array [root_xyz(3), root_quat(4), joint_angles(29)]
                  Root quaternion is in WXYZ format for MuJoCo
        """
        if frame_idx >= self.num_frames:
            frame_idx = self.num_frames - 1
        
        # Root position (3,)
        root_pos = self.global_root_positions[frame_idx]
        
        # Root quaternion (4,) - stored as WXYZ
        root_quat_wxyz = self.local_joint_rotations[frame_idx, 0]
        
        # Joint angles (29,) - first dimension of joints 1-29
        joint_angles = self.local_joint_rotations[frame_idx, 1:, 0]
        
        # Concatenate: [xyz(3), wxyz(4), angles(29)] = 36 DOF
        qpos = np.concatenate([
            root_pos,           # 3
            root_quat_wxyz,     # 4
            joint_angles        # 29
        ])
        
        return qpos
    
    def get_all_qpos(self):
        """
        Get qpos for all frames.
        
        Returns:
            qpos_sequence: (T, 36) array
        """
        qpos_seq = []
        for t in range(self.num_frames):
            qpos_seq.append(self.get_qpos(t))
        return np.array(qpos_seq)

    def get_past_qpos(self, frame_idx, past_frames=10):
        """
        Get past qpos sequence up to a specific frame.
        
        Args:
            frame_idx: Current frame index
            past_frames: Number of past frames to include
            
        Returns:
            past_qpos: (past_frames, 36) array
        """
        start_idx = max(0, frame_idx - past_frames)
        past_qpos = []
        for t in range(start_idx, frame_idx):
            past_qpos.append(self.get_qpos(t))
        
        # If not enough past frames, pad with the first frame
        while len(past_qpos) < past_frames:
            past_qpos.insert(0, self.get_qpos(start_idx))
        
        return np.array(past_qpos)
    
    def get_trajectory(self, frame_idx, past_frames=10, future_frames=45, kernel_idx=0, append_z=False):
        """
        Get past and future trajectory for a specific frame.
        
        Args:
            frame_idx: Current frame index
            past_frames: Number of past frames to include
            future_frames: Number of future frames to include
            kernel_idx: Which smoothing kernel to use (0 or 1, corresponding to kernel sizes 5 and 10)
            append_z: Whether to append the Z coordinate from root positions
            
        Returns:
            past_traj: (past_frames, 3) - XYZ positions of past trajectory
            future_traj: (future_frames, 3) - XYZ positions of future trajectory
            past_orient: (past_frames, 4) - WXYZ quaternions of past orientations
            future_orient: (future_frames, 4) - WXYZ quaternions of future orientations
        """
        if self.traj is None or self.traj_pose is None:
            return None, None, None, None
        
        # Get smoothed trajectory (XY only)
        traj_xy = self.traj[kernel_idx]  # (T, 2)
        traj_quat = self.traj_pose[kernel_idx]  # (T, 4) wxyz
        
        # Extract past trajectory
        past_start = max(0, frame_idx - past_frames)
        past_end = frame_idx
        past_xy = traj_xy[past_start:past_end]
        past_quat = traj_quat[past_start:past_end]
        
        # Extract future trajectory
        future_start = frame_idx
        future_end = min(self.num_frames, frame_idx + future_frames)
        future_xy = traj_xy[future_start:future_end]
        future_quat = traj_quat[future_start:future_end]
        
        # Add Z coordinate from actual root positions if requested
        past_z = self.global_root_positions[past_start:past_end, 2:3]
        future_z = self.global_root_positions[future_start:future_end, 2:3]
        if not append_z:
            past_z = np.zeros((past_xy.shape[0], 1))
            future_z = np.zeros((future_xy.shape[0], 1))
        
        past_traj = np.concatenate([past_xy, past_z], axis=-1)
        future_traj = np.concatenate([future_xy, future_z], axis=-1)
        
        return past_traj, future_traj, past_quat, future_quat
    
    def __repr__(self):
        return (f"MotionData(style='{self.style}(style_idx={self.style_idx})', "
                f"frames={self.num_frames}, "
                f"file='{os.path.basename(self.filepath)}')")


class MotionDataset:
    """Dataset loader for motion pickle files."""
    
    def __init__(self, pkl_path):
        """
        Load motion dataset from pickle file.
        
        Args:
            pkl_path: Path to .pkl file (e.g., 'data/pkls/lafan1_g1.pkl')
        """
        self.pkl_path = pkl_path
        
        print(f"Loading motion data from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
                
        # Parse all motion clips
        self.motions = []
        for motion_dict in data_dict["motions"]:
            self.motions.append(MotionData(motion_dict))
        
        # Build style index
        self.styles = sorted(list(set(m.style for m in self.motions)))
        self.style_to_motions = {style: [] for style in self.styles}
        for i, motion in enumerate(self.motions):
            self.style_to_motions[motion.style].append(i)
            motion.style_idx = self.styles.index(motion.style)
        
        print(f"✓ Loaded {len(self.motions)} motion clips")
        print(f"✓ Found {len(self.styles)} unique styles: {self.styles}")
        
    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        return self.motions[idx]
    
    def get_motion_by_style(self, style):
        """Get all motions with a specific style."""
        indices = self.style_to_motions.get(style, [])
        return [self.motions[i] for i in indices]
    
    def print_summary(self):
        """Print dataset summary."""
        print("\n" + "=" * 60)
        print(f"{'Motion Dataset Summary':^60}")
        print("=" * 60)
        print(f"Source file: {self.pkl_path}")
        print(f"Total clips: {len(self.motions)}")
        print(f"\n{'Style':<20} {'Clips':<10} {'Avg Frames':<15}")
        print("-" * 60)
        
        for style in self.styles:
            clips = self.get_motion_by_style(style)
            avg_frames = np.mean([c.num_frames for c in clips])
            print(f"{style:<20} {len(clips):<10} {avg_frames:<15.1f}")
        
        print("=" * 60 + "\n")


def test_loader():
    """Test the motion loader."""
    import sys
    
    # Try to load lafan1_g1 dataset
    pkl_path = "data/pkls/lafan1_g1.pkl"
    
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}")
        print("Available pkl files:")
        pkl_dir = "data/pkls"
        if os.path.exists(pkl_dir):
            for f in os.listdir(pkl_dir):
                if f.endswith('.pkl'):
                    print(f"  - {os.path.join(pkl_dir, f)}")
        return
    
    # Load dataset
    dataset = MotionDataset(pkl_path)
    dataset.print_summary()
    
    # Test loading a single motion
    print("Testing motion data extraction...")
    motion = dataset[0]
    print(f"\nFirst motion: {motion}")
    
    # Get qpos for first frame
    qpos = motion.get_qpos(0)
    print(f"\nFirst frame qpos shape: {qpos.shape}")
    print(f"Root position: {qpos[:3]}")
    print(f"Root quaternion (WXYZ): {qpos[3:7]}")
    print(f"First 5 joint angles: {qpos[7:12]}")
    
    # Verify qpos dimensions
    assert qpos.shape == (36,), f"Expected qpos shape (36,), got {qpos.shape}"
    print("\n✓ Motion loader test passed!")


if __name__ == "__main__":
    test_loader()
