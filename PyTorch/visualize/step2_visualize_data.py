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

from visualize.motion_loader import MotionDataset
from visualize.utils.geometry import draw_trajectory

class MotionPlayer:
    """Interactive motion player for MuJoCo viewer."""
    
    def __init__(self, model, data, dataset, show_trajectory=True, past_frames=10, future_frames=45):
        self.model = model
        self.data = data
        self.dataset = dataset
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
        
        # Load first motion
        self.load_motion(0)

    def load_motion(self, motion_idx):
        """Load a specific motion clip."""
        self.current_motion_idx = motion_idx % len(self.dataset)
        self.current_motion = self.dataset[self.current_motion_idx]
        self.current_frame = 0
        
        print(f"\n{'='*60}")
        print(f"Motion {self.current_motion_idx + 1}/{len(self.dataset)}")
        print(f"Style: {self.current_motion.style}")
        print(f"Frames: {self.current_motion.num_frames}")
        print(f"Duration: {self.current_motion.num_frames / self.fps:.2f}s")
        print(f"{'='*60}\n")
        
        self.update_pose()
    
    def update_pose(self):
        """Update MuJoCo model with current frame pose."""
        qpos = self.current_motion.get_qpos(self.current_frame)
        
        # Set the pose
        self.data.qpos[:] = qpos
        
        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)
        
        # Update trajectory visualization if enabled
        if self.show_trajectory:
            self._update_trajectory()
    
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
            if self.current_frame >= self.current_motion.num_frames:
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
        self.current_frame = (self.current_frame + 1) % self.current_motion.num_frames
        self.update_pose()
    
    def prev_frame(self):
        """Go back one frame."""
        self.current_frame = (self.current_frame - 1) % self.current_motion.num_frames
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
    
    def _update_trajectory(self):
        """Update trajectory visualization markers."""
        # Get trajectory data for current frame
        past_traj, future_traj, past_orient, future_orient = \
            self.current_motion.get_trajectory(
                self.current_frame, 
                self.past_frames, 
                self.future_frames,
                kernel_idx=0  # Use first smoothing kernel
            )
        
        # Store for rendering (will be used by custom rendering)
        self.past_traj = past_traj
        self.future_traj = future_traj
        self.past_orient = past_orient
        self.future_orient = future_orient
    
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
            if len(self.future_traj) > 0:
                draw_trajectory(viewer, self.future_traj, self.future_orient, color=[1.0, 0.2, 0.2, 1.0])

    def toggle_trajectory(self):
        """Toggle trajectory visualization."""
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")
    
    def print_status(self):
        """Print current status."""
        status = (
            f"Motion: {self.current_motion_idx + 1}/{len(self.dataset)} | "
            f"Frame: {self.current_frame}/{self.current_motion.num_frames} | "
            f"Style: {self.current_motion.style} | "
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
def key_callback(player: MotionPlayer, keycode):
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
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
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
    
    # Create motion player
    player = MotionPlayer(
        model, data, dataset,
        show_trajectory=True,
        past_frames=args.past_frames,
        future_frames=args.future_frames
    )
    
    # Start from specified motion
    if args.motion > 0:
        player.load_motion(args.motion)
    
    print_instruction()
    with mujoco.viewer.launch_passive(model, data, key_callback=lambda keycode: key_callback(player, keycode)) as viewer:
        viewer.sync()
        while viewer.is_running():
            player.step()
            # with viewer.lock():
            if True:
                viewer.user_scn.ngeom = 0
                player.render_trajectory(viewer)
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
            time.sleep(0.001)
if __name__ == "__main__":
    main()
