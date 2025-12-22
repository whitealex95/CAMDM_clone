"""
Interactive Trajectory Blending Visualizer
-------------------------------------------
Visualize and interactively modify trajectory blending parameters in real-time.

This tool allows you to:
- Adjust blend coefficient with keyboard controls
- Modify target trajectory parameters (position and rotation)
- Modify generated trajectory parameters
- See all three trajectories (generated, target, blended) in real-time
- Visualize the humanoid following the blended trajectory

Controls:
- B/N          : Decrease/Increase blend coefficient (step: 0.05)
- SHIFT+B/N    : Decrease/Increase blend coefficient (step: 0.01)
- Q/W          : Decrease/Increase target X endpoint
- A/S          : Decrease/Increase target Y endpoint
- Z/X          : Decrease/Increase target rotation angle
- E/R          : Decrease/Increase gen X endpoint
- D/F          : Decrease/Increase gen Y endpoint
- C/V          : Decrease/Increase gen rotation angle
- SPACE        : Pause/Resume animation
- LEFT/RIGHT   : Previous/Next frame (when paused)
- T            : Toggle trajectory visualization
- P            : Print current parameters
- 1-9          : Set playback speed
- ESC          : Exit

Usage:
    python visualize/interactive_trajectory_blend.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Add visualize_dir and pytorch project dir to path
VISUALIZE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTORCH_DIR = os.path.dirname((VISUALIZE_DIR))
sys.path.append(VISUALIZE_DIR)
sys.path.append(PYTORCH_DIR)

from visualize.utils.geometry import draw_trajectory
from visualize.utils.trajectory import blend_trajectories
from visualize.utils.geometry import draw_label


class InteractiveTrajectoryBlender:
    """Interactive trajectory blending demonstration."""
    
    def __init__(self, model, data, T=45, n_joints=29, fps=30):
        self.model = model
        self.data = data
        self.T = T
        self.n_joints = n_joints
        self.fps = fps
        self.frame_dt = 1.0 / self.fps
        
        # Playback state
        self.current_frame = 0
        self.playing = True
        self.playback_speed = 1.0
        self.last_update_time = time.time()
        self.show_trajectory = True
        
        # Trajectory parameters
        self.blend = 0.5
        self.gen_x_end = 10.0
        self.gen_y_end = 0.0
        self.gen_rot_end = 0.0  # radians
        self.target_x_end = 0.0
        self.target_y_end = 10.0
        self.target_rot_end = np.pi / 2  # 90 degrees
        
        # Generate initial trajectories
        self.update_trajectories()
        
        # Set initial pose
        self.update_pose()
        
    def generate_gen_qpos(self):
        """Generate the 'generated' trajectory based on parameters."""
        gen_qpos = np.zeros((self.T, 7 + self.n_joints))
        
        for t in range(self.T):
            progress = t / (self.T - 1)
            # Linear motion
            gen_qpos[t, 0] = self.gen_x_end * progress
            gen_qpos[t, 1] = self.gen_y_end * progress
            gen_qpos[t, 2] = 0.95  # Standing height
            
            # Rotation
            angle = self.gen_rot_end * progress
            half_sin = np.sin(angle / 2)
            gen_qpos[t, 3:7] = [np.cos(angle / 2), 0.0, 0.0, half_sin]  # wxyz
            
        return gen_qpos
    
    def generate_target_trajectory(self):
        """Generate target trajectory based on parameters."""
        target_trans = np.zeros((self.T, 2))
        target_pose = np.zeros((self.T, 4))
        
        for t in range(self.T):
            progress = t / (self.T - 1)
            # Position
            target_trans[t, 0] = self.target_x_end * progress
            target_trans[t, 1] = self.target_y_end * progress
            
            # Rotation
            angle = self.target_rot_end * progress
            half_sin = np.sin(angle / 2)
            target_pose[t] = [np.cos(angle / 2), 0.0, 0.0, half_sin]  # wxyz
            
        return target_trans, target_pose
    
    def update_trajectories(self):
        """Regenerate all trajectories based on current parameters."""
        self.gen_qpos = self.generate_gen_qpos()
        self.target_trans, self.target_pose = self.generate_target_trajectory()
        self.blended_qpos = blend_trajectories(
            self.gen_qpos, self.target_trans, self.target_pose, self.blend
        )
        
    def update_pose(self):
        """Update MuJoCo model with current frame pose."""
        qpos = self.blended_qpos[self.current_frame]
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
    
    def step(self):
        """Step the animation forward."""
        if not self.playing:
            return
        
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Check if enough time has passed for next frame
        if dt >= self.frame_dt / self.playback_speed:
            self.current_frame = (self.current_frame + 1) % self.T
            self.update_pose()
            self.last_update_time = current_time
    
    def render_trajectory(self, viewer: mujoco.viewer.Handle):
        """Render all trajectories in different colors."""
        if not self.show_trajectory:
            return
        
        viewer.user_scn.ngeom = 0
        
        # Extract orientation data
        gen_orient = self.gen_qpos[:, 3:7]
        blended_orient = self.blended_qpos[:, 3:7]
        
        # Draw generated trajectory (Gray)
        draw_trajectory(viewer, self.gen_qpos[:, :3], gen_orient, 
                       color=[0.5, 0.5, 0.5, 0.7])
        
        # Draw target trajectory (Red)
        # Need to create full 3D positions from 2D target_trans
        target_3d = np.zeros((self.T, 3))
        target_3d[:, :2] = self.target_trans
        target_3d[:, 2] = 0.95  # Same height
        draw_trajectory(viewer, target_3d, self.target_pose, 
                       color=[1.0, 0.2, 0.2, 0.9])
        
        # Draw blended trajectory (Green)
        draw_trajectory(viewer, self.blended_qpos[:, :3], blended_orient, 
                       color=[0.2, 1.0, 0.2, 1.0])

    def render_overlay(self, viewer: mujoco.viewer.Handle):
        """Render text overlay with blend coefficient."""
        # The viewer object in passive mode uses a different rendering context
        # We need to render text using the user_scn text objects
        if hasattr(viewer, '_overlay_text'):
            # Update overlay text if it exists
            pass
        
        # Alternative: Print status to title or use external rendering
        # For now, we'll rely on the visual indicator and console output
        pass

    # Control methods
    def adjust_blend(self, delta):
        """Adjust blend coefficient."""
        self.blend = np.clip(self.blend + delta, 0.0, 1.0)
        self.update_trajectories()
        print(f"\n{'='*60}")
        print(f"BLEND: {self.blend:.3f} (0.0=Gen, 1.0=Target)")
        print(f"{'='*60}\n")
    
    def adjust_target_x(self, delta):
        """Adjust target X endpoint."""
        self.target_x_end += delta
        self.update_trajectories()
        print(f"Target X end: {self.target_x_end:.2f}")
    
    def adjust_target_y(self, delta):
        """Adjust target Y endpoint."""
        self.target_y_end += delta
        self.update_trajectories()
        print(f"Target Y end: {self.target_y_end:.2f}")
    
    def adjust_target_rot(self, delta):
        """Adjust target rotation endpoint."""
        self.target_rot_end += delta
        self.update_trajectories()
        print(f"Target rotation end: {np.degrees(self.target_rot_end):.1f}°")
    
    def adjust_gen_x(self, delta):
        """Adjust generated X endpoint."""
        self.gen_x_end += delta
        self.update_trajectories()
        print(f"Generated X end: {self.gen_x_end:.2f}")
    
    def adjust_gen_y(self, delta):
        """Adjust generated Y endpoint."""
        self.gen_y_end += delta
        self.update_trajectories()
        print(f"Generated Y end: {self.gen_y_end:.2f}")
    
    def adjust_gen_rot(self, delta):
        """Adjust generated rotation endpoint."""
        self.gen_rot_end += delta
        self.update_trajectories()
        print(f"Generated rotation end: {np.degrees(self.gen_rot_end):.1f}°")
    
    def toggle_pause(self):
        """Toggle play/pause."""
        self.playing = not self.playing
        print(f"{'Playing' if self.playing else 'Paused'}")
    
    def next_frame(self):
        """Advance one frame."""
        self.current_frame = (self.current_frame + 1) % self.T
        self.update_pose()
    
    def prev_frame(self):
        """Go back one frame."""
        self.current_frame = (self.current_frame - 1) % self.T
        self.update_pose()
    
    def toggle_trajectory(self):
        """Toggle trajectory visualization."""
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")
    
    def set_speed(self, speed):
        """Set playback speed multiplier."""
        self.playback_speed = speed
        print(f"Playback speed: {speed}x")
    
    def print_parameters(self):
        """Print current parameter values."""
        print("\n" + "=" * 60)
        print("Current Parameters:")
        print("-" * 60)
        print(f"Blend coefficient: {self.blend:.3f}")
        print(f"  0.0 = Pure generated, 1.0 = Pure target")
        print()
        print("Generated trajectory endpoints:")
        print(f"  X: {self.gen_x_end:.2f}, Y: {self.gen_y_end:.2f}")
        print(f"  Rotation: {np.degrees(self.gen_rot_end):.1f}°")
        print()
        print("Target trajectory endpoints:")
        print(f"  X: {self.target_x_end:.2f}, Y: {self.target_y_end:.2f}")
        print(f"  Rotation: {np.degrees(self.target_rot_end):.1f}°")
        print("=" * 60 + "\n")


def key_callback(blender: InteractiveTrajectoryBlender, keycode, shift_held=False):
    """Handle keyboard input."""
    
    # Blend coefficient adjustment
    if keycode == ord('b') or keycode == ord('B'):
        delta = 0.01 if shift_held else 0.05
        blender.adjust_blend(-delta)
    elif keycode == ord('n') or keycode == ord('N'):
        delta = 0.01 if shift_held else 0.05
        blender.adjust_blend(delta)
    
    # Target trajectory adjustments
    elif keycode == ord('q') or keycode == ord('Q'):
        blender.adjust_target_x(-0.5)
    elif keycode == ord('w') or keycode == ord('W'):
        blender.adjust_target_x(0.5)
    elif keycode == ord('a') or keycode == ord('A'):
        blender.adjust_target_y(-0.5)
    elif keycode == ord('s') or keycode == ord('S'):
        blender.adjust_target_y(0.5)
    elif keycode == ord('z') or keycode == ord('Z'):
        blender.adjust_target_rot(-np.pi / 12)  # -15 degrees
    elif keycode == ord('x') or keycode == ord('X'):
        blender.adjust_target_rot(np.pi / 12)   # +15 degrees
    
    # Generated trajectory adjustments
    elif keycode == ord('e') or keycode == ord('E'):
        blender.adjust_gen_x(-0.5)
    elif keycode == ord('r') or keycode == ord('R'):
        blender.adjust_gen_x(0.5)
    elif keycode == ord('d') or keycode == ord('D'):
        blender.adjust_gen_y(-0.5)
    elif keycode == ord('f') or keycode == ord('F'):
        blender.adjust_gen_y(0.5)
    elif keycode == ord('c') or keycode == ord('C'):
        blender.adjust_gen_rot(-np.pi / 12)  # -15 degrees
    elif keycode == ord('v') or keycode == ord('V'):
        blender.adjust_gen_rot(np.pi / 12)   # +15 degrees
    
    # Playback controls
    elif keycode == 32:  # SPACE
        blender.toggle_pause()
    elif keycode == 263:  # LEFT
        blender.prev_frame()
    elif keycode == 262:  # RIGHT
        blender.next_frame()
    elif keycode == ord('t') or keycode == ord('T'):
        blender.toggle_trajectory()
    elif keycode == ord('p') or keycode == ord('P'):
        blender.print_parameters()
    elif ord('1') <= keycode <= ord('9'):
        speed_map = {
            ord('1'): 0.25, ord('2'): 0.5, ord('3'): 0.75,
            ord('4'): 0.9, ord('5'): 1.0, ord('6'): 1.25,
            ord('7'): 1.5, ord('8'): 1.75, ord('9'): 2.0,
        }
        blender.set_speed(speed_map[keycode])


def print_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("Interactive Trajectory Blending Visualizer")
    print("=" * 60)
    print("\nTrajectory Colors:")
    print("  Gray  : Generated trajectory")
    print("  Red   : Target trajectory")
    print("  Green : Blended trajectory (shown on humanoid)")
    print("  Yellow: Current position marker")
    print("\nControls:")
    print("-" * 60)
    print("Blend Coefficient:")
    print("  B / N           : Decrease/Increase blend (step: 0.05)")
    print("  SHIFT+B / N     : Fine adjust blend (step: 0.01)")
    print()
    print("Target Trajectory (Red):")
    print("  Q / W           : Decrease/Increase X endpoint")
    print("  A / S           : Decrease/Increase Y endpoint")
    print("  Z / X           : Decrease/Increase rotation (±15°)")
    print()
    print("Generated Trajectory (Gray):")
    print("  E / R           : Decrease/Increase X endpoint")
    print("  D / F           : Decrease/Increase Y endpoint")
    print("  C / V           : Decrease/Increase rotation (±15°)")
    print()
    print("Playback:")
    print("  SPACE           : Pause/Resume")
    print("  LEFT / RIGHT    : Previous/Next frame (when paused)")
    print("  1-9             : Set playback speed (1=0.25x, 5=1x, 9=2x)")
    print()
    print("Other:")
    print("  T               : Toggle trajectory visualization")
    print("  P               : Print current parameters")
    print("  ESC             : Exit")
    print("=" * 60 + "\n")


def main():
    # Paths
    scene_path = os.path.join(
        VISUALIZE_DIR,
        "assets",
        "scene.xml"
    )
    
    print("=" * 60)
    print("Interactive Trajectory Blending Visualizer")
    print("=" * 60)
    
    # Load MuJoCo model
    print(f"\nLoading MuJoCo scene: {scene_path}")
    if not os.path.exists(scene_path):
        print(f"✗ Scene file not found: {scene_path}")
        return
    
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Create interactive blender
    blender = InteractiveTrajectoryBlender(mj_model, mj_data)
    
    print_instructions()
    blender.print_parameters()
    
    # Track shift key state (approximate)
    shift_held = False
    
    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=lambda keycode: key_callback(blender, keycode, shift_held)
    ) as viewer:
        # Set up top-down camera view
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 90.0  # Rotate camera to align X-axis to the right
        viewer.cam.elevation = -89.9  # Top-down view (nearly 90 degrees down)
        viewer.cam.distance = 15.0  # Distance from lookat point
        viewer.cam.lookat[:] = [5.0, 5.0, 0.0]  # Look at center of trajectory area
        
        viewer.sync()
        while viewer.is_running():
            blender.step()
            blender.render_trajectory(viewer)
            draw_label(
                viewer,
                position=np.array([0.0, 0.0, 2.0]),
                label=f"Blend: {blender.blend:.2f} (B/N to adjust)",
                size=2,
            )
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
