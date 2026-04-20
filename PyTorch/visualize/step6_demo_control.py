"""
Step 6: User-controlled motion demo
------------------------------------
Control the robot in real-time using keyboard input (Unity-style):

  W / S   : Move forward / backward  (relative to current heading)
  A / D   : Strafe left / right
  Q / E   : Turn left / right
  SPACE   : Pause / Resume
  R       : Reset pose to origin
  ESC     : Exit

Movement is relative to the robot's current heading, matching the Unity
Controller.cs scheme (W=forward, A=left, D=right, Q=turn-left, E=turn-right).

The future trajectory is built by projecting the current user input forward
over future_frames steps, exactly as CAMDM does in Unity.

Usage
-----
    python visualize/step6_demo_control.py \\
        --checkpoint save/<run>/best.pt \\
        --dataset lafan1_g1_motion30
"""

import os
import sys
import argparse
import time
import threading
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer
import torch
import imageio.v2 as imageio
from pynput import keyboard as pynput_keyboard

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.common as common
import utils.nn_transforms as nn_transforms
from network.models_env2d import MotionDiffusionEnv
from diffusion.create_diffusion import create_gaussian_diffusion

from visualize.utils.geometry import draw_trajectory, draw_sensor_readings
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic
from utils.environment_sensor import EnvironmentSensor, quat_wxyz_to_yaw


# ---------------------------------------------------------------------------
# qpos converters  (same as step4/step5)
# ---------------------------------------------------------------------------

def _yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)], dtype=np.float32)


def qpos_to_model_format(qpos_seq: np.ndarray) -> np.ndarray:
    T = qpos_seq.shape[0]
    out = np.zeros((T, 31, 6), dtype=np.float32)
    for t in range(T):
        out[t, 0] = nn_transforms.quat2repr6d(
            torch.from_numpy(qpos_seq[t, 3:7]).float().unsqueeze(0)
        ).numpy()[0]
        out[t, 1:30, 0] = qpos_seq[t, 7:]
        out[t, 30, :3]  = qpos_seq[t, :3]
    return out


def model_format_to_qpos(model_out: np.ndarray) -> np.ndarray:
    T = model_out.shape[0]
    qpos = np.zeros((T, 36), dtype=np.float32)
    for t in range(T):
        qpos[t, :3]  = model_out[t, 30, :3]
        qpos[t, 3:7] = nn_transforms.repr6d2quat(
            torch.from_numpy(model_out[t, 0]).float().unsqueeze(0)
        ).numpy()[0]
        qpos[t, 7:]  = model_out[t, 1:30, 0]
    return qpos


# ---------------------------------------------------------------------------
# Key-state tracker  (pynput background listener)
# ---------------------------------------------------------------------------

class KeyState:
    """Thread-safe set of currently-held keyboard characters."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._pressed: set = set()
        listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        listener.daemon = True
        listener.start()

    def _on_press(self, key):
        try:
            with self._lock:
                self._pressed.add(key.char.lower())
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            with self._lock:
                self._pressed.discard(key.char.lower())
        except AttributeError:
            pass

    def __contains__(self, char: str) -> bool:
        with self._lock:
            return char.lower() in self._pressed

    def any_of(self, *chars: str) -> bool:
        with self._lock:
            return any(c.lower() in self._pressed for c in chars)


# ---------------------------------------------------------------------------
# User controller  (Unity Controller.cs → Python)
# ---------------------------------------------------------------------------

class UserController:
    """
    Integrates WASD + QE key state into a future trajectory for the model.

    Coordinate convention (matches training data):
      yaw=0  → facing +X
      +yaw   → counter-clockwise (left turn)

    Controls (mirroring Unity Controller.cs):
      W / S  : forward / backward along current heading
      A / D  : strafe left / right
      Q / E  : turn left / right
    """

    def __init__(
        self,
        future_frames: int = 45,
        fps: float = 30.0,
        move_speed: float = 2.0,    # m/s  (translation)
        turn_speed: float = 1.2,    # rad/s (rotation, ~70°/s)
    ):
        self.future_frames = int(future_frames)
        self.fps        = float(fps)
        self.move_speed = float(move_speed)
        self.turn_speed = float(turn_speed)

        self.target_pos = np.zeros(2, dtype=np.float64)
        self.target_yaw = 0.0

    def reset(self, pos: np.ndarray, yaw: float):
        self.target_pos[:] = pos[:2]
        self.target_yaw    = float(yaw)

    def update(self, keys: KeyState, dt: float):
        """Advance target state by one simulation step based on held keys."""
        # Turn
        if 'q' in keys:
            self.target_yaw += self.turn_speed * dt
        if 'e' in keys:
            self.target_yaw -= self.turn_speed * dt

        # Local velocity (W/S = forward/back, A/D = left/right)
        lx, ly = 0.0, 0.0
        if 'w' in keys: ly += 1.0
        if 's' in keys: ly -= 1.0
        if 'a' in keys: lx += 1.0   # +left
        if 'd' in keys: lx -= 1.0   # -left = right

        norm = np.hypot(lx, ly)
        if norm > 1e-6:
            lx /= norm; ly /= norm

        # Rotate local → world  (yaw=0 → facing +X, +yaw → CCW)
        c, s = np.cos(self.target_yaw), np.sin(self.target_yaw)
        fwd  = np.array([ c,  s])   # forward dir in world
        left = np.array([-s,  c])   # left    dir in world

        self.target_pos += (ly * fwd + lx * left) * self.move_speed * dt

    def get_future_trajectory(self, keys: KeyState):
        """
        Project current input forward over future_frames steps.

        Returns
        -------
        traj_xy   : (future_frames, 2)  world XY
        traj_quat : (future_frames, 4)  wxyz quaternions
        """
        dt = 1.0 / self.fps

        # Current turn rate + local velocity
        turn_rate = 0.0
        if 'q' in keys: turn_rate += self.turn_speed
        if 'e' in keys: turn_rate -= self.turn_speed

        lx, ly = 0.0, 0.0
        if 'w' in keys: ly += 1.0
        if 's' in keys: ly -= 1.0
        if 'a' in keys: lx += 1.0
        if 'd' in keys: lx -= 1.0
        norm = np.hypot(lx, ly)
        if norm > 1e-6:
            lx /= norm; ly /= norm

        pos = self.target_pos.copy()
        yaw = self.target_yaw
        traj_xy, traj_quat = [], []

        for _ in range(self.future_frames):
            yaw += turn_rate * dt
            c, s = np.cos(yaw), np.sin(yaw)
            fwd  = np.array([c,  s])
            left = np.array([-s, c])
            pos  = pos + (ly * fwd + lx * left) * self.move_speed * dt
            traj_xy.append(pos.copy())
            traj_quat.append(_yaw_to_quat(float(yaw)))

        return (
            np.array(traj_xy,   dtype=np.float32),
            np.array(traj_quat, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Model wrapper & generator
# ---------------------------------------------------------------------------

class _ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, **kw):
        return self.model.forward(
            x, timesteps,
            kw.get('past_motion'), kw.get('traj_pose'),
            kw.get('traj_trans'),  kw.get('style_idx'), kw.get('sensor'),
        )


class SensorMotionGenerator:
    def __init__(self, model, diffusion, config, sensor: EnvironmentSensor,
                 device='cuda', sampler='ddpm', cfg_scale=1.0):
        self.model         = _ModelWrapper(model)
        self.diffusion     = diffusion
        self.sensor        = sensor
        self.device        = device
        self.sampler       = sampler.lower()
        self.cfg_scale     = float(cfg_scale)
        self.future_frames = config.arch.future_frame
        self.rot_req       = config.arch.rot_req
        self.per_rot_feat  = 6

    def generate_motion(
        self,
        past_qpos: np.ndarray,
        traj_trans: np.ndarray,
        traj_pose: np.ndarray,
        style_idx: int,
        obstacles: list,
        cfg_scale: float = None,
    ) -> np.ndarray:
        curr_xy = past_qpos[-1, :2].copy()

        pq = past_qpos.copy()
        pq[:, :2] -= curr_xy
        past_t = torch.from_numpy(qpos_to_model_format(pq)).float() \
            .unsqueeze(0).permute(0, 2, 3, 1).to(self.device)

        traj_tr_t = torch.from_numpy((traj_trans - curr_xy).astype(np.float32)).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)
        traj_repr = nn_transforms.get_rotation(
            torch.from_numpy(traj_pose).float(), self.rot_req
        ).numpy()
        traj_po_t = torch.from_numpy(traj_repr).float() \
            .unsqueeze(0).permute(0, 2, 1).to(self.device)

        if obstacles:
            yaw = quat_wxyz_to_yaw(past_qpos[-1, 3:7])
            readings, _ = self.sensor.compute(past_qpos[-1, :3], yaw, obstacles)
        else:
            readings = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        sensor_t = torch.from_numpy(readings).float().unsqueeze(0).to(self.device)
        style_t  = torch.tensor([style_idx]).to(self.device)

        model_kwargs = dict(
            past_motion=past_t, traj_trans=traj_tr_t, traj_pose=traj_po_t,
            sensor=sensor_t, style_idx=style_t, y={},
        )
        uncond_kwargs = dict(
            past_motion=torch.zeros_like(past_t),
            traj_trans=traj_tr_t, traj_pose=traj_po_t,
            sensor=sensor_t, style_idx=style_t, y={},
        )

        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if scale == 1.0:
            sampling_model  = self.model
            sampling_kwargs = model_kwargs
        else:
            cond_m, uncond_kw, s = self.model, uncond_kwargs, scale

            class _CFGWrap(torch.nn.Module):
                def forward(self_, x, timesteps, **kw):  # noqa: N805
                    return cond_m(x, timesteps, **kw) + \
                           s * (cond_m(x, timesteps, **kw) -
                                cond_m(x, timesteps, **uncond_kw))

            sampling_model  = _CFGWrap()
            sampling_kwargs = model_kwargs

        shape = (1, 31, self.per_rot_feat, self.future_frames)
        with torch.no_grad():
            if self.sampler == 'ddim':
                out = self.diffusion.ddim_sample_loop(
                    sampling_model, shape, clip_denoised=False,
                    model_kwargs=sampling_kwargs, progress=False, eta=0., device=self.device,
                )
            else:
                out = self.diffusion.p_sample_loop(
                    sampling_model, shape, clip_denoised=False,
                    model_kwargs=sampling_kwargs, progress=False, device=self.device,
                )
        out      = out.squeeze(0).permute(2, 0, 1).cpu().numpy()
        qpos_out = model_format_to_qpos(out)
        qpos_out[:, :2] += curr_xy
        return qpos_out


# ---------------------------------------------------------------------------
# Demo player
# ---------------------------------------------------------------------------

class ControlPlayer:
    """
    Autoregressive motion player driven by user WASD+QE input.
    """

    def __init__(
        self,
        mj_model, mj_data,
        generator: SensorMotionGenerator,
        controller: UserController,
        keys: KeyState,
        init_qpos: np.ndarray,
        style_idx: int = 0,
        obstacles: list = None,
        past_frames: int = 10,
        future_frames: int = 45,
        traj_bias_pos: float = 0.1,   # low bias → model follows user input closely
        traj_bias_rot: float = 0.5,
        cfg_count: int = 2,
        applyframes: int = 15,
        inertialize: bool = True,
        inertialization_mode: str = 'camdm',
        blendtime_rotation: float = 0.2,
        blendtime_position: float = 0.2,
        spring_halflife_position: float = 0.12,
        spring_halflife_rotation: float = 0.12,
        inertial_quat_start: int = 3,
        inertial_quat_end: int = 7,
        show_trajectory: bool = True,
        show_sensor: bool = True,
    ):
        self.mj_model   = mj_model
        self.mj_data    = mj_data
        self.generator  = generator
        self.controller = controller
        self.keys       = keys
        self.init_qpos  = init_qpos.copy()
        self.style_idx  = style_idx
        self.obstacles  = obstacles or []

        self.past_frames   = past_frames
        self.future_frames = future_frames
        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)
        self.apply_frames    = int(applyframes)
        self.gen_idx         = 0
        self.gen_qpos        = None
        self.cfg_count_cache = int(cfg_count)
        self.cfg_count       = int(cfg_count)

        self.fps          = 30
        self.frame_dt     = 1.0 / self.fps
        self.playing      = True
        self.playback_speed = 1.0
        self.last_update  = time.time()

        self.show_trajectory = show_trajectory
        self.show_sensor     = show_sensor
        self.camera_follow   = True

        self.inertialize          = bool(inertialize)
        self.inertialization_mode = str(inertialization_mode).lower()
        self.blendtime_rotation   = float(blendtime_rotation)
        self.blendtime_position   = float(blendtime_position)
        self.spring_halflife_pos  = float(spring_halflife_position)
        self.spring_halflife_rot  = float(spring_halflife_rotation)
        self.quat_slice           = slice(int(inertial_quat_start), int(inertial_quat_end))
        self.transition_mgr       = None

        self.sensor         = generator.sensor
        self.readings       = np.zeros(self.sensor.feature_dim, dtype=np.float32)
        self.sphere_centers = np.zeros((self.sensor.feature_dim, 2), dtype=np.float64)

        self.future_traj    = None
        self.future_orient  = None
        self.command_traj   = None
        self.command_orient = None
        self.past_traj      = None
        self.past_orient    = None

        self.qpos_history = deque(maxlen=past_frames)
        self._init_pose()

    # ------------------------------------------------------------------

    def _init_pose(self):
        init_yaw = quat_wxyz_to_yaw(self.init_qpos[3:7])
        self.controller.reset(self.init_qpos[:2], init_yaw)
        for _ in range(self.past_frames):
            self.qpos_history.append(self.init_qpos.copy())
        self.mj_data.qpos[:] = self.init_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._update_traj()

    def reset(self):
        self.gen_idx   = 0
        self.gen_qpos  = None
        self.cfg_count = self.cfg_count_cache
        self.transition_mgr = None
        self.qpos_history.clear()
        self._init_pose()
        print("Reset.")

    # ------------------------------------------------------------------

    def _update_traj(self):
        qh = np.array(self.qpos_history)
        self.past_traj   = qh[-self.past_frames:, :3]
        self.past_orient = qh[-self.past_frames:, 3:7]

        # Sync controller to actual robot state so trajectory starts from current position
        actual_pos = self.mj_data.qpos[:2].copy()
        actual_yaw = quat_wxyz_to_yaw(self.mj_data.qpos[3:7])
        self.controller.reset(actual_pos, float(actual_yaw))

        # Advance controller state by one dt, then build future
        self.controller.update(self.keys, self.frame_dt)
        cmd_xy, cmd_quat = self.controller.get_future_trajectory(self.keys)
        self.command_traj   = cmd_xy
        self.command_orient = cmd_quat

        if self.gen_qpos is not None:
            pred_xy  = self.gen_qpos[self.gen_idx:, :2]
            pred_ori = self.gen_qpos[self.gen_idx:, 3:7]
            ext_t, ext_o = extend_future_traj_heusristic(pred_xy, pred_ori, self.future_frames)
            self.future_traj, self.future_orient = blend_trajectory(
                ext_t, ext_o, cmd_xy, cmd_quat,
                blend=self.traj_bias_pos, blend_rot=self.traj_bias_rot,
            )
        else:
            self.future_traj   = cmd_xy
            self.future_orient = cmd_quat

    def _generate(self) -> np.ndarray:
        eff_scale = self.generator.cfg_scale if self.cfg_count > 0 else 1.0
        gen = self.generator.generate_motion(
            np.array(self.qpos_history),
            self.future_traj, self.future_orient,
            self.style_idx, self.obstacles,
            cfg_scale=eff_scale,
        )
        if self.cfg_count > 0:
            self.cfg_count -= 1
        return gen

    def update_pose(self):
        if self.inertialize:
            self._update_inertialized()
        else:
            self._update_raw()

    def _update_raw(self):
        if self.gen_idx == 0:
            self.gen_qpos = self._generate()
        qpos = self.gen_qpos[self.gen_idx]
        self.qpos_history.append(qpos.copy())
        self._update_traj()
        self.mj_data.qpos[:] = qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.gen_idx = (self.gen_idx + 1) % self.apply_frames

    def _update_inertialized(self):
        if self.transition_mgr is None:
            self.transition_mgr = create_transition_manager(
                mode=self.inertialization_mode,
                frame_dt=self.frame_dt,
                quat_slice=self.quat_slice,
                blend_time_rotation=self.blendtime_rotation,
                blend_time_position=self.blendtime_position,
                halflife_position=self.spring_halflife_pos,
                halflife_rotation=self.spring_halflife_rot,
            )
        if self.gen_idx == 0:
            self.gen_qpos = self._generate()
            self.transition_mgr.start_transition(
                self.qpos_history, self.mj_data.qpos.copy(), self.gen_qpos
            )
        raw   = self.gen_qpos[self.gen_idx]
        final = self.transition_mgr.apply(raw)
        self.qpos_history.append(final.copy())
        self._update_traj()
        self.mj_data.qpos[:] = final
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.gen_idx = (self.gen_idx + 1) % self.apply_frames

    def step(self):
        if not self.playing:
            return
        now = time.time()
        if now - self.last_update < self.frame_dt / self.playback_speed:
            return

        qpos = self.mj_data.qpos
        if self.obstacles:
            self.readings, self.sphere_centers = self.sensor.compute(
                qpos[:3], quat_wxyz_to_yaw(qpos[3:7]), self.obstacles
            )

        self.update_pose()
        self.last_update = now

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, scene, clear: bool = True):
        if clear:
            scene.ngeom = 0

        if self.show_trajectory and self.past_traj is not None:
            draw_trajectory(scene, self.past_traj, self.past_orient,
                            color=[0.2, 0.5, 1.0, 1.0])   # blue: past
            if self.command_traj is not None:
                cmd3 = np.hstack([self.command_traj,
                                   np.zeros((len(self.command_traj), 1))])
                draw_trajectory(scene, cmd3, self.command_orient,
                                color=[1.0, 0.9, 0.0, 1.0])  # yellow: user command
            if self.future_traj is not None:
                ft3 = np.hstack([self.future_traj,
                                  np.zeros((len(self.future_traj), 1))])
                draw_trajectory(scene, ft3, self.future_orient,
                                color=[0.2, 1.0, 0.2, 1.0])  # green: blended NN input

        if self.show_sensor and self.obstacles:
            draw_sensor_readings(
                scene, self.mj_data.qpos[:3], self.readings, self.sphere_centers,
                z_height=0.08, dot_radius=0.04, draw_lines=False,
            )

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="WASD+QE user-controlled motion demo")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset",       default=None,
                   help="Optional pkl for initial pose & style (e.g. lafan1_g1_motion30)")
    p.add_argument("--motion",        type=int,   default=0)
    p.add_argument("--move-speed",    type=float, default=2.0,  help="Translation speed (m/s)")
    p.add_argument("--turn-speed",    type=float, default=1.2,  help="Rotation speed (rad/s)")
    p.add_argument("--traj-bias-pos", type=float, default=0.1,
                   help="Blend toward model prediction (low=follow user closely)")
    p.add_argument("--traj-bias-rot", type=float, default=0.5)
    p.add_argument("--cfg-scale",     type=float, default=0.5)
    p.add_argument("--cfg-count",     type=int,   default=2)
    p.add_argument("--resolution",    type=int,   default=9)
    p.add_argument("--max-range",     type=float, default=2.0)
    p.add_argument("--past-frames",   type=int,   default=10)
    p.add_argument("--future-frames", type=int,   default=45)
    p.add_argument("--sampler",       default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--applyframes",   type=int,   default=15)
    p.add_argument("--inertialize",   default="on", choices=["on", "off"])
    p.add_argument("--inertialization-mode", default="camdm", choices=["camdm", "spring"])
    p.add_argument("--blendtime-rotation",   type=float, default=0.2)
    p.add_argument("--blendtime-position",   type=float, default=0.2)
    p.add_argument("--spring-halflife-position", type=float, default=0.12)
    p.add_argument("--spring-halflife-rotation",  type=float, default=0.12)
    return p.parse_args()


def print_instructions():
    print("\n" + "=" * 50)
    print("  Step 6: User-Controlled Motion  –  Controls")
    print("-" * 50)
    print("  W / S   : Forward / Backward")
    print("  A / D   : Strafe Left / Right")
    print("  Q / E   : Turn Left / Right")
    print("  SPACE   : Pause / Resume")
    print("  R       : Reset")
    print("  T       : Toggle trajectory")
    print("  C       : Toggle camera follow")
    print("  ESC     : Exit")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    print("=" * 50)
    print("  Step 6: User-Controlled Motion")
    print("=" * 50)

    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")
    mj_model   = mujoco.MjModel.from_xml_path(scene_path)
    mj_data    = mujoco.MjData(mj_model)

    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config     = checkpoint["config"]

    # ── Dataset (optional) ───────────────────────────────────────────────
    style_idx = 0
    init_qpos = np.zeros(36, dtype=np.float32)
    init_qpos[2] = 0.78   # pelvis height
    init_qpos[3] = 1.0    # quaternion w

    if args.dataset is not None:
        from visualize.motion_loader import MotionDataset
        dataset    = MotionDataset(f"data/pkls/{args.dataset}.pkl")
        motion_idx = args.motion % len(dataset)
        style_idx  = dataset[motion_idx].style_idx
        raw        = dataset[motion_idx].get_qpos(0)
        init_qpos  = raw.copy()
        init_qpos[0] = 0.0
        init_qpos[1] = 0.0
        init_qpos[3:7] = np.array([1., 0., 0., 0.])   # face +X
        num_styles = len(dataset.styles)
        print(f"Dataset: {args.dataset}  clip={motion_idx}  style={style_idx}")
    else:
        num_styles = 1

    diffusion_model = MotionDiffusionEnv(
        31 * 6, num_styles, 31, 6,
        config.arch.rot_req, config.arch.clip_len,
        env_sensor_dim=config.arch.env_sensor_dim,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=device,
    ).to(device)
    diffusion_model.load_state_dict(checkpoint["state_dict"])
    diffusion_model.eval()

    diffusion  = create_gaussian_diffusion(config)
    sensor     = EnvironmentSensor(max_range=args.max_range, resolution=args.resolution)
    generator  = SensorMotionGenerator(
        diffusion_model, diffusion, config, sensor,
        device=device, sampler=args.sampler, cfg_scale=args.cfg_scale,
    )

    keys       = KeyState()
    controller = UserController(
        future_frames=args.future_frames,
        fps=30.,
        move_speed=args.move_speed,
        turn_speed=args.turn_speed,
    )

    player = ControlPlayer(
        mj_model, mj_data, generator, controller, keys, init_qpos,
        style_idx=style_idx,
        obstacles=[],
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        traj_bias_pos=args.traj_bias_pos,
        traj_bias_rot=args.traj_bias_rot,
        cfg_count=args.cfg_count,
        applyframes=args.applyframes,
        inertialize=(args.inertialize == 'on'),
        inertialization_mode=args.inertialization_mode,
        blendtime_rotation=args.blendtime_rotation,
        blendtime_position=args.blendtime_position,
        spring_halflife_position=args.spring_halflife_position,
        spring_halflife_rotation=args.spring_halflife_rotation,
    )

    # ── Video recording ───────────────────────────────────────────────────
    os.makedirs("videos", exist_ok=True)
    video_path = f"videos/control_{time.strftime('%m%d_%H%M')}.mp4"
    W, H = 1280, 720
    writer   = imageio.get_writer(video_path, fps=player.fps, codec="libx264",
                                  pixelformat="yuv420p")
    renderer = mujoco.Renderer(mj_model, height=H, width=W)
    frame_last_time = -np.inf

    print(f"\nRecording → {video_path}")

    def _key_cb(keycode: int):
        if keycode == 32:                          # SPACE
            player.toggle_pause()
        elif keycode in (ord('r'), ord('R')):
            player.reset()
        elif keycode in (ord('t'), ord('T')):
            player.toggle_trajectory()
        elif keycode in (ord('c'), ord('C')):
            player.toggle_camera_follow()

    print_instructions()

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=_key_cb,
    ) as viewer:
        viewer.cam.distance  = 5.0
        viewer.cam.elevation = -25.0
        viewer.sync()
        try:
            while viewer.is_running():
                player.step()
                viewer.user_scn.ngeom = 0
                player.render(viewer.user_scn)
                if player.camera_follow:
                    viewer.cam.lookat[:] = mj_data.qpos[:3]
                viewer.sync()

                if time.time() - frame_last_time > 1.0 / player.fps:
                    renderer.update_scene(mj_data, camera=viewer.cam)
                    player.render(renderer.scene, clear=False)
                    writer.append_data(renderer.render())
                    frame_last_time = time.time()

                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()
            print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()
