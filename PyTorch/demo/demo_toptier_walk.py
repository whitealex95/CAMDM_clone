import argparse
import atexit
import copy
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import utils.common as common
from diffusion.create_diffusion import create_gaussian_diffusion
from network.models_object import MotionDiffusionObject
from visualize.motion_loader import MotionDataset
from visualize.step3_demo_object import MotionGeneratorObject
from visualize.utils.geometry import draw_trajectory
from visualize.utils.transition_manager import create_transition_manager
from visualize.utils.trajectory import blend_trajectory, extend_future_traj_heusristic


DEFAULT_SCENE_XML = Path(
    "/home/jkim3662/Projects/TopTier_Procthor/mujoco_scene_library/train_00017__5-room/scene.xml"
)
DEFAULT_WAYPOINT_JSON = Path(
    "/home/jkim3662/Projects/TopTier_Procthor/mujoco_scene_library/train_00017__5-room/navigation_test/navigation_waypoints.json"
)
# DEFAULT_CHECKPOINT = ROOT / "save/camdm_g1_object_global_wandb_xyz_lr3e-4_merged_object_motion/best.pt"
DEFAULT_CHECKPOINT = ROOT / "save/camdm_g1_object_global_wandb_xyz_lr3e-4_merged_object_motion/best.pt"
DEFAULT_DATASET = ROOT / "data/pkls/merged_object_motion.pkl"
DEFAULT_ROBOT_XML = ROOT / "visualize/assets/g1_29dof_rev_1_0.xml"
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
NEUTRAL_OBJECT_WORLD = np.array([0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_wxyz_from_yaw(yaw):
    quat_xyzw = R.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def yaw_from_quat_wxyz(quat_wxyz):
    rot = R.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    forward = rot.apply([1.0, 0.0, 0.0])
    return float(np.arctan2(forward[1], forward[0]))


def rotate_quat_about_z(quat_wxyz, yaw_delta):
    base = R.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    yaw_rot = R.from_euler("z", yaw_delta)
    quat_xyzw = (yaw_rot * base).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float32)


def transform_qpos_sequence(qpos_seq, target_xy, target_yaw):
    qpos_seq = np.asarray(qpos_seq, dtype=np.float32).copy()
    ref_xy = qpos_seq[-1, :2].copy()
    ref_yaw = yaw_from_quat_wxyz(qpos_seq[-1, 3:7])
    yaw_delta = target_yaw - ref_yaw
    rot2 = R.from_euler("z", yaw_delta).as_matrix()[:2, :2]

    centered_xy = qpos_seq[:, :2] - ref_xy[None, :]
    qpos_seq[:, :2] = centered_xy @ rot2.T + np.asarray(target_xy, dtype=np.float32)[None, :]
    for i in range(qpos_seq.shape[0]):
        qpos_seq[i, 3:7] = rotate_quat_about_z(qpos_seq[i, 3:7], yaw_delta)
    return qpos_seq


def load_waypoints(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    waypoints = np.asarray(data["waypoints_world_xy"], dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2:
        raise ValueError(f"Invalid waypoints in {json_path}")
    return data, waypoints


class WaypointPath:
    def __init__(self, waypoints, speed_mps):
        filtered = [np.asarray(waypoints[0], dtype=np.float32)]
        for point in np.asarray(waypoints[1:], dtype=np.float32):
            if np.linalg.norm(point - filtered[-1]) > 1e-6:
                filtered.append(point)
        self.waypoints = np.asarray(filtered, dtype=np.float32)
        if len(self.waypoints) < 2:
            raise ValueError("Need at least 2 unique waypoints.")

        self.segment_vecs = self.waypoints[1:] - self.waypoints[:-1]
        self.segment_lengths = np.linalg.norm(self.segment_vecs, axis=1)
        self.segment_dirs = self.segment_vecs / np.maximum(self.segment_lengths[:, None], 1e-8)
        self.cumulative = np.concatenate([[0.0], np.cumsum(self.segment_lengths)])
        self.total_length = float(self.cumulative[-1])
        self.speed_mps = float(speed_mps)
        self.progress_s = 0.0
        self.final_yaw = float(np.arctan2(self.segment_dirs[-1, 1], self.segment_dirs[-1, 0]))

    @property
    def start_xy(self):
        return self.waypoints[0]

    @property
    def start_yaw(self):
        return float(np.arctan2(self.segment_dirs[0, 1], self.segment_dirs[0, 0]))

    def reset(self):
        self.progress_s = 0.0

    def project(self, point_xy):
        point_xy = np.asarray(point_xy, dtype=np.float32)
        best_dist2 = np.inf
        best_s = 0.0
        for idx, (start, vec, seg_len) in enumerate(
            zip(self.waypoints[:-1], self.segment_vecs, self.segment_lengths)
        ):
            if seg_len < 1e-8:
                continue
            t = float(np.clip(np.dot(point_xy - start, vec) / (seg_len * seg_len), 0.0, 1.0))
            proj = start + t * vec
            dist2 = float(np.sum((point_xy - proj) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_s = float(self.cumulative[idx] + t * seg_len)
        return best_s

    def sample_xy(self, s):
        s = float(np.clip(s, 0.0, self.total_length))
        if s >= self.total_length:
            return self.waypoints[-1].copy()
        idx = int(np.searchsorted(self.cumulative[1:], s, side="right"))
        idx = min(idx, len(self.segment_lengths) - 1)
        local_s = s - self.cumulative[idx]
        return self.waypoints[idx] + self.segment_dirs[idx] * local_s

    def sample_yaw(self, s, delta=0.1):
        s = float(np.clip(s, 0.0, self.total_length))
        p0 = self.sample_xy(max(0.0, s - delta))
        p1 = self.sample_xy(min(self.total_length, s + delta))
        diff = p1 - p0
        if np.linalg.norm(diff) < 1e-6:
            return self.final_yaw
        return float(np.arctan2(diff[1], diff[0]))

    def build_future(self, current_xy, horizon, frame_dt):
        projected_s = self.project(current_xy)
        self.progress_s = max(self.progress_s, projected_s)

        step_dist = self.speed_mps * float(frame_dt)
        samples_s = self.progress_s + step_dist * np.arange(1, horizon + 1, dtype=np.float32)
        traj_xy = np.stack([self.sample_xy(s) for s in samples_s], axis=0).astype(np.float32)
        traj_quat = np.stack([quat_wxyz_from_yaw(self.sample_yaw(s)) for s in samples_s], axis=0)
        remaining = max(0.0, self.total_length - self.progress_s)
        return traj_xy, traj_quat.astype(np.float32), remaining

    def make_display_path(self, spacing=0.25, z=0.05):
        num = max(2, int(np.ceil(self.total_length / spacing)) + 1)
        samples_s = np.linspace(0.0, self.total_length, num=num, dtype=np.float32)
        path_xy = np.stack([self.sample_xy(s) for s in samples_s], axis=0)
        path_z = np.full((path_xy.shape[0], 1), z, dtype=np.float32)
        path_xyz = np.concatenate([path_xy, path_z], axis=1)
        path_quat = np.stack([quat_wxyz_from_yaw(self.sample_yaw(s)) for s in samples_s], axis=0)
        return path_xyz, path_quat.astype(np.float32)


def build_seed_history(walk_motion, past_frames, seed_frame, target_xy, target_yaw):
    seed_frame = int(np.clip(seed_frame, past_frames, walk_motion.num_frames - 1))
    seed_history = walk_motion.get_past_qpos(seed_frame, past_frames=past_frames)
    if len(seed_history) != past_frames:
        raise ValueError("Failed to build initial past motion history.")
    return transform_qpos_sequence(seed_history, target_xy, target_yaw)


def traj_xy_to_xyz(traj_xy, z_value=0.05):
    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    z = np.full((traj_xy.shape[0], 1), z_value, dtype=np.float32)
    return np.concatenate([traj_xy, z], axis=1)


def _set_mesh_paths_absolute(asset_elem, mesh_dir):
    for child in asset_elem:
        if child.tag == "mesh" and "file" in child.attrib:
            child.set("file", str((mesh_dir / child.get("file")).resolve()))


def build_combined_scene_xml(scene_xml_path, robot_xml_path):
    scene_xml_path = Path(scene_xml_path).resolve()
    robot_xml_path = Path(robot_xml_path).resolve()

    scene_root = ET.parse(scene_xml_path).getroot()
    robot_root = ET.parse(robot_xml_path).getroot()

    robot_asset_sections = robot_root.findall("asset")
    if not robot_asset_sections:
        raise ValueError(f"No <asset> found in {robot_xml_path}")

    final_root = ET.Element("mujoco", model="toptier_g1_walk_demo")

    for tag in ("option", "compiler", "size"):
        elem = scene_root.find(tag)
        if elem is not None:
            elem_copy = copy.deepcopy(elem)
            if tag == "compiler":
                elem_copy.set("meshdir", str(scene_xml_path.parent))
            final_root.append(elem_copy)

    robot_default = robot_root.find("default")
    if robot_default is not None:
        final_root.append(copy.deepcopy(robot_default))

    scene_stat = scene_root.find("statistic")
    if scene_stat is not None:
        final_root.append(copy.deepcopy(scene_stat))

    scene_visual = scene_root.find("visual")
    if scene_visual is not None:
        final_root.append(copy.deepcopy(scene_visual))

    final_asset = ET.SubElement(final_root, "asset")
    scene_asset = scene_root.find("asset")
    if scene_asset is not None:
        for child in scene_asset:
            final_asset.append(copy.deepcopy(child))

    robot_mesh_asset = copy.deepcopy(robot_asset_sections[0])
    robot_mesh_dir = robot_xml_path.parent / robot_root.find("compiler").get("meshdir", ".")
    _set_mesh_paths_absolute(robot_mesh_asset, robot_mesh_dir)
    for child in robot_mesh_asset:
        final_asset.append(child)

    final_worldbody = ET.SubElement(final_root, "worldbody")
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise ValueError(f"No <worldbody> found in {scene_xml_path}")
    for child in scene_worldbody:
        final_worldbody.append(copy.deepcopy(child))

    robot_worldbody = robot_root.find("worldbody")
    if robot_worldbody is None:
        raise ValueError(f"No robot <worldbody> found in {robot_xml_path}")
    for child in robot_worldbody:
        final_worldbody.append(copy.deepcopy(child))

    robot_actuator = robot_root.find("actuator")
    if robot_actuator is not None:
        final_root.append(copy.deepcopy(robot_actuator))

    robot_sensor = robot_root.find("sensor")
    if robot_sensor is not None:
        final_root.append(copy.deepcopy(robot_sensor))

    tmp = tempfile.NamedTemporaryFile(prefix="toptier_g1_walk_", suffix=".xml", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    ET.ElementTree(final_root).write(tmp_path, encoding="utf-8", xml_declaration=True)
    atexit.register(lambda: tmp_path.unlink(missing_ok=True))
    return tmp_path


class WalkDemoPlayer:
    def __init__(
        self,
        model,
        data,
        motion_generator,
        path_follower,
        walk_style_idx,
        seed_history,
        future_frames,
        past_frames,
        applyframes,
        cfg_scale,
        cfg_count,
        traj_bias_pos,
        traj_bias_rot,
        inertialize,
        inertialization_mode,
        blendtime_rotation,
        blendtime_position,
        spring_halflife_position,
        spring_halflife_rotation,
    ):
        self.model = model
        self.data = data
        self.motion_generator = motion_generator
        self.path_follower = path_follower
        self.walk_style_idx = int(walk_style_idx)
        self.seed_history = np.asarray(seed_history, dtype=np.float32)
        self.future_frames = int(future_frames)
        self.past_frames = int(past_frames)
        self.apply_generated_frames = min(int(applyframes), self.future_frames)
        self.cfg_scale = float(cfg_scale)
        self.cfg_count_cache = int(cfg_count)
        self.cfg_count = int(cfg_count)
        self.traj_bias_pos = float(traj_bias_pos)
        self.traj_bias_rot = float(traj_bias_rot)
        self.object_pose_relative_frame = "root_yaw_gravity_aligned"

        self.inertialize = bool(inertialize)
        self.inertialization_mode = str(inertialization_mode).lower()
        self.blendtime_rotation = float(blendtime_rotation)
        self.blendtime_position = float(blendtime_position)
        self.spring_halflife_position = float(spring_halflife_position)
        self.spring_halflife_rotation = float(spring_halflife_rotation)

        self.show_trajectory = True
        self.camera_follow = True
        self.playing = True
        self.playback_speed = 1.0
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.last_update_time = time.time()

        self._pending_commands = deque()
        self.qpos_history = deque(maxlen=self.past_frames)

        # Explicit motion-generation I/O buffers.
        self.past_traj = None
        self.past_orient = None
        self.future_traj_dataset = None
        self.future_orient_dataset = None
        self.future_traj = None
        self.future_orient = None
        self.future_obj_traj_dataset = None
        self.future_obj_orient_dataset = None
        self.future_obj_traj = None
        self.future_obj_orient = None
        self.desired_contact = None
        self.past_qpos43 = None
        self.past_contact = None
        self.generated_qpos = None
        self.generated_future_traj = None
        self.generated_future_orient = None
        self.generated_obj_rel = None
        self.generated_contact = None
        self.current_pred_contact = 0.0
        self.generated_frame_idx = 0
        self.transition_manager = None
        self.remaining_distance = self.path_follower.total_length

        self.static_path_xyz, self.static_path_quat = self.path_follower.make_display_path()
        self.reset()

    def reset(self):
        self.path_follower.reset()
        self.qpos_history.clear()
        for qpos in self.seed_history:
            self.qpos_history.append(qpos.copy())

        self.data.qpos[:] = self.seed_history[-1].copy()
        mujoco.mj_forward(self.model, self.data)

        self.generated_qpos = None
        self.generated_future_traj = None
        self.generated_future_orient = None
        self.generated_obj_rel = None
        self.generated_contact = None
        self.current_pred_contact = 0.0
        self.generated_frame_idx = 0
        self.cfg_count = self.cfg_count_cache
        self.transition_manager = None
        self.last_update_time = time.time()
        self.update_past_trajectory()
        self.update_future_trajectory()
        print("Reset to waypoint start.")

    def process_pending_commands(self):
        while self._pending_commands:
            cmd = self._pending_commands.popleft()
            cmd()

    def toggle_pause(self):
        self.playing = not self.playing
        print("Playing" if self.playing else "Paused")

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        print(f"Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow
        print(f"Camera follow: {'ON' if self.camera_follow else 'OFF'}")

    def set_speed(self, speed):
        self.playback_speed = float(speed)
        print(f"Playback speed: {self.playback_speed}x")

    def print_status(self):
        root_xy = self.data.qpos[:2]
        print(
            f"pos=({root_xy[0]:.2f}, {root_xy[1]:.2f}) | "
            f"path={self.path_follower.progress_s:.2f}/{self.path_follower.total_length:.2f}m | "
            f"remaining={self.remaining_distance:.2f}m | "
            f"past={len(self.past_traj)} future={len(self.future_traj)} | "
            f"cfg_count={self.cfg_count} | pred_contact={self.current_pred_contact:.2f}"
        )

    def load_future_trajectory(self):
        traj_xy, traj_quat, remaining = self.path_follower.build_future(
            self.data.qpos[:2], self.future_frames, self.frame_dt
        )
        self.future_traj_dataset = traj_xy
        self.future_orient_dataset = traj_quat
        self.future_obj_traj_dataset = np.zeros((self.future_frames, 3), dtype=np.float32)
        self.future_obj_orient_dataset = np.repeat(IDENTITY_QUAT_WXYZ[None, :], self.future_frames, axis=0)
        self.desired_contact = np.zeros((self.future_frames,), dtype=np.float32)
        self.remaining_distance = remaining

    def update_future_trajectory(self):
        self.load_future_trajectory()
        self.future_obj_traj = self.future_obj_traj_dataset
        self.future_obj_orient = self.future_obj_orient_dataset

        if self.generated_qpos is None:
            self.generated_future_traj = None
            self.generated_future_orient = None
            self.future_traj = self.future_traj_dataset
            self.future_orient = self.future_orient_dataset
            return

        self.generated_future_traj = self.generated_qpos[:, :3].copy()
        self.generated_future_orient = self.generated_qpos[:, 3:7].copy()

        t_cur = self.generated_frame_idx
        model_pred_future_traj = self.generated_future_traj[t_cur + 1 :, :2]
        model_pred_future_orient = self.generated_future_orient[t_cur + 1 :]

        extended_future_traj, extended_future_orient = extend_future_traj_heusristic(
            model_pred_future_traj,
            model_pred_future_orient,
            self.future_frames,
            K=1,
        )
        blended_future_traj, blended_future_orient = blend_trajectory(
            extended_future_traj,
            extended_future_orient,
            self.future_traj_dataset,
            self.future_orient_dataset,
            blend=self.traj_bias_pos,
            blend_rot=self.traj_bias_rot,
        )
        self.future_traj = blended_future_traj.astype(np.float32)
        self.future_orient = blended_future_orient.astype(np.float32)

    def update_past_trajectory(self):
        qpos_history = np.asarray(self.qpos_history, dtype=np.float32)
        self.past_traj = qpos_history[:, :3].copy()
        self.past_orient = qpos_history[:, 3:7].copy()

    def build_generator_inputs(self):
        past_qpos = np.asarray(self.qpos_history, dtype=np.float32)
        neutral_obj_world = np.repeat(NEUTRAL_OBJECT_WORLD[None, :], past_qpos.shape[0], axis=0)
        self.past_qpos43 = np.concatenate([past_qpos, neutral_obj_world], axis=-1).astype(np.float32)
        self.past_contact = np.zeros((past_qpos.shape[0],), dtype=np.float32)

    def generate_motion(self):
        # Motion generator input:
        #   past_qpos43    -> explicit past motion history + neutral object pose
        #   past_contact   -> zero contact history for walk mode
        #   future_traj    -> blended future XY trajectory
        #   future_orient  -> blended future orientation
        #   desired_contact / future_obj_* -> neutral object conditions
        # Motion generator output:
        #   generated_qpos / generated_obj_rel / generated_contact
        self.build_generator_inputs()
        effective_cfg_scale = self.cfg_scale if self.cfg_count > 0 else 1.0
        generated_qpos, generated_obj_rel, generated_contact = self.motion_generator.generate_motion(
            self.past_qpos43,
            self.past_contact,
            self.future_traj,
            self.future_orient,
            self.desired_contact,
            self.future_obj_traj,
            self.future_obj_orient,
            self.walk_style_idx,
            cfg_scale=effective_cfg_scale,
            frame_mode=self.object_pose_relative_frame,
            has_object=False,
        )
        if self.cfg_count > 0:
            self.cfg_count -= 1
        return generated_qpos, generated_obj_rel, generated_contact

    def _apply_pose(self, qpos):
        self.qpos_history.append(np.asarray(qpos, dtype=np.float32).copy())
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.update_past_trajectory()
        self.update_future_trajectory()

    def update_pose_raw(self):
        if self.generated_frame_idx == 0:
            self.generated_qpos, self.generated_obj_rel, self.generated_contact = self.generate_motion()
        self.current_pred_contact = float(self.generated_contact[self.generated_frame_idx])
        self._apply_pose(self.generated_qpos[self.generated_frame_idx])

    def update_pose_inertialized(self):
        if self.transition_manager is None:
            self.transition_manager = create_transition_manager(
                mode=self.inertialization_mode,
                frame_dt=self.frame_dt,
                quat_slice=slice(3, 7),
                blend_time_rotation=self.blendtime_rotation,
                blend_time_position=self.blendtime_position,
                halflife_position=self.spring_halflife_position,
                halflife_rotation=self.spring_halflife_rotation,
            )

        if self.generated_frame_idx == 0:
            self.generated_qpos, self.generated_obj_rel, self.generated_contact = self.generate_motion()
            self.transition_manager.start_transition(
                self.qpos_history, self.data.qpos.copy(), self.generated_qpos
            )

        self.current_pred_contact = float(self.generated_contact[self.generated_frame_idx])
        qpos = self.transition_manager.apply(self.generated_qpos[self.generated_frame_idx])
        self._apply_pose(qpos)

    def step(self):
        if not self.playing:
            return

        dt = time.time() - self.last_update_time
        if dt < self.frame_dt / self.playback_speed:
            return

        if self.inertialize:
            self.update_pose_inertialized()
        else:
            self.update_pose_raw()

        self.generated_frame_idx = (self.generated_frame_idx + 1) % self.apply_generated_frames
        self.last_update_time = time.time()

    def render_trajectory(self, scene):
        if not self.show_trajectory:
            return
        draw_trajectory(scene, self.static_path_xyz, self.static_path_quat, color=[1.0, 0.85, 0.2, 1.0])
        draw_trajectory(scene, traj_xy_to_xyz(self.past_traj[:, :2]), self.past_orient, color=[0.2, 0.5, 1.0, 1.0])
        draw_trajectory(scene, traj_xy_to_xyz(self.future_traj_dataset), self.future_orient_dataset, color=[1.0, 0.2, 0.2, 1.0])
        draw_trajectory(scene, traj_xy_to_xyz(self.future_traj), self.future_orient, color=[0.2, 1.0, 0.2, 1.0])
        if self.generated_future_traj is not None:
            draw_trajectory(scene, self.generated_future_traj, self.generated_future_orient, color=[0.2, 0.2, 0.2, 0.5])


def key_callback(player, keycode):
    if keycode == 32:
        player._pending_commands.append(player.toggle_pause)
    elif keycode in (ord("r"), ord("R")):
        player._pending_commands.append(player.reset)
    elif keycode in (ord("t"), ord("T")):
        player._pending_commands.append(player.toggle_trajectory)
    elif keycode in (ord("c"), ord("C")):
        player._pending_commands.append(player.toggle_camera_follow)
    elif keycode in (ord("s"), ord("S")):
        player._pending_commands.append(player.print_status)
    elif ord("1") <= keycode <= ord("9"):
        speed_map = {
            ord("1"): 0.25,
            ord("2"): 0.5,
            ord("3"): 0.75,
            ord("4"): 0.9,
            ord("5"): 1.0,
            ord("6"): 1.25,
            ord("7"): 1.5,
            ord("8"): 1.75,
            ord("9"): 2.0,
        }
        player._pending_commands.append(lambda s=speed_map[keycode]: player.set_speed(s))


def print_instruction():
    print("\n" + "=" * 68)
    print("Controls:")
    print("SPACE pause | R reset | T trajectory | C camera | S status | 1-9 speed | ESC exit")
    print("Trajectory colors: yellow=waypoints | blue=past | red=target | green=blended | gray=predicted")
    print("=" * 68 + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Waypoint-following G1 demo in a TopTier MuJoCo scene.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--waypoints", type=Path, default=DEFAULT_WAYPOINT_JSON)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--speed-mps", type=float, default=0.9, help="Target speed along the waypoint path.")
    parser.add_argument("--seed-frame", type=int, default=30, help="Frame index used to initialize the walking seed.")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--cfg-scale", type=float, default=0.5)
    parser.add_argument("--cfg-count", type=int, default=2)
    parser.add_argument("--applyframes", type=int, default=15)
    parser.add_argument("--traj-bias-pos", type=float, default=0.4)
    parser.add_argument("--traj-bias-rot", type=float, default=2.2)
    parser.add_argument("--inertialize", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--inertialization-mode", type=str, default="camdm", choices=["camdm", "spring"])
    parser.add_argument("--blendtime-rotation", type=float, default=0.2)
    parser.add_argument("--blendtime-position", type=float, default=0.2)
    parser.add_argument("--spring-halflife-position", type=float, default=0.12)
    parser.add_argument("--spring-halflife-rotation", type=float, default=0.12)
    return parser.parse_args()


def main():
    args = get_args()

    for path in (args.scene, args.waypoints, args.robot_xml, args.dataset, args.checkpoint):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    common.fixseed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waypoint_meta, waypoints = load_waypoints(args.waypoints)
    path_follower = WaypointPath(waypoints, speed_mps=args.speed_mps)

    dataset = MotionDataset(str(args.dataset))
    if "walk" not in dataset.style_to_motions:
        raise ValueError("'walk' style not found in dataset.")
    walk_motion = dataset[dataset.style_to_motions["walk"][0]]
    walk_style_idx = dataset.styles.index("walk")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    past_frames = int(config.arch.past_frame)
    future_frames = int(config.arch.future_frame)
    if args.applyframes > future_frames:
        raise ValueError(f"--applyframes must be <= future_frames ({future_frames})")

    diffusion = create_gaussian_diffusion(config)
    diffusion_model = MotionDiffusionObject(
        input_feats=199,
        nstyles=checkpoint["state_dict"]["embed_style.action_embedding"].shape[0],
        njoints=199,
        nfeats=1,
        rot_req=config.arch.rot_req,
        clip_len=config.arch.clip_len,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        traj_pose_feats=6,
        traj_trans_feats=2,
        traj_contact_feats=1,
        traj_obj_pose_feats=6,
        traj_obj_trans_feats=3,
        device=device,
    ).to(device)
    diffusion_model.load_state_dict(checkpoint["state_dict"], strict=True)
    diffusion_model.eval()

    generator = MotionGeneratorObject(
        diffusion_model,
        diffusion,
        config,
        device=device,
        sampler=args.sampler,
        cfg_scale=args.cfg_scale,
    )

    seed_history = build_seed_history(
        walk_motion,
        past_frames=past_frames,
        seed_frame=args.seed_frame,
        target_xy=path_follower.start_xy,
        target_yaw=path_follower.start_yaw,
    )

    combined_scene_path = build_combined_scene_xml(args.scene, args.robot_xml)
    mj_model = mujoco.MjModel.from_xml_path(str(combined_scene_path))
    mj_data = mujoco.MjData(mj_model)

    player = WalkDemoPlayer(
        model=mj_model,
        data=mj_data,
        motion_generator=generator,
        path_follower=path_follower,
        walk_style_idx=walk_style_idx,
        seed_history=seed_history,
        future_frames=future_frames,
        past_frames=past_frames,
        applyframes=args.applyframes,
        cfg_scale=args.cfg_scale,
        cfg_count=args.cfg_count,
        traj_bias_pos=args.traj_bias_pos,
        traj_bias_rot=args.traj_bias_rot,
        inertialize=(args.inertialize == "on"),
        inertialization_mode=args.inertialization_mode,
        blendtime_rotation=args.blendtime_rotation,
        blendtime_position=args.blendtime_position,
        spring_halflife_position=args.spring_halflife_position,
        spring_halflife_rotation=args.spring_halflife_rotation,
    )

    print("=" * 68)
    print("TopTier waypoint demo")
    print(f"scene: {waypoint_meta.get('scene_name', args.scene.stem)}")
    print(f"start: {waypoints[0].tolist()} -> goal: {waypoints[-1].tolist()}")
    print(f"walk style idx: {walk_style_idx} | speed: {args.speed_mps:.2f} m/s")
    print(f"checkpoint: {args.checkpoint} | cfg-scale: {args.cfg_scale} | cfg-count: {args.cfg_count}")
    print("=" * 68)
    print_instruction()

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=lambda keycode: key_callback(player, keycode),
    ) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20.0
        viewer.sync()
        while viewer.is_running():
            player.process_pending_commands()
            player.step()
            viewer.user_scn.ngeom = 0
            player.render_trajectory(viewer.user_scn)
            if player.camera_follow:
                viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
