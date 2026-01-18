"""
Robot Configuration Loader

This module loads robot-specific configurations from YAML files,
enabling easy adaptation to different humanoid robots.
"""

import os
import yaml
import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class KeypointConfig:
    """Configuration for a virtual keypoint."""
    name: str
    parent_joint: str
    parent_joint_idx: int
    offset: np.ndarray  # [3] offset in parent frame
    type: str  # "foot_contact", "end_effector", "tracking", etc.


@dataclass
class RobotConfig:
    """Configuration for a humanoid robot."""
    name: str
    mjcf_path: str
    root_body_name: str
    default_height: float

    # Joint information
    joint_names: List[str]
    num_joints: int

    # Skeleton structure (auto-extracted from MuJoCo)
    joint_parents: np.ndarray  # [num_joints] parent indices
    joint_offsets: np.ndarray  # [num_joints, 3] local offsets
    joint_axes: np.ndarray     # [num_joints, 3] rotation axes
    body_init_quats: np.ndarray  # [num_joints, 4] initial orientations (wxyz)

    # Virtual keypoints
    keypoints: Dict[str, KeypointConfig] = field(default_factory=dict)

    # Visualization settings
    viz_joint_colors: Dict[str, List[float]] = field(default_factory=dict)
    viz_keypoint_colors: Dict[str, List[float]] = field(default_factory=dict)
    viz_skeleton_colors: Dict[str, List[float]] = field(default_factory=dict)
    viz_joint_ranges: Dict[str, List[int]] = field(default_factory=dict)

    # Contact settings
    foot_keypoint_names: List[str] = field(default_factory=list)
    contact_velocity_threshold: float = 0.02
    contact_height_threshold: float = 0.05
    contact_color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.8])
    no_contact_color: List[float] = field(default_factory=lambda: [0.2, 1.0, 0.2, 0.3])

    # Loss settings
    loss_position_weight: float = 1.0
    loss_foot_contact_weight: float = 0.5
    loss_velocity_weight: float = 0.1


class RobotConfigLoader:
    """Loads robot configuration from YAML and MuJoCo XML."""

    def __init__(self, config_path: str, project_root: Optional[str] = None):
        """
        Args:
            config_path: Path to robot config YAML file
            project_root: Root directory for resolving relative paths
        """
        self.config_path = config_path
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def load(self) -> RobotConfig:
        """Load complete robot configuration."""
        # Load YAML config
        with open(self.config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        robot_cfg = yaml_config['robot']

        # Resolve MJCF path
        mjcf_path = os.path.join(self.project_root, robot_cfg['mjcf_path'])

        # Load MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        # Extract skeleton from MuJoCo
        joint_names = robot_cfg['joints']['names']
        num_joints = len(joint_names) + 1  # +1 for root joint

        parents, offsets, axes, init_quats = self._extract_skeleton_from_mujoco(
            mj_model, joint_names, robot_cfg['root']['body_name']
        )

        # Create robot config
        config = RobotConfig(
            name=robot_cfg['name'],
            mjcf_path=mjcf_path,
            root_body_name=robot_cfg['root']['body_name'],
            default_height=robot_cfg['root']['default_height'],
            joint_names=joint_names,
            num_joints=num_joints,  # This includes root
            joint_parents=parents,
            joint_offsets=offsets,
            joint_axes=axes,
            body_init_quats=init_quats
        )

        # Load keypoints
        if 'keypoints' in yaml_config:
            self._load_keypoints(config, yaml_config['keypoints'], joint_names)

        # Load visualization settings
        if 'visualization' in yaml_config:
            viz = yaml_config['visualization']
            config.viz_joint_colors = viz.get('joint_colors', {})
            config.viz_keypoint_colors = viz.get('keypoint_colors', {})
            config.viz_skeleton_colors = viz.get('skeleton_colors', {})
            config.viz_joint_ranges = viz.get('joint_ranges', {})

        # Load contact settings
        if 'contact' in yaml_config:
            contact = yaml_config['contact']
            config.foot_keypoint_names = contact.get('foot_keypoints', [])
            config.contact_velocity_threshold = contact.get('velocity_threshold', 0.02)
            config.contact_height_threshold = contact.get('height_threshold', 0.05)
            config.contact_color = contact.get('contact_color', [1.0, 0.0, 0.0, 0.8])
            config.no_contact_color = contact.get('no_contact_color', [0.2, 1.0, 0.2, 0.3])

        # Load loss settings
        if 'losses' in yaml_config:
            losses = yaml_config['losses']
            config.loss_position_weight = losses.get('position_weight', 1.0)
            config.loss_foot_contact_weight = losses.get('foot_contact_weight', 0.5)
            config.loss_velocity_weight = losses.get('velocity_weight', 0.1)

        return config

    def _extract_skeleton_from_mujoco(
        self,
        model: mujoco.MjModel,
        joint_names: List[str],
        root_body_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract skeleton structure from MuJoCo model.

        Returns:
            parents: [num_joints+1] parent indices (-1 for root)
            offsets: [num_joints+1, 3] local offsets
            axes: [num_joints+1, 3] rotation axes
            init_quats: [num_joints+1, 4] initial orientations (wxyz)
        """
        num_joints = len(joint_names) + 1  # +1 for root

        parents = np.zeros(num_joints, dtype=np.int32)
        offsets = np.zeros((num_joints, 3), dtype=np.float32)
        axes = np.zeros((num_joints, 3), dtype=np.float32)
        init_quats = np.zeros((num_joints, 4), dtype=np.float32)

        # Root has no parent
        parents[0] = -1
        offsets[0] = [0, 0, 0]
        axes[0] = [0, 0, 0]
        init_quats[0] = [1, 0, 0, 0]  # Identity quaternion (wxyz)

        # Build joint name to MuJoCo joint index mapping
        mj_joint_map = {}
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            mj_joint_map[jnt_name] = i

        # Build body name to body index mapping
        body_map = {}
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_map[body_name] = i

        # Extract data for each joint
        for idx, jnt_name in enumerate(joint_names, start=1):
            if jnt_name not in mj_joint_map:
                raise ValueError(f"Joint '{jnt_name}' not found in MuJoCo model")

            mj_jnt_idx = mj_joint_map[jnt_name]

            # Get body this joint is attached to
            body_id = model.jnt_bodyid[mj_jnt_idx]

            # Get parent body
            parent_body_id = model.body_parentid[body_id]

            # Find parent joint index in our skeleton
            parent_joint_idx = 0  # Default to root
            if parent_body_id > 0:  # Not world, not root body
                # Find which joint is attached to parent body
                for j in range(model.njnt):
                    if model.jnt_bodyid[j] == parent_body_id:
                        parent_jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                        # Find in our joint list
                        if parent_jnt_name in joint_names:
                            parent_joint_idx = joint_names.index(parent_jnt_name) + 1
                        break

            parents[idx] = parent_joint_idx

            # Get offset (body position relative to parent)
            offsets[idx] = model.body_pos[body_id]

            # Get joint axis
            axes[idx] = model.jnt_axis[mj_jnt_idx]

            # Get initial body quaternion (wxyz format in MuJoCo)
            init_quats[idx] = model.body_quat[body_id]

        return parents, offsets, axes, init_quats

    def _load_keypoints(
        self,
        config: RobotConfig,
        keypoints_cfg: Dict,
        joint_names: List[str]
    ):
        """Load virtual keypoint configurations."""
        for kp_name, kp_cfg in keypoints_cfg.items():
            parent_joint_name = kp_cfg['parent_joint']

            # Find parent joint index
            if parent_joint_name not in joint_names:
                raise ValueError(f"Parent joint '{parent_joint_name}' not found for keypoint '{kp_name}'")

            parent_joint_idx = joint_names.index(parent_joint_name) + 1  # +1 for root

            keypoint = KeypointConfig(
                name=kp_name,
                parent_joint=parent_joint_name,
                parent_joint_idx=parent_joint_idx,
                offset=np.array(kp_cfg['offset'], dtype=np.float32),
                type=kp_cfg['type']
            )

            config.keypoints[kp_name] = keypoint


def load_robot_config(config_name: str = "g1") -> RobotConfig:
    """
    Convenience function to load a robot configuration by name.

    Args:
        config_name: Name of robot config file (without .yaml extension)

    Returns:
        RobotConfig object
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "robots", f"{config_name}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Robot config not found: {config_path}")

    loader = RobotConfigLoader(config_path, project_root)
    return loader.load()


# Example usage
if __name__ == "__main__":
    config = load_robot_config("g1")

    print("=" * 80)
    print(f"Loaded Robot: {config.name}")
    print("=" * 80)
    print(f"Number of joints: {config.num_joints}")
    print(f"Number of keypoints: {len(config.keypoints)}")
    print()

    print("Joint hierarchy (first 10):")
    for i in range(min(10, config.num_joints)):
        if i == 0:
            print(f"  {i}: root (no parent)")
        else:
            print(f"  {i}: {config.joint_names[i-1]} -> parent {config.joint_parents[i]}")
    print()

    print("Virtual keypoints:")
    for kp_name, kp in config.keypoints.items():
        print(f"  {kp_name}: parent={kp.parent_joint} (idx={kp.parent_joint_idx}), "
              f"offset={kp.offset}, type={kp.type}")
