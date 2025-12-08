"""
Step 1: Test MuJoCo Model Loading
----------------------------------
This script verifies that:
1. MuJoCo can load your scene.xml and g1_29dof model
2. The robot displays correctly
3. Joint names match expected order from make_pose_data_g1.py

Run this to verify the basic setup before loading motion data.
"""

import os
import sys
import mujoco
import mujoco.viewer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # Path to your scene XML
    scene_path = os.path.join(
        os.path.dirname(__file__), 
        "assets", 
        "scene.xml"
    )
    
    print("=" * 60)
    print("Step 1: Testing MuJoCo Model Loading")
    print("=" * 60)
    print(f"\nLoading scene from: {scene_path}")
    
    # Load the model
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Print model info
    print(f"\n{'Model Information':^60}")
    print("-" * 60)
    print(f"Number of bodies:      {model.nbody}")
    print(f"Number of joints:      {model.njnt}")
    print(f"Number of DOFs (qpos): {model.nq}")
    print(f"Number of DOFs (qvel): {model.nv}")
    
    # Expected joint order from make_pose_data_g1.py
    expected_joints = [
        "floating_base_joint",  # 7 DOF (3 pos + 4 quat)
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    
    # List all joints in the model
    print(f"\n{'Joint Names and DOF Addresses':^60}")
    print("-" * 60)
    print(f"{'Joint Name':<35} {'qpos_adr':<10} {'qpos_dim':<10}")
    print("-" * 60)
    
    actual_joints = []
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        qpos_adr = model.jnt_qposadr[i]
        joint_type = model.jnt_type[i]
        
        # Joint types: 0=free (7 dof), 1=ball (4 dof), 2=slide (1 dof), 3=hinge (1 dof)
        if joint_type == 0:  # free joint
            qpos_dim = 7
        elif joint_type == 1:  # ball joint
            qpos_dim = 4
        else:  # slide or hinge
            qpos_dim = 1
            
        print(f"{joint_name:<35} {qpos_adr:<10} {qpos_dim:<10}")
        actual_joints.append(joint_name)
    
    # Verify joint order matches expected
    print(f"\n{'Joint Order Verification':^60}")
    print("-" * 60)
    
    if actual_joints == expected_joints:
        print("✓ Joint order matches expected configuration!")
    else:
        print("⚠ Joint order differs from expected:")
        for i, (exp, act) in enumerate(zip(expected_joints, actual_joints)):
            if exp != act:
                print(f"  Index {i}: expected '{exp}', got '{act}'")
    
    # Check qpos size (should be 7 + 29 = 36)
    expected_qpos_size = 36
    print(f"\nExpected qpos size: {expected_qpos_size}")
    print(f"Actual qpos size:   {model.nq}")
    
    if model.nq == expected_qpos_size:
        print("✓ qpos size matches (7 root + 29 joints = 36)")
    else:
        print(f"⚠ qpos size mismatch!")
    
    print("\n" + "=" * 60)
    print("Opening MuJoCo viewer...")
    print("The robot should be displayed in a neutral pose.")
    print("Press ESC or close the window to exit.")
    print("=" * 60 + "\n")
    
    # Reset to neutral pose
    mujoco.mj_resetData(model, data)
    
    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run simulation loop
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer with updated state
            viewer.sync()


if __name__ == "__main__":
    main()
