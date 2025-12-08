"""
Step 1: Test MuJoCo Model Loading
----------------------------------
This script verifies that:
1. MuJoCo can load your scene.xml and g1_29dof model
2. The robot displays correctly
Run this to verify the basic setup before loading motion data.
"""

import os
import sys
import mujoco
import mujoco.viewer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATE_PHYSICS = False

def main():
    scene_path = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    # mujoco.mj_resetData(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if SIMULATE_PHYSICS:
                mujoco.mj_step(model, data) # Step the simulation
                print(f"Simulation frequency: {1.0 / model.opt.timestep} Hz")
                viewer.sync() # Sync viewer with updated state

if __name__ == "__main__":
    main()
