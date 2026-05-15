"""
Step 6 (DiP variant) – WASD+QE motion demo for a MotionDiffusionDiP
checkpoint (network/models_dip2d.py).

This is a thin wrapper over ``step6_demo_control.py``. Both models expose
the same positional ``forward(x, timesteps, past, traj_pose, traj_trans,
style_idx, sensor)`` contract, so all the demo plumbing (key listener,
trajectory blending, CFG wrapper, inertialisation, video recording)
carries over unchanged — we only need to swap the model class.

Usage
-----
    python visualize/step6_demo_control_dip.py \\
        --checkpoint save/dip_avoid2d_yc_ep3k/best.pt \\
        --dataset lafan1_g1_motion30 \\
        --obstacle-x 2.0 --obstacle-radius 0.4
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visualize.step6_demo_control as base
from network.models_dip2d import MotionDiffusionDiP

# The base demo instantiates ``MotionDiffusionEnv`` by symbol lookup
# inside main(); rebinding the name here is enough to redirect it.
base.MotionDiffusionEnv = MotionDiffusionDiP


if __name__ == "__main__":
    base.main()
