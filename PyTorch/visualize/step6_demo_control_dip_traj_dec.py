"""
Step 6 (DIPTRAJ-Dec experimental hybrid) – WASD+QE motion demo for a
MotionDiffusionDipTrajDec checkpoint (network/models_dip2d_traj_dec.py).

Thin wrapper over ``step6_demo_control.py``. The model exposes the same
positional ``forward(x, timesteps, past, traj_pose, traj_trans, style_idx,
sensor)`` contract as the env-sensor / DIPTRAJ models, so the demo
plumbing carries over unchanged.

Usage
-----
    python visualize/step6_demo_control_dip_traj_dec.py \\
        --checkpoint save/dip_traj_dec_yc_ep3k/best.pt \\
        --dataset lafan1_g1_motion30 \\
        --obstacle-x 2.0 --obstacle-radius 0.4
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visualize.step6_demo_control as base
from network.models_dip2d_traj_dec import MotionDiffusionDipTrajDec

base.MotionDiffusionEnv = MotionDiffusionDipTrajDec


if __name__ == "__main__":
    base.main()
