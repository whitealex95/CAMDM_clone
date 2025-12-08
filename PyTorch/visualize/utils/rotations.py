import numpy as np

def rot_from_wxyz(quat):
    """
    Convert Quaternion [w, x, y, z] to 3x3 Rotation Matrix.
    Standard analytical formula avoids external dependencies like scipy.
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

