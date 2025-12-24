import numpy as np


def read_calib_file(filepath):
    data = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line == "" or ":" not in line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            key = parts[0].strip()
            values = parts[1].strip().split()

            try:
                data[key] = np.array([float(v) for v in values], dtype=np.float32)
            except:
                continue

    return data


def get_matrices(calib):
    # Camera projection matrix
    if "P2" not in calib:
        raise KeyError("P2 not found in calibration file")

    P2 = calib["P2"].reshape(3, 4)

    # KITTI Tracking images are already rectified
    R0 = np.eye(3, dtype=np.float32)

    # KITTI Tracking does NOT provide LiDAR->Camera extrinsics
    Tr = None

    return P2, R0, Tr

