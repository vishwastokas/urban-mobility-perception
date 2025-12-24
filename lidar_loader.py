import numpy as np

def load_lidar(bin_path):
    """
    Loads KITTI .bin LiDAR file
    Returns Nx4 array (x, y, z, intensity)
    """
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
