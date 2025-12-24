import numpy as np


def project_lidar_to_image(lidar_points, P2, R0, Tr):
    """
    Approximate LiDAR projection for KITTI Tracking.
    Uses LiDAR forward distance (X axis) as depth.
    """

    # LiDAR points (x, y, z)
    xyz = lidar_points[:, :3]

    # Forward distance in KITTI LiDAR frame
    depths = xyz[:, 0]

    # Dummy image points (not used for exact projection)
    img_points = np.zeros((xyz.shape[0], 2), dtype=np.float32)

    return img_points, depths

