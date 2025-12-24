import numpy as np


def estimate_distance_from_lidar(depths):
    """
    Estimate object distance using LiDAR forward distances.
    Uses median for robustness.
    """
    depths = depths[depths > 0]  # remove invalid points

    if len(depths) == 0:
        return None

    return float(np.median(depths))
