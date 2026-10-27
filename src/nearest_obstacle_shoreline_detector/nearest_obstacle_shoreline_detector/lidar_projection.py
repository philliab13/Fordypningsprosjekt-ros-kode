"""Height filtering and orthogonal projection onto a horizontal water plane."""
import numpy as np


def project_to_water(points, water_from_lidar, water_level_m=0.0,
                     min_height_m=0.0, max_height_m=2.0):
    """Return filtered XYZ, projected XYZ, and original heights in metres.

    Transform before filtering: sensor z need not be vertical. Height interval
    is [min_height_m, max_height_m). Water is z=water_level_m in the output frame.
    """
    if (not np.isfinite([water_level_m, min_height_m, max_height_m]).all()
            or min_height_m >= max_height_m):
        raise ValueError('Water level and height limits must be finite, with min < max')
    transform = np.asarray(water_from_lidar, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected a finite 4x4 LiDAR-to-water transform')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    heights = xyz[:, 2] - water_level_m
    keep = np.isfinite(xyz).all(axis=1) & (heights >= min_height_m) & (heights < max_height_m)
    filtered, heights = xyz[keep], heights[keep]
    projected = filtered.copy()
    projected[:, 2] = water_level_m
    return filtered, projected, heights
