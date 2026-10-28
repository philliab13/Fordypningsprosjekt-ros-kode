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


def nearest_boundary(points, origin_xy, sector_width_deg=2.0, min_points=2,
                     range_cluster_m=1.0, min_range_m=1.0, max_range_m=100.0,
                     max_segment_m=3.0):
    """Nearest supported radial group per sector; join only adjacent sectors.

    Points are already height-filtered and flattened in the water frame. Sectors
    cover 360 degrees around the projected LiDAR origin. Unknown sectors remain
    NaN; isolated nearby returns cannot hide a farther supported surface.
    """
    if (not np.isfinite([sector_width_deg, range_cluster_m, min_range_m,
                         max_range_m, max_segment_m]).all()
            or not .1 <= sector_width_deg <= 180 or min_points < 1
            or int(min_points) != min_points or range_cluster_m <= 0
            or min_range_m < 0 or max_range_m <= min_range_m or max_segment_m <= 0):
        raise ValueError('Invalid angular boundary support, range or connection settings')
    origin = np.asarray(origin_xy, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError('Expected a finite XY sector origin')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    relative = points[:, :2] - origin
    ranges = np.linalg.norm(relative, axis=1)
    keep = (ranges >= min_range_m) & (ranges <= max_range_m) & (ranges > 0)
    points, relative, ranges = points[keep], relative[keep], ranges[keep]
    count = int(np.ceil(360 / sector_width_deg))
    angles = (np.arctan2(relative[:, 1], relative[:, 0]) + np.pi) % (2 * np.pi)
    bins = np.minimum((angles * count / (2 * np.pi)).astype(int), count - 1)
    boundary = np.full((count, 3), np.nan)
    for index in np.unique(bins):
        indices = np.flatnonzero(bins == index)
        indices = indices[np.argsort(ranges[indices])]
        depths = ranges[indices]
        stops = np.searchsorted(depths, depths + range_cluster_m, side='right')
        supported = np.flatnonzero(stops - np.arange(len(depths)) >= min_points)
        if len(supported):
            start = supported[0]
            boundary[index] = np.median(points[indices[start:stops[start]]], axis=0)
    # Circular adjacency includes the -pi/+pi seam, but never skips empty sectors.
    following = np.roll(boundary, -1, axis=0)
    connected = np.isfinite(boundary).all(axis=1) & np.isfinite(following).all(axis=1)
    connected &= np.linalg.norm(following - boundary, axis=1) <= max_segment_m
    segments = np.stack((boundary[connected], following[connected]), axis=1)
    return boundary, segments
