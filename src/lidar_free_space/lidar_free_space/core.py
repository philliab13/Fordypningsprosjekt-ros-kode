"""Pure NumPy polar estimator. Unknown is NaN; no inferred clear ranges."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Config:
    min_range: float = 0.1
    max_range: float = 130.0
    min_obstacle_height: float = 0.15
    max_obstacle_height: float = 3.0
    angular_resolution: float = 1.0  # degrees; must divide 360

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('All estimator parameters must be finite')
        if not 0 <= self.min_range < self.max_range:
            raise ValueError('Require 0 <= min_range < max_range')
        if self.min_obstacle_height >= self.max_obstacle_height:
            raise ValueError('Height limits must be increasing')
        if not 0.05 <= self.angular_resolution <= 10:
            raise ValueError('angular_resolution must be 0.05..10 degrees')
        if not math.isclose(360 / self.angular_resolution, round(360 / self.angular_resolution)):
            raise ValueError('angular_resolution must divide 360')


def estimate(points, origin, water_level, config):
    """Points and lidar origin are in a common gravity-aligned frame."""
    n = round(360 / config.angular_resolution)
    step = 2 * np.pi / n
    angles = -np.pi + (np.arange(n) + 0.5) * step
    ranges = np.full(n, np.inf)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    delta = points[:, :2] - np.asarray(origin)[:2]
    r = np.linalg.norm(delta, axis=1)
    height = points[:, 2] - water_level
    keep = ((r >= config.min_range) & (r <= config.max_range)
            & (height >= config.min_obstacle_height)
            & (height <= config.max_obstacle_height))
    theta = np.arctan2(delta[keep, 1], delta[keep, 0])
    bins = np.floor((theta + np.pi) / step).astype(int) % n
    np.minimum.at(ranges, bins, r[keep])
    ranges[~np.isfinite(ranges)] = np.nan
    return angles, ranges


def contiguous_runs(valid):
    """Circular runs, including seam wrapping, without crossing unknown bins."""
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return [list(range(len(valid)))]
    if not valid.any():
        return []
    start = int(np.flatnonzero(~valid)[0])
    runs, run = [], []
    for offset in range(1, len(valid) + 1):
        i = (start + offset) % len(valid)
        if valid[i]:
            run.append(i)
        elif run:
            runs.append(run)
            run = []
    return runs


def rotation_matrix(q):
    q = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid transform quaternion')
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform_points(points, transform):
    t = transform.translation
    return points @ rotation_matrix(transform.rotation).T + [t.x, t.y, t.z]


def extend_no_returns(ranges, scan_ranges, directions, scan_max, clear_range):
    """Only exclusively +inf scan bins can fill unknown cloud bins.

    directions are unit scan rays rotated into the target frame. Any finite or
    invalid scan sample in a bin vetoes extension. A cloud obstacle always wins.
    """
    result = ranges.copy()
    status = np.isfinite(result).astype(np.float32)
    if not np.isfinite(scan_max) or scan_max <= 0:
        return result, status
    values = np.asarray(scan_ranges)
    directions = np.asarray(directions)
    theta = np.arctan2(directions[:, 1], directions[:, 0])
    bins = np.floor((theta + np.pi) / (2*np.pi/len(ranges))).astype(int) % len(ranges)
    good = np.zeros(len(ranges), dtype=int)
    bad = np.zeros(len(ranges), dtype=int)
    horizontal = np.linalg.norm(directions[:, :2], axis=1)
    supported = np.isposinf(values) & (horizontal * scan_max >= clear_range)
    np.add.at(good, bins[supported], 1)
    np.add.at(bad, bins[~supported], 1)
    fill = ~np.isfinite(result) & (good > 0) & (bad == 0)
    result[fill] = clear_range
    status[fill] = 2  # ASSUMED_CLEAR, never an observed obstacle
    return result, status
