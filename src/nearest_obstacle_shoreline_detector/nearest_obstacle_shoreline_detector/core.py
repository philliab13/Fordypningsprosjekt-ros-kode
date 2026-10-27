"""Boundary extraction and measured barrier geometry, independent of ROS."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Settings:
    column_width_px: int = 10
    bottom_margin_px: int = 50
    boundary_band_px: float = 30.0
    min_points: int = 2
    depth_cluster_m: float = 1.0
    max_range_m: float = 100.0
    max_segment_m: float = 3.0
    water_level_m: float = 0.0

    def __post_init__(self):
        for name in ('column_width_px', 'min_points'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.bottom_margin_px < 0:
            raise ValueError('bottom_margin_px must be nonnegative')
        for name in ('boundary_band_px', 'depth_cluster_m', 'max_range_m', 'max_segment_m'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not np.isfinite(self.water_level_m):
            raise ValueError('water_level_m must be finite')


def boundary_rows(mask, bottom_margin_px=50):
    """First non-water pixel scanning upward, with NaN for unsupported columns.

    Like upstream, omit the bottom 50 pixels by default. Require water at the
    scan start: a blocked bottom or an all-water column is not a measured edge.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('Expected a nonempty 2-D water mask')
    end = mask.shape[0] - bottom_margin_px
    if bottom_margin_px < 0 or end <= 0:
        raise ValueError('Bottom margin leaves no search region')
    water = mask[:end] != 0
    nonwater = ~water[::-1]
    valid = water[-1] & np.any(nonwater, axis=0)
    rows = np.full(mask.shape[1], np.nan)
    rows[valid] = end - 1 - np.argmax(nonwater[:, valid], axis=0)
    return rows


def barrier(mask, camera_points, k, water_from_camera, settings=Settings()):
    """Return per-column XYZ (NaN where unknown), line pairs, and pixel boundary.

    Select the nearest depth cluster with enough returns near the image edge.
    Back-project the boundary pixel at its measured optical depth, transform to
    the horizontal output frame, and flatten to that frame's water elevation.
    No interpolation across unknown columns, no extrapolation to the horizon.
    """
    rows = boundary_rows(mask, settings.bottom_margin_px)
    h, w = mask.shape
    k = np.asarray(k, dtype=float).reshape(3, 3)
    transform = np.asarray(water_from_camera, dtype=float)
    if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8
            or transform.shape != (4, 4) or not np.isfinite(transform).all()):
        raise ValueError('Invalid camera calibration or transform')
    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] > 0) & (np.linalg.norm(points, axis=1) <= settings.max_range_m)]
    uvw = points @ k.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    visible = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, points = uv[visible], points[visible]
    pixels = uv[:, 0].astype(int)
    near = np.isfinite(rows[pixels]) & (np.abs(uv[:, 1] - rows[pixels]) <= settings.boundary_band_px)
    uv, points = uv[near], points[near]
    count = (w + settings.column_width_px - 1) // settings.column_width_px
    xyz = np.full((count, 3), np.nan)
    bins = (uv[:, 0] // settings.column_width_px).astype(int)
    for index in range(count):
        depths = np.sort(points[bins == index, 2])
        # A nearby singleton must not hide the next, well-supported surface.
        depth = None
        for start in range(len(depths) - settings.min_points + 1):
            stop = np.searchsorted(depths, depths[start] + settings.depth_cluster_m, side='right')
            if stop - start >= settings.min_points:
                depth = float(np.median(depths[start:stop]))
                break
        if depth is None:
            continue
        left = index * settings.column_width_px
        right = min(w, left + settings.column_width_px)
        valid_columns = np.flatnonzero(np.isfinite(rows[left:right])) + left
        if not len(valid_columns):
            continue
        u = float(np.median(valid_columns))
        v = float(np.median(rows[valid_columns]))
        camera = np.array([(u - k[0, 2]) * depth / k[0, 0],
                           (v - k[1, 2]) * depth / k[1, 1], depth])
        xyz[index] = transform[:3, :3] @ camera + transform[:3, 3]
        xyz[index, 2] = settings.water_level_m
    valid = np.isfinite(xyz).all(axis=1)
    connected = valid[:-1] & valid[1:]
    connected &= np.linalg.norm(np.diff(xyz, axis=0), axis=1) <= settings.max_segment_m
    indices = np.flatnonzero(connected)
    segments = np.stack((xyz[indices], xyz[indices + 1]), axis=1)
    return xyz, segments, rows
