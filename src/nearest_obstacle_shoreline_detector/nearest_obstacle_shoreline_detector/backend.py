"""Segmentation only: reuse upstream implementations without importing stixels."""
import importlib
from pathlib import Path
import sys

import numpy as np


class WaterBackend:
    def __init__(self, params):
        root = Path(params['repo_path']).expanduser().resolve()
        if not (root / 'free_space/rwps.py').is_file():
            raise ValueError('repo_path must point to the free-space-estimation repository')
        sys.path.insert(0, str(root))
        rwps_module = importlib.import_module('free_space.rwps')
        if Path(rwps_module.__file__).resolve() != root / 'free_space/rwps.py':
            raise RuntimeError('A different free_space package is already imported')
        torch = importlib.import_module('torch')
        if not torch.cuda.is_available():
            raise RuntimeError('The upstream FastSAM implementation requires CUDA')

        def asset(name, default):
            path = Path(params[name]).expanduser() if params[name] else root / default
            if not path.is_file():
                raise ValueError(f'Missing {name}: {path}')
            return str(path)

        self.rwps = rwps_module.RWPS(asset('rwps_config', 'configs/rwps.json'))
        self.fastsam = importlib.import_module('free_space.fastsam').FastSAMSeg(
            model_path=asset('fastsam_weights', 'weights/FastSAM-x.pt'),
            retina_masks=params['retina_masks'])
        self.temporal_class = importlib.import_module('free_space.temporal_filtering').TemporalFiltering
        self.reset_gap_s = params['reset_gap_s']
        self.signature = None
        self.previous_stamp = None
        self.temporal = self.temporal_class(N=3)

    def reset(self):
        self.rwps.prev_planemodel = None
        self.rwps.prev_mask = None
        self.temporal = self.temporal_class(N=3)

    def segment(self, image, depth, k, timestamp_ns, camera_height):
        signature = (image.shape[:2], tuple(np.asarray(k).flat))
        if (signature != self.signature or self.previous_stamp is None
                or timestamp_ns <= self.previous_stamp
                or timestamp_ns - self.previous_stamp > self.reset_gap_s * 1e9):
            self.reset()
        self.signature, self.previous_stamp = signature, timestamp_ns
        self.temporal.camera_height = -float(camera_height)
        cam = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])
        self.rwps.set_camera_params(cam, k @ np.hstack((np.eye(3), np.zeros((3, 1)))))
        depth = np.array(depth, dtype=np.float32, copy=True)
        depth[(~np.isfinite(depth)) | (depth <= 0) | (depth > 100)] = 0
        seed, plane, succeeded = self.rwps.segment_water_plane_using_point_cloud(depth)
        # Do not infer a water surface from unseeded contours after plane failure.
        if not succeeded or seed is None:
            self.reset()
            return np.zeros(image.shape[:2], dtype=np.uint8)
        result = self.fastsam.get_contours_and_water_mask(image, seed)
        # Upstream returns two rather than three values when there are no masks.
        if len(result) != 3 or result[2] is None:
            self.reset()
            return np.zeros(image.shape[:2], dtype=np.uint8)
        mask = self.temporal.filter((result[2] != 0).astype(np.uint8))
        if self.temporal.detect_segmentation_failure(mask, plane):
            self.reset()
            return np.zeros(image.shape[:2], dtype=np.uint8)
        return (mask != 0).astype(np.uint8) * 255
