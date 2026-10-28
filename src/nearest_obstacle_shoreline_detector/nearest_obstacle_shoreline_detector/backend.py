"""Water segmentation using implementations bundled with this ROS package."""
from pathlib import Path
from time import perf_counter

import numpy as np


class WaterBackend:
    def __init__(self, params):
        from ament_index_python.packages import get_package_share_directory
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('The FastSAM implementation requires CUDA')

        root = Path(get_package_share_directory('nearest_obstacle_shoreline_detector'))

        def asset(name, default):
            path = Path(params[name]).expanduser() if params[name] else root / default
            if not path.is_file():
                raise ValueError(f'Missing {name}: {path}')
            return str(path)

        config = asset('rwps_config', 'config/rwps.json')
        weights = asset('fastsam_weights', 'weights/FastSAM-x.pt')
        from .segmentation.rwps import RWPS
        from .segmentation.fastsam import FastSAMSeg
        from .segmentation.temporal_filtering import TemporalFiltering

        self.rwps = RWPS(config)
        self.fastsam = FastSAMSeg(
            model_path=weights,
            retina_masks=params['retina_masks'])
        self.temporal_class = TemporalFiltering
        self.reset_gap_s = params['reset_gap_s']
        self.signature = None
        self.previous_stamp = None
        self.temporal = self.temporal_class(N=3)

    def reset(self):
        self.rwps.prev_planemodel = None
        self.rwps.prev_mask = None
        self.temporal = self.temporal_class(N=3)

    def segment(self, image, depth, k, timestamp_ns, camera_height):
        self.last_timings_ms = {}
        self.last_status = 'error'
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
        started = perf_counter()
        seed, plane, succeeded = self.rwps.segment_water_plane_using_point_cloud(depth)
        self.last_timings_ms['rwps'] = (perf_counter() - started) * 1000
        # Do not infer a water surface from unseeded contours after plane failure.
        if not succeeded or seed is None:
            self.last_status = 'rwps rejected'
            self.reset()
            return np.zeros(image.shape[:2], dtype=np.uint8)
        started = perf_counter()
        result = self.fastsam.get_contours_and_water_mask(image, seed)
        self.last_timings_ms['fastsam'] = (perf_counter() - started) * 1000
        # Upstream returns two rather than three values when there are no masks.
        if len(result) != 3 or result[2] is None:
            self.last_status = 'no FastSAM masks'
            self.reset()
            return np.zeros(image.shape[:2], dtype=np.uint8)
        started = perf_counter()
        mask = self.temporal.filter((result[2] != 0).astype(np.uint8))
        rejected = self.temporal.detect_segmentation_failure(mask, plane)
        self.last_timings_ms['temporal'] = (perf_counter() - started) * 1000
        if rejected:
            self.last_status = 'temporal rejected'
            self.reset()
            return np.zeros(image.shape[:2], dtype=np.uint8)
        self.last_status = 'accepted'
        return (mask != 0).astype(np.uint8) * 255
