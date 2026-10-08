"""Load the user's original checkout without copying or modifying its algorithm."""
import importlib
from pathlib import Path
import sys

import numpy as np


class PipelineBackend:
    def __init__(self, params):
        self.params = params
        root = Path(params['repo_path']).expanduser().resolve()
        if not (root / 'free_space/pipeline.py').is_file():
            raise ValueError(f'repo_path does not contain free_space/pipeline.py: {root}')
        sys.path.insert(0, str(root))
        try:
            self.module = importlib.import_module('free_space.pipeline')
            self.torch = importlib.import_module('torch')
        except ImportError as exc:
            raise RuntimeError('Missing pipeline dependencies. Follow stixel_free_space/README.md '
                               f'using the ROS Python interpreter: {exc}') from exc
        if Path(self.module.__file__).resolve() != root / 'free_space/pipeline.py':
            raise RuntimeError('A different free_space package is already imported')
        if not self.torch.cuda.is_available():
            raise RuntimeError('The unchanged FastSAM/YOLO pipeline requires CUDA; PyTorch cannot access a GPU')
        self.runtime_info = dict(torch_version=self.torch.__version__,
                                 torch_cuda_runtime=self.torch.version.cuda,
                                 gpu_name=self.torch.cuda.get_device_name(),
                                 upstream_pipeline=str(self.module.__file__))
        self.root = root
        self.pipeline = None
        self.signature = None

    def configure(
        self,
        k,
        baseline,
        shape,
        camera_height,
        t_body_to_cam=None,
        R_body_to_cam=None,
    ):
        if t_body_to_cam is None:
            t_body_to_cam = np.zeros(2)

        if R_body_to_cam is None:
            R_body_to_cam = np.eye(2)
            
        signature = (
            tuple(k.flat),
            float(baseline),
            shape,
            tuple(np.asarray(t_body_to_cam).flat),
            tuple(np.asarray(R_body_to_cam).flat),
        )        
        
        if self.pipeline is not None:
            (
                old_k,
                old_baseline,
                old_shape,
                old_t_body_to_cam,
                old_R_body_to_cam,
            ) = self.signature

            calibration_changed = (
                shape != old_shape
                or not np.allclose(
                    k.flat,
                    old_k,
                    rtol=1e-6,
                    atol=1e-6,
                )
                or not np.isclose(
                    baseline,
                    old_baseline,
                    rtol=1e-6,
                    atol=1e-6,
                )
                or not np.allclose(
                    np.asarray(t_body_to_cam).flat,
                    old_t_body_to_cam,
                    rtol=1e-6,
                    atol=1e-6,
                )
                or not np.allclose(
                    np.asarray(R_body_to_cam).flat,
                    old_R_body_to_cam,
                    rtol=1e-6,
                    atol=1e-6,
                )
            )

            if calibration_changed:
                raise ValueError(
                    'Calibration changed; restart the node to reinitialize the pipeline'
                )
        else:
            p = self.params

            def path(name, default):
                value = Path(p[name]).expanduser() if p[name] else self.root / default
                return str(value.resolve())

            self.pipeline = self.module.FreeSpacePipeline(
                k, baseline, shape, t_body_to_cam=t_body_to_cam, R_body_to_cam=R_body_to_cam,
                fastsam_weights=path('fastsam_weights', 'weights/FastSAM-x.pt'),
                yolo_weights=path('yolo_weights', 'weights/yolo11n-seg.pt'),
                rwps_config=path('rwps_config', 'configs/rwps.json'),
                num_stixels=p['num_stixels'], retina_masks=p['retina_masks'],
                depth_filter=p['depth_filter'], max_prop_frames=p['max_prop_frames'])
            self.signature = signature
            self.stixel_args = dict(
                num_stixels=p['num_stixels'], img_shape=shape,
                cam_params=self.pipeline.stixels.cam_params,
                t_body_to_cam=t_body_to_cam, R_body_to_cam=R_body_to_cam,
                depth_filter=p['depth_filter'], max_prop_frames=p['max_prop_frames'],
                cam_fov=float(np.rad2deg(np.arctan((shape[1] - k[0, 2]) / k[0, 0])
                                        + np.arctan(k[0, 2] / k[0, 0]))))
            self.reset()
        self.pipeline.temporal_filtering.camera_height = -camera_height

    def reset(self):
        if self.pipeline is None:
            return
        self.pipeline.stixels = self.module.Stixels(**self.stixel_args)
        self.pipeline.temporal_filtering = self.module.TemporalFiltering(N=3)
        self.pipeline.rwps.prev_planemodel = None
        self.pipeline.rwps.prev_mask = None
        self.pipeline.prev_timestamp = 0
        self.pipeline.prev_pose = np.array([0., 0., 0., 0., 0., 0., 1.])

    def synchronize(self):
        self.torch.cuda.synchronize()

    def run(self, **kwargs):
        result = self.pipeline.process(**kwargs)
        rwps_succeeded = result.rwps_mask is not None
        plane_height = (float(getattr(self.pipeline.rwps, 'prev_height', np.nan))
                        if rwps_succeeded else float('nan'))
        expected_height = float(self.pipeline.temporal_filtering.camera_height)
        self.last_frame_info = dict(
            rwps_succeeded=rwps_succeeded,
            rwps_plane_height_m=plane_height,
            expected_plane_height_m=expected_height,
            plane_height_deviation_m=(abs(plane_height - expected_height)
                                      if np.isfinite(plane_height) else float('nan')),
        )
        return result, self.pipeline.stixels
