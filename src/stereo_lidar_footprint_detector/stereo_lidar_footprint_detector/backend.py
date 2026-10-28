"""Adapter for the footprint pipeline bundled in this ROS package."""
from pathlib import Path

from .detection import outputs, pipeline


class PipelineBackend:
    def __init__(self, **options):
        self.module = pipeline
        self.outputs = outputs
        if options.get('detector') is None:
            weights = options.get('yolo_weights')
            if weights:
                path = Path(weights).expanduser().resolve()
            else:
                from ament_index_python.packages import get_package_share_directory
                path = (Path(get_package_share_directory('stereo_lidar_footprint_detector'))
                        / 'weights/yolo26n-seg.pt')
            if not path.is_file():
                raise ValueError(f'Missing YOLO weights: {path}. Set the weights parameter '
                                 'to an existing checkpoint; weights are not stored in Git.')
            options['yolo_weights'] = str(path)
        self.options = options
        self.pipeline = None

    def configure(self, k, baseline, shape):
        import numpy as np

        if self.pipeline is None:
            self.pipeline = self.module.FreeSpacePipeline(
                k, baseline, shape, **self.options
            )
        elif (
            shape != (self.pipeline.height, self.pipeline.width)
            or not np.allclose(k, self.pipeline.K)
            or not np.isclose(baseline, self.pipeline.baseline)
        ):
            raise ValueError("Calibration changed; restart the footprint node")

    def run(self, **kwargs):
        return self.pipeline.process(**kwargs)
