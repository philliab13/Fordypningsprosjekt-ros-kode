"""Load the footprint repository without dependencies."""

import importlib
from pathlib import Path
import sys


class PipelineBackend:
    def __init__(self, repo_path, **options):
        root = Path(repo_path).expanduser().resolve()
        if not (root / "free_space/pipeline.py").is_file():
            raise ValueError(f"No footprint pipeline at {root}")
        sys.path.insert(0, str(root))
        self.module = importlib.import_module("free_space.pipeline")
        if Path(self.module.__file__).resolve() != root / "free_space/pipeline.py":
            raise RuntimeError(
                "Another free_space package is loaded; use a separate ROS process"
            )
        self.outputs = importlib.import_module("free_space.outputs")
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
