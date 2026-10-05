from types import SimpleNamespace

import numpy as np
import pytest

from stixel_free_space.backend import PipelineBackend


def test_tf_roundoff_does_not_trigger_calibration_change():
    backend = PipelineBackend.__new__(PipelineBackend)
    k = np.array([[760., 0., 640.], [0., 760., 360.], [0., 0., 1.]])
    backend.signature = tuple(k.flat), 0.2, (720, 1280)
    backend.pipeline = SimpleNamespace(temporal_filtering=SimpleNamespace(camera_height=-1.5))
    backend.configure(k, 0.20000000000015916, (720, 1280), 1.4)
    assert backend.pipeline.temporal_filtering.camera_height == -1.4
    with pytest.raises(ValueError, match='Calibration changed'):
        backend.configure(k, 0.25, (720, 1280), 1.4)


def test_run_exposes_rwps_plane_diagnostics():
    backend = PipelineBackend.__new__(PipelineBackend)
    result = SimpleNamespace(rwps_mask=np.ones((2, 2), dtype=np.uint8))
    backend.pipeline = SimpleNamespace(
        process=lambda **kwargs: result,
        stixels=object(),
        rwps=SimpleNamespace(prev_height=-1.35),
        temporal_filtering=SimpleNamespace(camera_height=-1.4),
    )
    returned, _ = backend.run()
    assert returned is result
    assert backend.last_frame_info['rwps_succeeded'] is True
    assert backend.last_frame_info['plane_height_deviation_m'] == pytest.approx(0.05)
