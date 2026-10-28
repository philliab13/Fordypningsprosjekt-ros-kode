"""Exercise the packaged pipeline without an external checkout or model inference."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from stereo_lidar_footprint_detector.backend import PipelineBackend


K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])


def detector():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    return SimpleNamespace(detect=lambda image: (
        np.array([[40., 40., 60., 60.]]), [mask], np.array([8]), np.array([.9]), image))


def test_packaged_pipeline_masks_geometry_and_serialization(monkeypatch):
    # Fail on any accidental reintroduction of the external free_space package.
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'free_space' or name.startswith('free_space.'):
            raise AssertionError('External footprint imports are forbidden')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    path_before = list(sys.path)
    backend = PipelineBackend(detector=detector(), max_stereo_depth=20.)
    backend.configure(K, .12, (100, 100))
    result = backend.run(timestamp=123, left_img=np.zeros((100, 100, 3), dtype=np.uint8),
        disparity_img=None, depth_img=np.full((100, 100), 10., dtype=np.float32),
        lidar_uv=np.empty((0, 2)), lidar_xyz=np.empty((0, 3)))
    feature = result.obstacles['features'][0]
    assert feature['properties']['class_id'] == 8
    assert feature['properties']['depth_m'] == pytest.approx(10.)
    np.testing.assert_allclose(feature['geometry']['coordinates'][0],
                               [[-1, 10], [1, 10], [1, 12], [-1, 12], [-1, 10]])
    assert result.boat_mask[50, 50] == 255 and result.boat_mask[0, 0] == 0
    collection = backend.outputs.obstacle_collection(result.obstacles, 123, 'water')
    assert collection['timestamp_ns'] == 123 and collection['frame_id'] == 'water'
    assert collection['units'] == 'm'
    assert sys.path == path_before
    instance = backend.pipeline
    backend.configure(K.copy(), .12, (100, 100))
    assert backend.pipeline is instance
    with pytest.raises(ValueError, match='Calibration changed'):
        backend.configure(K, .13, (100, 100))


def test_weights_default_explicit_and_missing(monkeypatch, tmp_path):
    ament = ModuleType('ament_index_python')
    packages = ModuleType('ament_index_python.packages')
    packages.get_package_share_directory = lambda name: str(tmp_path)
    monkeypatch.setitem(sys.modules, 'ament_index_python', ament)
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', packages)
    weights = tmp_path / 'weights/yolo26n-seg.pt'
    weights.parent.mkdir()
    weights.write_bytes(b'test checkpoint')
    assert PipelineBackend().options['yolo_weights'] == str(weights)
    custom = tmp_path / 'custom.pt'
    custom.write_bytes(b'custom checkpoint')
    assert PipelineBackend(yolo_weights=str(custom)).options['yolo_weights'] == str(custom)
    with pytest.raises(ValueError, match='Missing YOLO weights'):
        PipelineBackend(yolo_weights=str(tmp_path / 'missing.pt'))
    weights.unlink()
    with pytest.raises(ValueError, match='Missing YOLO weights'):
        PipelineBackend()
