from types import SimpleNamespace
import numpy as np
import pytest

from nearest_obstacle_shoreline_detector.backend import WaterBackend


class Temporal:
    def __init__(self, N):
        self.N = N
        self.reject = False

    def filter(self, mask):
        return mask

    def detect_segmentation_failure(self, mask, plane):
        return self.reject


def backend():
    b = WaterBackend.__new__(WaterBackend)
    b.rwps = SimpleNamespace(
        prev_planemodel=None, prev_mask=None,
        set_camera_params=lambda *args: None,
        segment_water_plane_using_point_cloud=lambda depth: (
            np.ones(depth.shape), np.array([0, 1, 0, -1.6]), True))
    b.fastsam = SimpleNamespace(get_contours_and_water_mask=lambda image, seed: (
        None, None, np.ones(image.shape[:2], dtype=np.uint8)))
    b.temporal_class = Temporal
    b.temporal = Temporal(3)
    b.signature = None
    b.previous_stamp = None
    b.reset_gap_s = 1.0
    return b


def run(b, timestamp=1):
    return b.segment(np.zeros((6, 8, 3), dtype=np.uint8), np.full((6, 8), 10.),
                     np.eye(3), timestamp, 1.6)


def test_segmentation_without_stixels_and_expected_height():
    b = backend()
    np.testing.assert_array_equal(run(b), 255)
    assert b.temporal.camera_height == -1.6


@pytest.mark.parametrize('failure', ['plane', 'no_masks', 'empty_masks', 'rejected'])
def test_segmentation_failures_clear_output_and_history(failure):
    b = backend()
    run(b)
    old_temporal = b.temporal
    if failure == 'plane':
        b.rwps.segment_water_plane_using_point_cloud = lambda depth: (None, None, False)
    elif failure == 'no_masks':
        b.fastsam.get_contours_and_water_mask = lambda *args: (None, None)
    elif failure == 'empty_masks':
        b.fastsam.get_contours_and_water_mask = lambda *args: (None, None, None)
    else:
        b.temporal.reject = True
    np.testing.assert_array_equal(run(b, 2), 0)
    assert b.temporal is not old_temporal


@pytest.mark.parametrize('timestamp', [1, 0, 2_000_000_000])
def test_rewind_repeat_and_gap_reset_history(timestamp):
    b = backend()
    run(b)
    previous = b.temporal
    run(b, timestamp)
    assert b.temporal is not previous


@pytest.fixture
def local_runtime(monkeypatch, tmp_path):
    """Emulate an installed package with no upstream checkout or GPU libraries."""
    import sys
    from types import ModuleType

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module('ament_index_python')
    module('ament_index_python.packages', get_package_share_directory=lambda name: str(tmp_path))
    module('torch', cuda=SimpleNamespace(is_available=lambda: True))
    module('nearest_obstacle_shoreline_detector.segmentation.rwps',
           RWPS=lambda path: SimpleNamespace(config_file=path))
    module('nearest_obstacle_shoreline_detector.segmentation.fastsam',
           FastSAMSeg=lambda **kwargs: SimpleNamespace(**kwargs))
    module('nearest_obstacle_shoreline_detector.segmentation.temporal_filtering',
           TemporalFiltering=Temporal)
    for relative in ('config/rwps.json', 'weights/FastSAM-x.pt'):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'test asset')
    return tmp_path


@pytest.mark.parametrize('override', [False, True])
def test_initialization_uses_package_assets_or_explicit_paths(local_runtime, tmp_path, override):
    import sys
    path_before = list(sys.path)
    params = dict(rwps_config='', fastsam_weights='', retina_masks=False, reset_gap_s=1.)
    config = local_runtime / 'config/rwps.json'
    weights = local_runtime / 'weights/FastSAM-x.pt'
    if override:
        config = tmp_path / 'custom.json'
        weights = tmp_path / 'custom.pt'
        config.write_text('{}')
        weights.write_bytes(b'custom')
        params.update(rwps_config=str(config), fastsam_weights=str(weights))
    b = WaterBackend(params)
    assert b.rwps.config_file == str(config)
    assert b.fastsam.model_path == str(weights)
    assert b.fastsam.retina_masks is False
    assert sys.path == path_before


def test_missing_weights_fail_before_model_can_download(local_runtime):
    (local_runtime / 'weights/FastSAM-x.pt').unlink()
    with pytest.raises(ValueError, match='Missing fastsam_weights'):
        WaterBackend(dict(rwps_config='', fastsam_weights='', retina_masks=False, reset_gap_s=1.))
