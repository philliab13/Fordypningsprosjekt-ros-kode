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


