from types import SimpleNamespace
import numpy as np
import pytest

from nearest_obstacle_shoreline_detector.inputs import (
    image_array, nearest_cloud, stamp_ns, subtract_boats, transform_matrix)


def test_rgb_padding_and_big_endian_depth():
    msg = SimpleNamespace(encoding='rgb8', height=1, width=1, step=4,
                          is_bigendian=False, data=bytes([10, 20, 30, 99]))
    np.testing.assert_array_equal(image_array(msg), [[[30, 20, 10]]])
    msg = SimpleNamespace(encoding='32FC1', height=1, width=2, step=12,
                          is_bigendian=True, data=np.array([1.5, 2., 99.], dtype='>f4').tobytes())
    np.testing.assert_array_equal(image_array(msg), [[1.5, 2.]])


def test_cloud_matching_uses_acquisition_time():
    def cloud(n):
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=n, nanosec=0)))
    clouds = [cloud(1), cloud(2)]
    assert nearest_cloud(clouds, 2_000_000_000, .1) is clouds[1]
    assert nearest_cloud(clouds, 3_000_000_000, .1) is None


def test_transform_preserves_translation_and_rotation():
    t = SimpleNamespace(rotation=SimpleNamespace(x=0, y=0, z=0, w=1),
                        translation=SimpleNamespace(x=1, y=2, z=3))
    matrix = transform_matrix(t)
    np.testing.assert_array_equal(matrix[:3, :3], np.eye(3))
    np.testing.assert_array_equal(matrix[:3, 3], [1, 2, 3])


def boat_message():
    return SimpleNamespace(header=SimpleNamespace(frame_id='camera',
                           stamp=SimpleNamespace(sec=10, nanosec=5)),
                           height=2, width=2, step=3, encoding='mono8',
                           data=bytes([0, 1, 99, 255, 0, 99]))


def test_external_boats_subtracted_without_changing_water_input():
    boat = boat_message()
    water = np.full((2, 2), 255, dtype=np.uint8)
    result = subtract_boats(water, boat, boat)
    np.testing.assert_array_equal(result, [[255, 0], [0, 255]])
    np.testing.assert_array_equal(water, 255)


def test_empty_detection_is_a_valid_mask_not_missing_input():
    boat = boat_message()
    boat.data = bytes(6)
    water = np.full((2, 2), 255, dtype=np.uint8)
    np.testing.assert_array_equal(subtract_boats(water, boat, boat), water)


@pytest.mark.parametrize('mismatch', ['stamp', 'frame', 'shape', 'encoding', 'step', 'buffer'])
def test_external_mask_mismatch_is_rejected(mismatch):
    boat, image = boat_message(), boat_message()
    if mismatch == 'stamp':
        boat.header.stamp.nanosec += 1
    elif mismatch == 'frame':
        boat.header.frame_id = 'other_camera'
    elif mismatch == 'shape':
        boat.height = 1
    elif mismatch == 'encoding':
        boat.encoding = 'rgb8'
    elif mismatch == 'step':
        boat.step = 1
    else:
        boat.data = b''
    with pytest.raises(ValueError):
        subtract_boats(np.ones((2, 2), dtype=np.uint8), boat, image)
