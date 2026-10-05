from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stixel_free_space.core import (LatestSlot, compute_stereo, nearest_cloud,
                                     pipeline_pose, project_lidar, stereo_calibration, stixel_rows)


def info():
    return SimpleNamespace(height=720, width=1280, k=[760., 0., 640., 0., 760., 360., 0., 0., 1.],
                           d=[], binning_x=0, binning_y=0, roi=SimpleNamespace(x_offset=0, y_offset=0))


def test_calibration_from_tf_not_mono_projection_baseline():
    transform = np.eye(4)
    transform[0, 3] = 0.2
    k, baseline, shape = stereo_calibration(info(), info(), transform, 128)
    assert baseline == 0.2 and shape == (720, 1280) and k[0, 0] == 760
    with pytest.raises(ValueError, match='divisible'):
        stereo_calibration(info(), info(), transform, 192)
    transform[0, 3] = -0.2
    with pytest.raises(ValueError, match='positive optical x'):
        stereo_calibration(info(), info(), transform, 128)


def test_reject_unrectified_pair():
    transform = np.eye(4)
    transform[0, 3] = 0.2
    distorted = info()
    distorted.d = [0.1, 0., 0., 0., 0.]
    with pytest.raises(ValueError, match='Distorted'):
        stereo_calibration(distorted, info(), transform, 128)
    transform[:3, :3] = Rotation.from_euler('y', 0.1).as_matrix()
    with pytest.raises(ValueError, match='aligned'):
        stereo_calibration(info(), info(), transform, 128)


def test_projection_discards_behind_outside_and_nonfinite_points():
    transform = np.eye(4)
    transform[0, 3] = 1
    k = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    uv, xyz = project_lidar([[0, 0, 10], [0, 0, -10], [100, 0, 1], [np.nan, 1, 2]],
                            transform, k, (100, 100))
    np.testing.assert_allclose(uv, [[60, 50]])
    np.testing.assert_allclose(xyz, [[1, 0, 10]])
    uv, xyz = project_lidar([], transform, k, (100, 100))
    assert uv.shape == (0, 2) and xyz.shape == (0, 3)


def camera_transform(x, y, heading):
    # Optical right, down, forward in an ENU-like world with forward initially +X.
    rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('z', heading).as_matrix() @ rotation
    transform[:3, 3] = [x, y, 1.5]
    return transform


def upstream_planar_transform(pose):
    yaw = Rotation.from_quat(pose[3:]).as_euler('xyz')[2] + np.pi
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, pose[0]], [-s, c, pose[1]], [0., 0., 1.]])


@pytest.mark.parametrize('x,y,heading', [(2., 0., 0.), (0., 2., 0.), (1., 2., 0.3), (0., 0., -1.2)])
def test_adapted_pose_preserves_translation_rotation_and_camera_origin(x, y, heading):
    before = camera_transform(0, 0, 0)
    after = camera_transform(x, y, heading)
    point_old = np.array([10., 2., 1.])  # forward, right, homogeneous
    relative = np.linalg.inv(upstream_planar_transform(pipeline_pose(after))) @ upstream_planar_transform(pipeline_pose(before))
    propagated = relative @ point_old
    # Compare to independent full optical-frame rigid transform of same point.
    point_world = before @ np.array([2., 0., 10., 1.])
    point_new_optical = np.linalg.inv(after) @ point_world
    np.testing.assert_allclose(propagated[:2], point_new_optical[[2, 0]], atol=1e-8)


def test_fixed_point_disparity_and_invalid_depth():
    matcher = SimpleNamespace(compute=lambda a, b: np.array([[160, -16], [0, 320]], dtype=np.int16))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    disparity, depth = compute_stereo(matcher, image, image, 100., 0.2)
    np.testing.assert_allclose(disparity, [[10, 0], [0, 20]])
    np.testing.assert_allclose(depth, [[2, 0], [0, 1]])


def test_pending_frames_are_replaced_and_close_discards_pending():
    slot = LatestSlot()
    slot.put('old')
    slot.put('new')
    assert slot.take() == 'new' and slot.dropped == 1
    slot.put('pending')
    slot.close()
    assert slot.take() is None


def test_nearest_cloud_has_bounded_skew():
    def cloud(ns):
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=ns)))
    a, b = cloud(100_000_000), cloud(190_000_000)
    assert nearest_cloud([a, b], 200_000_000, 0.02) is b
    assert nearest_cloud([a, b], 300_000_000, 0.02) is None
    assert nearest_cloud([], 100, 0.1) is None


def test_invalid_stixels_cannot_be_interpreted_as_origin_or_clearance():
    stixels = SimpleNamespace(
        stixel_validity=np.array([True, False]), height_base_list=np.array([[20, 40], [10, 30]]),
        stixel_fused_depths=np.array([10., 0.]), stixel_fused_depths_var=np.array([0.1, 1e10]),
        dynamic_stixel_list=np.array([False, True]), using_prop_depth=np.array([True, False]),
        stixel_has_measurement=np.array([False, False]))
    rows = stixel_rows(stixels, np.array([[10., 2.], [0., 0.]]))
    np.testing.assert_allclose(rows[0, :3], [2, 0, 10])
    assert np.isnan(rows[1, :3]).all() and rows[1, 8] == 0
    assert rows[0, 10] == 1 and rows[0, 11] == 0
