"""Exercise the subscriber's frame processing with real ROS message buffers."""

from collections import OrderedDict, deque
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from stereo_lidar_footprint_detector import bev_node
from stereo_lidar_footprint_detector.gnss import lla_to_ned


@pytest.mark.parametrize('with_scans', [True, False])
@pytest.mark.parametrize('with_shoreline', [True, False])
def test_real_cloud_mask_and_footprint_messages(monkeypatch, with_scans, with_shoreline):
    header = Header(frame_id='camera')
    header.stamp.sec = 1
    cloud = point_cloud2.create_cloud_xyz32(header, [[0, 0, 5], [1, 0, 5]])
    later_header = Header(frame_id='camera')
    later_header.stamp.sec = 1
    later_header.stamp.nanosec = 50_000_000
    later_cloud = point_cloud2.create_cloud_xyz32(later_header, [[0, 0, 6]])
    rgb = Image(header=header, height=100, width=100, encoding='rgb8', step=300)
    rgb.data = np.tile(np.uint8([10, 20, 30]), (100, 100, 1)).tobytes()
    mask = Image(header=header, height=100, width=100, encoding='mono8', step=104)
    # Padded image rows exercise the ROS stride handling.
    data = np.zeros((100, 104), dtype=np.uint8)
    data[50, 50] = 1
    mask.data = data.tobytes()
    info = CameraInfo(header=header, height=100, width=100)
    info.p = [100., 0., 50., 0., 0., 100., 50., 0., 0., 0., 1., 0.]
    collection = {'frame_id': 'camera/right_forward_plane', 'features': [
        {'geometry': {'coordinates': [[[-1, 4], [1, 4], [1, 6], [-1, 4]]]}}]}
    output, geometry, rgb_output, overlays = [], [], [], []
    harness = SimpleNamespace(
        clouds=([(1_000_000_000, cloud), (1_050_000_000, later_cloud)]
                if with_scans else []),
        p=dict(scan_count=3, max_skew_s=0.2, fixed_frame='', shoreline_fixed_frame='piren', size_px=400, range_m=20., gnss_enabled=False),
        publisher=SimpleNamespace(publish=output.append),
        rgb_publisher=SimpleNamespace(publish=rgb_output.append))
    if with_shoreline:
        harness.shorelines = {1_000_000_000: (
            'water_up', np.array([[[1., 2., 3.], [4., 5., 6.]]]))}
        camera_from_water = np.eye(4)
        camera_from_water[:3, 3] = [10., 20., 30.]
        harness.lookup = lambda target, source, stamp: (
            np.eye(4) if target == 'piren' else camera_from_water)
    monkeypatch.setattr(bev_node, 'H_POINTS_ZED_FROM_LIDAR', np.eye(4))

    def render(points, rings, status, size, span, **kwargs):
        geometry.append((points, rings, status, kwargs))
        return np.zeros((size, size, 3), dtype=np.uint8)

    def overlay(image, scans, legend, status):
        overlays.append((image, scans, legend, status))
        return image

    monkeypatch.setattr(bev_node, 'render_bev', render)
    monkeypatch.setattr(bev_node, 'render_lidar_overlay', overlay)
    bev_node.BevNode.publish_frame(harness, 1_000_000_000,
                                 dict(mask=mask, info=info, footprints=collection, rgb=rgb))
    if with_scans:
        np.testing.assert_allclose(geometry[0][0], [[0, 5], [0, 6]])
        assert '2 scans [+0,+50] ms' in geometry[0][2]
        np.testing.assert_array_equal(overlays[0][1][0]['boat'], [True, False])
        np.testing.assert_array_equal(overlays[0][1][1]['boat'], [True])
        for index in range(2):
            np.testing.assert_array_equal(geometry[0][3]['point_colors'][index],
                                          overlays[0][1][index]['color'])
        assert geometry[0][3]['scan_legend'] == overlays[0][2]
    else:
        assert len(geometry[0][0]) == 0
        assert overlays[0][1] == []
    np.testing.assert_array_equal(overlays[0][0][0, 0], [30, 20, 10])
    assert len(output) == 1
    expected = [[[11, 33], [14, 36]]] if with_shoreline else np.empty((0, 2, 2))
    np.testing.assert_allclose(geometry[0][3]['shoreline_segments'], expected)
    assert output[0].header == header
    assert output[0].encoding == 'bgr8'
    assert len(output[0].data) == 400 * 400 * 3
    assert len(rgb_output) == 1
    assert rgb_output[0].header == output[0].header
    assert rgb_output[0].encoding == 'bgr8'
    assert len(rgb_output[0].data) == 100 * 100 * 3
    if with_shoreline:
        harness.shorelines.clear()  # retained geometry no longer needs old TF
        calls = []
        def current_tf(target, source, stamp):
            calls.append((target, source, stamp.sec))
            assert (target, source) == ('camera', 'piren')
            moved = camera_from_water.copy()
            moved[:3, 3] -= [2., 0., 1.]
            return moved
        harness.lookup = current_tf
        mask.header.stamp.sec = 3
        bev_node.BevNode.publish_frame(harness, 3_000_000_000,
            dict(mask=mask, info=info, footprints=collection, rgb=rgb))
        np.testing.assert_allclose(geometry[-1][3]['shoreline_segments'], [[[9, 32], [12, 35]]])
        assert calls == [('camera', 'piren', 3)]
        assert 'age=2.0s (held, ego-compensated)' in geometry[-1][2]


def test_source_timestamps_match_independent_of_arrival_order():
    harness = SimpleNamespace(pending=OrderedDict(), done=deque())
    for timestamp, kind in [(2, 'mask'), (1, 'rgb'), (1, 'footprints'),
                            (2, 'info'), (2, 'footprints'), (2, 'rgb')]:
        bev_node.BevNode.store(harness, timestamp, kind, kind)
    assert set(harness.pending[2]) == {'arrival', 'mask', 'info', 'footprints', 'rgb'}
    assert set(harness.pending[1]) == {'arrival', 'footprints', 'rgb'}


@pytest.mark.parametrize('axes', ['ned', 'enu'])
def test_gnss_camera_offset_axes_and_missing_pose(axes):
    body_from_camera = np.array([[0., 0., 1., 2.],
                                 [1., 0., 0., 1.],
                                 [0., 1., 0., -1.],
                                 [0., 0., 0., 1.]])
    pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    if axes == 'enu':
        # This rotation maps NED body coordinates into an ENU world.
        pose[3:] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
    harness = SimpleNamespace(
        gnss_tracks=[('test receiver', (1, 2, 3), SimpleNamespace(
            at=lambda *args: np.array([12., 4., 0.])))],
        poses=[(1_000_000_000, pose)],
        p=dict(pose_max_gap_s=.5, gnss_pose_axes=axes,
               gnss_body_frame='body', gnss_time_offset_s=0.,
               gnss_max_gap_s=2., piren_lla=[63., 10., 40.]),
        lookup=lambda *args: body_from_camera)
    header = Header(frame_id='camera')
    header.stamp.sec = 1
    references, status = bev_node.BevNode.gnss_overlay(
        harness, 1_000_000_000, header, body_from_camera)
    np.testing.assert_allclose(references[0]['xy'], [3, 10], atol=1e-10)
    assert 'GNSS 1/1' in status
    assert references[0]['label'] == 'test receiver'
    assert references[1]['label'] == 'Ego MA2'
    assert references[1]['ego'] is True
    np.testing.assert_allclose(references[1]['xy'], [-1, -2], atol=1e-10)
    np.testing.assert_allclose(references[1]['position_ned'], [0, 0, 0])
    references, status = bev_node.BevNode.gnss_overlay(
        harness, 2_000_000_000, header, body_from_camera)
    assert references == []
    assert 'no matching pose' in status


@pytest.mark.parametrize('axes', ['ned', 'enu'])
def test_ego_gnss_without_recorded_tracks(axes):
    # Camera is 2 m ahead and 1 m right of the body origin. The vessel's
    # geographic position must not become an offset in this camera-local BEV.
    body_from_camera = np.array([[0., 0., 1., 2.], [1., 0., 0., 1.],
                                 [0., 1., 0., -1.], [0., 0., 0., 1.]])
    position = [100., 200., 3.] if axes == 'ned' else [200., 100., -3.]
    harness = SimpleNamespace(
        gnss_tracks=[], poses=[(1_000_000_000, np.array([*position, 0, 0, 0, 1]))],
        p=dict(gnss_enabled=True, pose_max_gap_s=.5, gnss_pose_axes=axes,
               gnss_body_frame='body', gnss_time_offset_s=0., piren_lla=[63., 10., 40.]),
        lookup=lambda *args: body_from_camera)
    references, status = bev_node.BevNode.gnss_overlay(
        harness, 1_000_000_000, Header(frame_id='camera'), body_from_camera)
    assert len(references) == 1
    np.testing.assert_allclose(references[0]['xy'], [-1, -2], atol=1e-10)
    np.testing.assert_allclose(references[0]['position_ned'], [100, 200, 3])
    origin = harness.p['piren_lla']
    np.testing.assert_allclose(lla_to_ned(references[0]['position_lla'], origin),
                               [100, 200, 3], atol=1e-6)
    camera_ned = [102, 201, 2] if axes == 'ned' else [101, 202, 4]
    np.testing.assert_allclose(lla_to_ned(references[0]['camera_lla'], origin),
                               camera_ned, atol=1e-6)
    assert references[0]['camera_heading_deg'] == pytest.approx(0. if axes == 'ned' else 90.)
    assert '+ ego' in status
    harness.p['gnss_enabled'] = False
    assert bev_node.BevNode.gnss_overlay(harness, 1_000_000_000, None, None) == ([], '')


def test_late_shoreline_redraws_current_frame_and_empty_retains():
    import json
    from std_msgs.msg import String
    rendered, warnings = [], []
    harness = SimpleNamespace(
        shorelines=OrderedDict(), rendered_jobs=OrderedDict([(10, 'frame10'), (20, 'frame20')]),
        publish_frame=lambda stamp, job: rendered.append((stamp, job)),
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: warnings.append(args)))

    def receive(timestamp, segments, status='ok'):
        msg = String(data=json.dumps(dict(frame_id='water',
            stamp=dict(sec=0, nanosec=timestamp), segments=segments, status=status)))
        bev_node.BevNode.receive_shoreline(harness, msg)

    receive(10, [[[1, 2, 0], [3, 4, 0]]])
    assert rendered == [(10, 'frame10'), (20, 'frame20')]
    np.testing.assert_array_equal(harness.shorelines[10][1], [[[1, 2, 0], [3, 4, 0]]])
    receive(11, [])
    assert len(rendered) == 2  # empty update leaves held geometry intact
    receive(10, [])
    assert harness.shorelines[10][1].shape == (1, 2, 3)
    receive(30, [], status='input timeout')
    assert 10 in harness.shorelines
    assert rendered[-1] == (20, 'frame20')
    receive(10, [[[1, 2], [3, 4]]])
    assert warnings  # malformed endpoint dimensions are rejected


def test_world_shoreline_rotation_tf_loss_and_replacement():
    warnings, calls = [], []
    # Optical right is world +X; optical forward is world +Y.
    water_from_camera = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                                 [0., -1., 0., 2.], [0., 0., 0., 1.]])
    pose = water_from_camera.copy()
    fail = set()
    def lookup(target, source, stamp):
        calls.append((target, source, stamp.sec))
        if (target, source) in fail:
            raise RuntimeError('TF unavailable')
        if target == 'piren':
            return np.eye(4)
        return np.linalg.inv(pose)
    harness = SimpleNamespace(
        p=dict(shoreline_fixed_frame='piren'), held_shoreline=None,
        shorelines={1_000_000_000: ('water', np.array([[[2., 10., 0.], [4., 10., 0.]]]))},
        lookup=lookup,
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: warnings.append(args)))
    header = Header(frame_id='camera')
    header.stamp.sec = 1
    xy, _ = bev_node.BevNode.shoreline_overlay(harness, 1_000_000_000, header, water_from_camera)
    np.testing.assert_allclose(xy, [[[2, 10], [4, 10]]])
    anchor = harness.held_shoreline
    # Vessel travels 3 m forward and turns 90 degrees left in the world XY plane.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pose[:3, :3] = rotation @ water_from_camera[:3, :3]
    pose[:3, 3] = [0., 3., 2.]
    header.stamp.sec = 2
    xy, status = bev_node.BevNode.shoreline_overlay(harness, 2_000_000_000, header, water_from_camera)
    np.testing.assert_allclose(xy, [[[7, -2], [7, -4]]], atol=1e-10)
    assert calls.count(('piren', 'water', 1)) == 1
    assert 'age=1.0s' in status
    fail.add(('camera', 'piren'))
    xy, status = bev_node.BevNode.shoreline_overlay(harness, 2_000_000_000, header, water_from_camera)
    assert xy.shape == (0, 2, 2) and 'missing TF' in status
    assert harness.held_shoreline is anchor
    fail.clear()
    harness.shorelines[2_000_000_000] = ('water', np.array([[[0., 13., 0.], [0., 15., 0.]]]))
    fail.add(('piren', 'water'))
    xy, status = bev_node.BevNode.shoreline_overlay(harness, 2_000_000_000, header, water_from_camera)
    np.testing.assert_allclose(xy, [[[7, -2], [7, -4]]], atol=1e-10)
    assert 'update: missing TF' in status
    fail.clear()
    xy, _ = bev_node.BevNode.shoreline_overlay(harness, 2_000_000_000, header, water_from_camera)
    np.testing.assert_allclose(xy, [[[10, 0], [12, 0]]], atol=1e-10)
    assert harness.held_shoreline[0] == 2_000_000_000
    # A cached older BEV must never display geometry from a future measurement.
    harness.shorelines.clear()
    xy, _ = bev_node.BevNode.shoreline_overlay(harness, 1_000_000_000, header, water_from_camera)
    assert xy.shape == (0, 2, 2)
