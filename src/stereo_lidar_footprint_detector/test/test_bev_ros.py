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


@pytest.mark.parametrize('with_scans', [True, False])
def test_real_cloud_mask_and_footprint_messages(monkeypatch, with_scans):
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
        p=dict(scan_count=3, max_skew_s=0.2, fixed_frame='', size_px=400, range_m=20.),
        publisher=SimpleNamespace(publish=output.append),
        rgb_publisher=SimpleNamespace(publish=rgb_output.append))
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
    assert output[0].header == header
    assert output[0].encoding == 'bgr8'
    assert len(output[0].data) == 400 * 400 * 3
    assert len(rgb_output) == 1
    assert rgb_output[0].header == output[0].header
    assert rgb_output[0].encoding == 'bgr8'
    assert len(rgb_output[0].data) == 100 * 100 * 3


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
    references, status = bev_node.BevNode.gnss_overlay(
        harness, 2_000_000_000, header, body_from_camera)
    assert references == []
    assert 'no matching pose' in status
