"""Exercise mounting direction, chained TF and motion compensation with tf2."""

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException

from stereo_lidar_footprint_detector.transforms import camera_from_lidar


def transform(parent, child, xyz, seconds=0, yaw=0.):
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = parent, child
    msg.header.stamp.sec = seconds
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, xyz)
    msg.transform.rotation.z = float(np.sin(yaw / 2))
    msg.transform.rotation.w = float(np.cos(yaw / 2))
    return msg


def header(frame, seconds):
    msg = Header(frame_id=frame)
    msg.stamp.sec = seconds
    return msg


def mounting_buffer():
    buffer = Buffer()
    buffer.set_transform_static(transform('vessel', 'lidar', [2, 1, 0]), 'test')
    buffer.set_transform_static(transform('vessel', 'camera', [1, 0, 0], yaw=np.pi / 2), 'test')
    return buffer


def test_chained_mounting_maps_lidar_points_into_camera_axes():
    result = camera_from_lidar(mounting_buffer(), header('camera', 2), header('lidar', 1))
    # LiDAR origin is [2,1,0] in vessel axes. Camera is at [1,0,0],
    # rotated +90 degrees: inverse camera rotation maps [1,1,0] to [1,-1,0].
    np.testing.assert_allclose(result @ [0, 0, 0, 1], [1, -1, 0, 1], atol=1e-10)
    np.testing.assert_allclose(result @ [1, 0, 0, 1], [1, -2, 0, 1], atol=1e-10)


def test_fixed_frame_compensates_motion_between_acquisition_times():
    buffer = mounting_buffer()
    buffer.set_transform(transform('world', 'vessel', [0, 0, 0], seconds=1), 'test')
    buffer.set_transform(transform('world', 'vessel', [3, 0, 0], seconds=2), 'test')
    result = camera_from_lidar(buffer, header('camera', 2), header('lidar', 1), 'world')
    np.testing.assert_allclose(result @ [0, 0, 0, 1], [1, 2, 0, 1], atol=1e-10)


def test_missing_or_stale_tf_does_not_fall_back_to_ma2():
    with pytest.raises(TransformException):
        camera_from_lidar(Buffer(), header('camera', 2), header('lidar', 1))
    buffer = mounting_buffer()
    buffer.set_transform(transform('world', 'vessel', [0, 0, 0], seconds=1), 'test')
    with pytest.raises(TransformException):
        camera_from_lidar(buffer, header('camera', 2), header('lidar', 1), 'world')


@pytest.mark.parametrize('camera,lidar', [('', 'lidar'), ('camera', '')])
def test_empty_sensor_frame_is_rejected(camera, lidar):
    with pytest.raises(ValueError, match='frame_id'):
        camera_from_lidar(Buffer(), header(camera, 2), header(lidar, 1))
