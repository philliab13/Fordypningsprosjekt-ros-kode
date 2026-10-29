"""Check that live message frames supply the detector's LiDAR calibration."""

from types import SimpleNamespace
import threading

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer

from stereo_lidar_footprint_detector.node import FootprintNode


@pytest.mark.parametrize('connected', [True, False])
def test_detector_requires_message_frame_tf_before_fusion(connected):
    header = Header(frame_id='camera')
    header.stamp.sec = 1
    image = Image(header=header, height=10, width=10, encoding='rgb8', step=30,
                  data=bytes(300))
    depth = Image(header=header, height=10, width=10, encoding='32FC1', step=40,
                  data=np.ones((10, 10), dtype=np.float32).tobytes())
    info = CameraInfo(header=header, height=10, width=10)
    info.p = [5., 0., 5., 0., 0., 5., 5., 0., 0., 0., 1., 0.]
    cloud_header = Header(frame_id='front_lidar')
    cloud_header.stamp.sec = 1
    cloud = point_cloud2.create_cloud_xyz32(cloud_header, [[0., 0., 5.]])
    buffer = Buffer()
    if connected:
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'camera', 'front_lidar'
        tf.transform.translation.x = 1.
        tf.transform.rotation.w = 1.
        buffer.set_transform_static(tf, 'test')
    fused, errors = [], []

    def run(**kwargs):
        fused.append(kwargs)
        # Stop after inspecting fusion inputs; no model loading/inference needed.
        raise RuntimeError('fusion inputs captured')

    harness = SimpleNamespace(
        p=dict(baseline_m=.12, lidar_max_skew_s=.5, require_lidar=False,
               fixed_frame='', water_frame=''),
        tf_buffer=buffer, clouds=[cloud], cloud_lock=threading.Lock(),
        backend=SimpleNamespace(configure=lambda *args: None, run=run),
        get_logger=lambda: SimpleNamespace(info=lambda *args, **kwargs: None,
                                           error=errors.append))
    FootprintNode.process(harness, image, depth, info)
    if connected:
        assert len(fused) == 1
        np.testing.assert_allclose(fused[0]['lidar_xyz'], [[1., 0., 5.]])
        np.testing.assert_allclose(fused[0]['lidar_uv'], [[6., 5.]])
        assert 'fusion inputs captured' in errors[0]
    else:
        assert not fused  # require_lidar=False does not hide a broken mounting TF
        assert errors and 'Footprint frame rejected' in errors[0]
