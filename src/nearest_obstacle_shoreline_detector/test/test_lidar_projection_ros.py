from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import Transform
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from nearest_obstacle_shoreline_detector.lidar_projection_node import LidarProjectionNode


class Capture:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


@pytest.mark.parametrize('missing_frame', [False, True])
def test_cloud_callback_preserves_stamp_and_publishes_water_coordinates(missing_frame):
    fields = [PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(('x', 'y', 'z', 'height_above_water'))]
    header = Header(frame_id='' if missing_frame else 'lidar')
    header.stamp.sec = 12
    cloud = point_cloud2.create_cloud_xyz32(header, [[1., 2., 1.], [3., 4., 2.]])
    transform = Transform()
    transform.rotation.w = 1.
    fake = SimpleNamespace(
        p=dict(water_frame='water', water_level_m=0., min_height_m=0., max_height_m=2.),
        buffer=SimpleNamespace(lookup_transform=lambda *args, **kwargs:
                               SimpleNamespace(transform=transform)),
        projected_pub=Capture(), filtered_pub=Capture(), fields=fields,
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: None))
    LidarProjectionNode.receive_cloud(fake, cloud)
    projected = fake.projected_pub.messages[0]
    assert projected.header.frame_id == 'water'
    assert projected.header.stamp == header.stamp
    assert projected.width == (0 if missing_frame else 1)
    if not missing_frame:
        np.testing.assert_allclose(point_cloud2.read_points_numpy(
            projected, field_names=('x', 'y', 'z', 'height_above_water')), [[1, 2, 0, 1]])
        np.testing.assert_allclose(point_cloud2.read_points_numpy(
            fake.filtered_pub.messages[0], field_names=('x', 'y', 'z')), [[1, 2, 1]])
