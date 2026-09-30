import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from lidar_free_space.node import FreeSpaceNode


class Capture:
    def publish(self, msg):
        self.msg = msg


def test_ros_adapter_transforms_masks_and_clears_polygons():
    rclpy.init()
    node = FreeSpaceNode()
    try:
        tf = TransformStamped()
        tf.header.frame_id = 'gymkhana_task'
        tf.child_frame_id = 'wamv/wamv/base_link'
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        node.buffer.set_transform_static(tf, 'test')
        base = TransformStamped()
        base.header.frame_id = 'wamv/wamv/base_link'
        base.child_frame_id = 'test_lidar'
        base.transform.translation.z = 2.0
        base.transform.rotation.w = 1.0
        node.buffer.set_transform_static(base, 'test')
        boundary, markers = Capture(), Capture()
        node.boundary_pub = boundary
        node.marker_pub = markers
        msg = point_cloud2.create_cloud_xyz32(Header(frame_id='test_lidar'),
                                             [[10, 0, -1], [10, 0.2, -1], [1, 0, -1]])
        node.on_cloud(msg)
        data = point_cloud2.read_points(boundary.msg)
        assert boundary.msg.header.frame_id == 'gymkhana_task'
        assert np.count_nonzero(data['status']) == 2
        assert np.nanmin(data['range']) == 10
        assert sum(m.type == Marker.TRIANGLE_LIST for m in markers.msg.markers) == 1
        msg = point_cloud2.create_cloud_xyz32(Header(frame_id='test_lidar'), [])
        node.on_cloud(msg)
        assert markers.msg.markers[0].action == Marker.DELETEALL
        assert not any(m.type == Marker.TRIANGLE_LIST for m in markers.msg.markers)
    finally:
        node.destroy_node()
        rclpy.shutdown()
