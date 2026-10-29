"""LiDAR height projection and nearest supported angular-sector boundary."""
import json
import numpy as np
from geometry_msgs.msg import Point
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformException, TransformListener

from .inputs import transform_matrix
from .lidar_projection import nearest_boundary, project_to_water


class LidarProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_water_projection')
        defaults = dict(lidar_topic="/front_lidar/ouster/points",# or "/back_lidar/ouster/points" Sim ->  #"/lidar_aft/points", #Subscribe from rosbag
                        water_frame='ma2_water_plane_up',
                        water_level_m=0.0, min_height_m=0.0, max_height_m=2.0,
                        sector_width_deg=2.0, min_sector_points=2, range_cluster_m=1.0,
                        min_range_m=1.0, max_range_m=100.0, max_segment_m=3.0)
        self.p = {name: self.declare_parameter(
            name, value, ParameterDescriptor(read_only=True)).value
            for name, value in defaults.items()}
        if not self.p['water_frame'] or not self.p['lidar_topic']:
            raise ValueError('Provide a water_frame and lidar_topic')
        # Validate configuration even before a scan arrives.
        project_to_water(np.empty((0, 3)), np.eye(4), self.p['water_level_m'],
                         self.p['min_height_m'], self.p['max_height_m'])
        self.boundary_settings = dict(
            sector_width_deg=self.p['sector_width_deg'], min_points=self.p['min_sector_points'],
            **{key: self.p[key] for key in
               ('range_cluster_m', 'min_range_m', 'max_range_m', 'max_segment_m')})
        nearest_boundary([], [0, 0], **self.boundary_settings)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.projected_pub = self.create_publisher(PointCloud2, '~/projected_points', 1)
        self.filtered_pub = self.create_publisher(PointCloud2, '~/filtered_points', 1)
        self.boundary_pub = self.create_publisher(PointCloud2, '~/boundary_points', 1)
        self.barrier_pub = self.create_publisher(MarkerArray, '~/barrier', 1)
        self.json_pub = self.create_publisher(String, '~/barrier_json', 1)
        self.fields = [PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
                       for i, name in enumerate(('x', 'y', 'z', 'height_above_water'))]
        self.subscription = self.create_subscription(PointCloud2, self.p['lidar_topic'],
                                                      self.receive_cloud, qos_profile_sensor_data)
        self.get_logger().info(
            f"Projecting LiDAR heights [{self.p['min_height_m']}, {self.p['max_height_m']}) m "
            f"onto z={self.p['water_level_m']} in {self.p['water_frame']}")

    def receive_cloud(self, msg):
        header = Header(stamp=msg.header.stamp, frame_id=self.p['water_frame'])
        try:
            if not msg.header.frame_id:
                raise ValueError('LiDAR message has no frame_id')
            transform = transform_matrix(self.buffer.lookup_transform(
                self.p['water_frame'], msg.header.frame_id, Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.0)).transform)
            points = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            filtered, projected, heights = project_to_water(
                points, transform, self.p['water_level_m'],
                self.p['min_height_m'], self.p['max_height_m'])
            boundary, segments = nearest_boundary(
                projected, transform[:2, 3], **self.boundary_settings)
            status = 'ok' if len(segments) else 'no supported segments'
        except (ValueError, TransformException, RuntimeError, AssertionError) as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            filtered = projected = np.empty((0, 3))
            heights = np.empty(0)
            boundary, segments = np.empty((0, 3)), np.empty((0, 2, 3))
            status = str(exc)
        for publisher, xyz in ((self.filtered_pub, filtered), (self.projected_pub, projected)):
            rows = np.column_stack((xyz, heights)).astype(np.float32)
            publisher.publish(point_cloud2.create_cloud(header, self.fields, rows))
        supported = boundary[np.isfinite(boundary).all(axis=1)]
        self.boundary_pub.publish(point_cloud2.create_cloud_xyz32(header, supported))
        markers = [Marker(header=header, action=Marker.DELETEALL)]
        if len(segments):
            marker = Marker(header=header, ns='lidar_boundary', id=0,
                            type=Marker.LINE_LIST, action=Marker.ADD)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.15
            marker.color.r, marker.color.g, marker.color.a = 1.0, 0.5, 1.0
            marker.points = [Point(x=float(x), y=float(y), z=float(z))
                             for segment in segments for x, y, z in segment]
            markers.append(marker)
        self.barrier_pub.publish(MarkerArray(markers=markers))
        self.json_pub.publish(String(data=json.dumps(dict(
            frame_id=header.frame_id,
            stamp=dict(sec=header.stamp.sec, nanosec=header.stamp.nanosec),
            status=status, segments=segments.tolist()), allow_nan=False)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarProjectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
