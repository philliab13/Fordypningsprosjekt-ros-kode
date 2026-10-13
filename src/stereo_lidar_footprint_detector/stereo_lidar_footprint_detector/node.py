"""First-test ROS adapter: registered optical depth + rectified image + LiDAR."""
from collections import deque
import json

import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from stixel_free_space.core import nearest_cloud, project_lidar, stamp_ns, transform_matrix
from .backend import PipelineBackend
from .images import image_array


class FootprintNode(Node):
    def __init__(self):
        super().__init__('stereo_lidar_footprint_detector')
        defaults = dict(repo_path='', baseline_m=0.0, weights='', device='',
                        left_topic='/ma2/zed/left/image',
                        depth_topic='/ma2/zed/depth', camera_info_topic='/ma2/zed/camera_info',
                        lidar_topic='/lidar_aft/points', output_topic='/obstacles/json',
                        water_frame='', max_stereo_depth_m=10.0, lidar_max_skew_s=0.1,
                        require_lidar=False)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if not np.isfinite(self.p['baseline_m']) or self.p['baseline_m'] <= 0:
            raise ValueError('Set baseline_m to the actual stereo baseline in meters')
        self.backend = PipelineBackend(self.p['repo_path'], yolo_weights=self.p['weights'] or None,
                                       device=self.p['device'] or None,
                                       max_stereo_depth=self.p['max_stereo_depth_m'])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.clouds = deque(maxlen=100)
        self.publisher = self.create_publisher(String, self.p['output_topic'], 10)
        self.cloud_sub = self.create_subscription(PointCloud2, self.p['lidar_topic'],
                                                  self.clouds.append, qos_profile_sensor_data)
        self.inputs = [message_filters.Subscriber(self, kind, self.p[name], qos_profile=qos_profile_sensor_data)
                       for kind, name in [(Image, 'left_topic'), (Image, 'depth_topic'),
                                          (CameraInfo, 'camera_info_topic')]]
        self.sync = message_filters.ApproximateTimeSynchronizer(self.inputs, 5, 0.03)
        self.sync.registerCallback(self.process)

    def lookup(self, target, source, stamp):
        return transform_matrix(self.tf_buffer.lookup_transform(
            target, source, Time.from_msg(stamp), timeout=Duration(seconds=0.0)).transform)

    def process(self, image, depth_msg, info):
        try:
            optical_frame = image.header.frame_id
            if not optical_frame or depth_msg.header.frame_id != optical_frame or info.header.frame_id != optical_frame:
                raise ValueError('Image, depth and CameraInfo must share a registered optical frame')
            if depth_msg.encoding != '32FC1':
                raise ValueError('Expected 32FC1 optical Z depth in meters')
            left = image_array(image)
            depth = image_array(depth_msg)
            # P describes the rectified image; raw/distorted streams must be rectified upstream.
            k = np.asarray(info.p, dtype=float).reshape(3, 4)[:, :3]
            if (info.height, info.width) != left.shape[:2]:
                raise ValueError('CameraInfo resolution differs from the image')
            if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
                raise ValueError('Cropped/binned streams require adjusted CameraInfo')
            self.backend.configure(k, self.p['baseline_m'], left.shape[:2])
            cloud = nearest_cloud(self.clouds, stamp_ns(image), self.p['lidar_max_skew_s'])
            uv, xyz = np.empty((0, 2)), np.empty((0, 3))
            if cloud is not None:
                points = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
                transform = self.lookup(optical_frame, cloud.header.frame_id, image.header.stamp)
                uv, xyz = project_lidar(points, transform, k, left.shape[:2])
            elif self.p['require_lidar']:
                raise ValueError('No LiDAR scan within the allowed timestamp skew')
            water_frame = self.p['water_frame']
            transform = self.lookup(water_frame, optical_frame, image.header.stamp) if water_frame else None
            result = self.backend.run(timestamp=stamp_ns(image), left_img=left, disparity_img=None,
                                      depth_img=depth, lidar_uv=uv, lidar_xyz=xyz,
                                      camera_to_water=transform)
            # The default is a synthetic local plane, not a TF optical frame.
            frame_id = water_frame or optical_frame + '/right_forward_plane'
            collection = self.backend.outputs.obstacle_collection(result.obstacles, stamp_ns(image), frame_id)
            self.publisher.publish(String(data=json.dumps(collection, allow_nan=False)))
        except Exception as exc:
            self.get_logger().error(f'Footprint frame rejected: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FootprintNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
