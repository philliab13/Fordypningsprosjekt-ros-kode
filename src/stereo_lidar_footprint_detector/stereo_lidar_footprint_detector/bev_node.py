"""Compare detector footprints with nearby, segmentation-selected LiDAR scans."""

from collections import OrderedDict, deque
from functools import partial
import json
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from stixel_free_space.core import project_lidar, stamp_ns, transform_matrix
from stixel_free_space.ma2_calibration import H_POINTS_ZED_FROM_LIDAR
from tf2_ros import Buffer, TransformListener

from .bev import (closest_scans, local_geometry, mask_selection, render_bev,
                  render_lidar_overlay, scan_color)
from .images import image_array


class BevNode(Node):
    def __init__(self):
        super().__init__('footprint_bev')
        defaults = dict(
            lidar_topic='/lidar_aft/points', footprint_topic='/obstacles/json',
            mask_topic='/obstacles/boat_mask', camera_info_topic='/obstacles/camera_info',
            rgb_topic='/obstacles/source_image',
            rgb_output_topic='/debug/bev_lidar_overlay',
            output_topic='/debug/footprint_bev', scan_count=3, max_skew_s=0.2,
            scan_wait_s=0.2, range_m=60.0, size_px=900, fixed_frame='',
        )
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if (self.p['scan_count'] < 1 or self.p['scan_count'] > 100
                or self.p['size_px'] < 300
                or not all(np.isfinite(self.p[key]) and self.p[key] >= 0
                           for key in ('max_skew_s', 'scan_wait_s'))
                or not np.isfinite(self.p['range_m']) or self.p['range_m'] <= 0):
            raise ValueError('Invalid BEV scan count, time window, size or range')
        self.clouds = deque(maxlen=100)
        self.pending = OrderedDict()
        self.done = deque(maxlen=100)
        self.tf = Buffer(cache_time=Duration(seconds=30))
        self.listener = TransformListener(self.tf, self)
        self.publisher = self.create_publisher(Image, self.p['output_topic'], 10)
        self.rgb_publisher = self.create_publisher(Image, self.p['rgb_output_topic'], 10)
        self.subscriptions_ = [
            self.create_subscription(PointCloud2, self.p['lidar_topic'],
                                     self.receive_cloud, qos_profile_sensor_data),
            self.create_subscription(String, self.p['footprint_topic'],
                                     self.receive_footprints, 10),
            self.create_subscription(Image, self.p['mask_topic'],
                                     partial(self.receive, 'mask'), 10),
            self.create_subscription(CameraInfo, self.p['camera_info_topic'],
                                     partial(self.receive, 'info'), 10),
            self.create_subscription(Image, self.p['rgb_topic'],
                                     partial(self.receive, 'rgb'), 10),
        ]
        self.timer = self.create_timer(0.05, self.flush)
        if not self.p['fixed_frame']:
            self.get_logger().info(
                'BEV uses calibrated MA2 LiDAR extrinsics without ego-motion '
                'compensation. Set fixed_frame to an available odom/map TF frame '
                'to align scans at the image timestamp.')

    def receive_cloud(self, msg):
        self.clouds.append((stamp_ns(msg), msg))

    def receive_footprints(self, msg):
        try:
            collection = json.loads(msg.data)
            timestamp = int(collection['timestamp_ns'])
            if not isinstance(collection['features'], list) or not collection['frame_id']:
                raise ValueError('Missing features or frame_id')
            self.store(timestamp, 'footprints', collection)
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().warning(f'Invalid BEV footprints: {exc}')

    def receive(self, kind, msg):
        self.store(stamp_ns(msg), kind, msg)

    def store(self, timestamp, kind, value):
        if timestamp in self.done:
            return
        if timestamp not in self.pending:
            self.pending[timestamp] = {'arrival': time.monotonic()}
        self.pending[timestamp][kind] = value
        while len(self.pending) > 20:
            self.pending.popitem(last=False)

    def lookup(self, target, source, stamp):
        return transform_matrix(self.tf.lookup_transform(
            target, source, Time.from_msg(stamp),
            timeout=Duration(seconds=0)).transform)

    def flush(self):
        now = time.monotonic()
        for timestamp, job in list(self.pending.items()):
            age = now - job['arrival']
            if not all(key in job for key in ('mask', 'info', 'footprints', 'rgb')):
                if age > 3.0 + self.p['scan_wait_s']:
                    self.pending.pop(timestamp)
                    self.get_logger().warning('BEV dropped an incomplete timestamp match',
                                              throttle_duration_sec=2.0)
                continue
            if age < self.p['scan_wait_s']:
                continue
            try:
                self.publish_frame(timestamp, job)
            except Exception as exc:
                # TF may arrive after the detector messages; retry briefly.
                if age <= 3.0 + self.p['scan_wait_s']:
                    continue
                self.get_logger().warning(f'BEV frame rejected: {exc}',
                                          throttle_duration_sec=2.0)
            else:
                self.done.append(timestamp)
            self.pending.pop(timestamp)

    def publish_frame(self, timestamp, job):
        mask_msg, info = job['mask'], job['info']
        rgb_msg = job['rgb']
        rgb = image_array(rgb_msg)
        if (rgb_msg.header.frame_id != mask_msg.header.frame_id
                or rgb.shape[:2] != (mask_msg.height, mask_msg.width)
                or rgb.ndim != 3):
            raise ValueError('RGB and mask frames/resolutions differ')
        if (mask_msg.encoding != 'mono8' or mask_msg.step < mask_msg.width
                or mask_msg.height <= 0 or mask_msg.width <= 0
                or len(mask_msg.data) < mask_msg.height * mask_msg.step):
            raise ValueError('Expected a valid mono8 boat mask')
        if (info.header.frame_id != mask_msg.header.frame_id
                or (info.height, info.width) != (mask_msg.height, mask_msg.width)):
            raise ValueError('Mask and calibration frames/resolutions differ')
        mask = np.ndarray((mask_msg.height, mask_msg.width), dtype=np.uint8,
                          buffer=mask_msg.data, strides=(mask_msg.step, 1))
        k = np.asarray(info.p, dtype=float).reshape(3, 4)[:, :3]
        if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError('Invalid rectified camera intrinsics')
        optical = mask_msg.header.frame_id
        collection = job['footprints']
        synthetic = optical + '/right_forward_plane'
        camera_to_water = (None if collection['frame_id'] == synthetic else
                           self.lookup(collection['frame_id'], optical,
                                       mask_msg.header.stamp))
        selected = closest_scans(self.clouds, timestamp, self.p['scan_count'],
                                 self.p['max_skew_s'])
        boats, colors, projections, legend = [], [], [], []
        for index, (scan_stamp, cloud) in enumerate(selected):
            points = point_cloud2.read_points_numpy(
                cloud, field_names=('x', 'y', 'z'), skip_nans=True)
            camera_from_lidar = H_POINTS_ZED_FROM_LIDAR
            if self.p['fixed_frame']:
                # The extrinsics are the same calibrated MA2 transform used by
                # the detector; TF supplies camera motion between the two times.
                motion = transform_matrix(self.tf.lookup_transform_full(
                    optical, Time.from_msg(mask_msg.header.stamp),
                    optical, Time.from_msg(cloud.header.stamp), self.p['fixed_frame'],
                    timeout=Duration(seconds=0)).transform)
                camera_from_lidar = motion @ camera_from_lidar
            uv, xyz = project_lidar(points, camera_from_lidar, k, mask.shape)
            boat = mask_selection(uv, xyz, mask)
            color = scan_color(index)
            boats.append(xyz[boat])
            colors.append(np.tile(color, (int(np.count_nonzero(boat)), 1)))
            projections.append(dict(uv=uv, boat=boat, color=color))
            legend.append((f'S{index + 1}: {(scan_stamp - timestamp) / 1e6:+.0f} ms '
                           f'({np.count_nonzero(boat)} boat)', color))
        xyz = np.concatenate(boats) if boats else np.empty((0, 3))
        points, rings = local_geometry(xyz, collection, camera_to_water)
        skews = ','.join(f'{(stamp - timestamp) / 1e6:+.0f}' for stamp, _ in selected)
        mode = ('ego-compensated' if self.p['fixed_frame'] else 'uncompensated')
        plane = 'water z=0' if camera_to_water is not None else 'optical X/Z plane'
        status = f'{len(selected)} scans [{skews}] ms | {mode} | {plane}'
        image = render_bev(
            points, rings, status, self.p['size_px'], self.p['range_m'],
            point_colors=np.concatenate(colors) if colors else np.empty((0, 3)),
            scan_legend=legend)
        rgb_overlay = render_lidar_overlay(rgb, projections, legend, status)
        for publisher, array in [(self.publisher, image), (self.rgb_publisher, rgb_overlay)]:
            msg = Image()
            msg.header = mask_msg.header
            msg.height, msg.width = array.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = msg.width * 3
            msg.data = array.tobytes()
            publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BevNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
