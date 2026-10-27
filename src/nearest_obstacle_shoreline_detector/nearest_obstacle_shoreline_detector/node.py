"""Standalone water segmentation and LiDAR shoreline barriers."""
from collections import deque
from dataclasses import fields
import json
import time
from pathlib import Path
import queue
import threading

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .core import Settings, barrier
from .inputs import image_array, nearest_cloud, stamp_ns, subtract_boats, transform_matrix


def mask_array(msg):
    if msg.encoding not in ('mono8', '8UC1'):
        raise ValueError('Water mask must use mono8 or 8UC1 encoding')
    if (msg.height <= 0 or msg.width <= 0 or msg.step < msg.width
            or len(msg.data) < msg.step * msg.height):
        raise ValueError('Invalid mask dimensions, stride or buffer')
    return np.ndarray((msg.height, msg.width), dtype=np.uint8,
                      buffer=msg.data, strides=(msg.step, 1))


class ShorelineNode(Node):
    def __init__(self):
        super().__init__('nearest_obstacle_shoreline_detector')
        defaults = dict(
            input_mode='standalone',
            repo_path=str(Path.home() / 'Documents/NTNU/free-space-estimation'),
            left_topic='/obstacles/source_image', depth_topic='/ma2/zed/depth',
            sync_slop_s=0.03, camera_height_m=0.0, reset_gap_s=1.0,
            fastsam_weights='', rwps_config='',
            boat_mask_topic='/obstacles/boat_mask', sync_queue_size=20,
            remove_boats=True, retina_masks=True,
            water_mask_topic='/stixel_free_space/validated_water_mask',
            camera_info_topic='/ma2/zed/camera_info', lidar_topic='/lidar_aft/points',
            water_frame='ma2_water_plane', fixed_frame='map',
            lidar_max_skew_s=0.10, cloud_buffer_size=300,
            input_timeout_s=5.0, marker_width_m=0.15)
        defaults.update({f.name: getattr(Settings(), f.name) for f in fields(Settings)})
        self.p = {name: self.declare_parameter(
            name, value, ParameterDescriptor(read_only=True)).value
            for name, value in defaults.items()}
        self.settings = Settings(**{f.name: self.p[f.name] for f in fields(Settings)})
        for name in ('lidar_max_skew_s', 'input_timeout_s', 'marker_width_m',
                     'sync_slop_s', 'reset_gap_s'):
            if not np.isfinite(self.p[name]) or self.p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.p['cloud_buffer_size'] < 1 or not self.p['water_frame'] or not self.p['fixed_frame']:
            raise ValueError('Provide frame names and a positive cloud_buffer_size')
        if self.p['sync_queue_size'] < 1:
            raise ValueError('sync_queue_size must be positive')
        if self.p['input_mode'] not in ('standalone', 'water_mask'):
            raise ValueError('input_mode must be standalone or water_mask')
        if not np.isfinite(self.p['camera_height_m']) or self.p['camera_height_m'] < 0:
            raise ValueError('camera_height_m must be nonnegative and finite')
        self.clouds = deque(maxlen=self.p['cloud_buffer_size'])
        self.cloud_lock = threading.Lock()
        self.output_lock = threading.RLock()
        self.pending = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = None
        self.info = None
        self.last_success = None
        self.active = False
        self.buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.listener = TransformListener(self.buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, '~/barrier', 1)
        self.json_pub = self.create_publisher(String, '~/barrier_json', 1)
        self.boundary_pub = self.create_publisher(Image, '~/boundary_mask', 1)
        self.water_pub = self.create_publisher(Image, '~/water_mask', 1)
        self.info_sub = self.create_subscription(CameraInfo, self.p['camera_info_topic'],
                                                 self.receive_info, qos_profile_sensor_data)
        self.cloud_sub = self.create_subscription(PointCloud2, self.p['lidar_topic'],
                                                  self.receive_cloud, qos_profile_sensor_data)
        if self.p['input_mode'] == 'water_mask':
            self.mask_sub = self.create_subscription(Image, self.p['water_mask_topic'],
                                                     self.receive_mask, qos_profile_sensor_data)
        else:
            topics = ['left_topic', 'depth_topic']
            if self.p['remove_boats']:
                topics.append('boat_mask_topic')
            self.image_subs = [Subscriber(self, Image, self.p[name],
                                          qos_profile=qos_profile_sensor_data)
                               for name in topics]
            self.sync = ApproximateTimeSynchronizer(self.image_subs, self.p['sync_queue_size'],
                                                     self.p['sync_slop_s'])
            self.sync.registerCallback(self.receive_images)
            self.worker = threading.Thread(target=self.work, name='shoreline-segmentation', daemon=True)
            self.worker.start()
        self.timer = self.create_timer(0.25, self.expire)
        self.get_logger().info(f"Shoreline node started in {self.p['input_mode']} mode")

    def receive_cloud(self, msg):
        with self.cloud_lock:
            self.clouds.append(msg)

    def receive_images(self, left, depth, boat_mask=None):
        # Bound latency: keep one pending pair while inference runs.
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass
        self.pending.put_nowait((left, depth, self.info, boat_mask))

    def work(self):
        from .backend import WaterBackend
        backend = None
        while not self.stop_event.is_set():
            try:
                left_msg, depth_msg, info, boat_mask = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.validate_info(left_msg, info)
                if (depth_msg.header.frame_id != left_msg.header.frame_id
                        or depth_msg.encoding != '32FC1'):
                    raise ValueError('Depth must be registered 32FC1 metres in the image optical frame')
                left, depth = image_array(left_msg), image_array(depth_msg)
                if left.shape[:2] != depth.shape:
                    raise ValueError('Image and registered depth dimensions differ')
                if left_msg.encoding not in ('bgr8', 'rgb8'):
                    raise ValueError('Left image must be bgr8 or rgb8')
                height = self.p['camera_height_m']
                if height == 0:
                    water_from_camera = transform_matrix(self.buffer.lookup_transform(
                        self.p['water_frame'], left_msg.header.frame_id,
                        Time.from_msg(left_msg.header.stamp), timeout=Duration(seconds=0.0)).transform)
                    height = water_from_camera[2, 3] - self.p['water_level_m']
                if height <= 0:
                    raise ValueError('Camera must be above the configured water plane')
                if backend is None:
                    backend = WaterBackend(self.p)
                mask = backend.segment(left, depth, np.asarray(info.k).reshape(3, 3),
                                       stamp_ns(left_msg), height)
                if self.p['remove_boats']:
                    if boat_mask is None:
                        raise ValueError('Waiting for matching footprint boat mask')
                    mask = subtract_boats(mask, boat_mask, left_msg)
                if self.stop_event.is_set():
                    break
                msg = Image(header=left_msg.header, height=mask.shape[0], width=mask.shape[1],
                            encoding='mono8', step=mask.shape[1], data=mask.tobytes())
                self.water_pub.publish(msg)
                self.receive_mask(msg, info)
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                if backend is not None:
                    backend.reset()
                empty = np.zeros((left_msg.height, left_msg.width), dtype=np.uint8)
                self.water_pub.publish(Image(header=left_msg.header, height=left_msg.height,
                                             width=left_msg.width, encoding='mono8',
                                             step=left_msg.width, data=empty.tobytes()))
                self.publish_boundary(left_msg, np.full(left_msg.width, np.nan))
                self.publish(Header(stamp=left_msg.header.stamp, frame_id=self.p['water_frame']),
                             [], f'segmentation error: {exc}')
                self.get_logger().error(str(exc), throttle_duration_sec=5.0)

    @staticmethod
    def validate_info(msg, info):
        if info is None:
            raise ValueError('Waiting for CameraInfo')
        if (not msg.header.frame_id or info.header.frame_id != msg.header.frame_id
                or (info.height, info.width) != (msg.height, msg.width)):
            raise ValueError('Image and CameraInfo must share dimensions and optical frame')
        k = np.asarray(info.k).reshape(3, 3)
        if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8
                or not np.isfinite(info.d).all() or np.any(np.abs(info.d) > 1e-8)
                or info.binning_x > 1 or info.binning_y > 1
                or info.roi.x_offset or info.roi.y_offset):
            raise ValueError('Provide undistorted, unbinned pinhole CameraInfo matching the image')

    def receive_info(self, msg):
        self.info = msg

    def receive_mask(self, msg, info=None):
        header = Header(stamp=msg.header.stamp, frame_id=self.p['water_frame'])
        try:
            mask = mask_array(msg)
            if not np.any(mask):
                self.publish(header, [], 'empty or rejected water mask')
                self.publish_boundary(msg, np.full(msg.width, np.nan))
                return
            info = info if info is not None else self.info
            self.validate_info(msg, info)
            with self.cloud_lock:
                cloud = nearest_cloud(tuple(self.clouds), stamp_ns(msg), self.p['lidar_max_skew_s'])
            if cloud is None or not cloud.header.frame_id:
                raise ValueError('No timestamp-matched LiDAR scan in buffer')
            stamp = Time.from_msg(msg.header.stamp)
            camera_from_lidar = transform_matrix(self.buffer.lookup_transform_full(
                msg.header.frame_id, stamp, cloud.header.frame_id,
                Time.from_msg(cloud.header.stamp), self.p['fixed_frame'],
                timeout=Duration(seconds=0.0)).transform)
            water_from_camera = transform_matrix(self.buffer.lookup_transform(
                self.p['water_frame'], msg.header.frame_id, stamp,
                timeout=Duration(seconds=0.0)).transform)
            raw = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
            raw = np.asarray(raw).reshape(-1, 3)
            camera_points = raw @ camera_from_lidar[:3, :3].T + camera_from_lidar[:3, 3]
            _, segments, rows = barrier(mask, camera_points, np.asarray(info.k).reshape(3, 3),
                                        water_from_camera, self.settings)
            self.publish(header, segments.tolist(), 'ok' if len(segments) else 'no supported segments')
            self.publish_boundary(msg, rows)
            self.last_success = time.monotonic()
        except (ValueError, TransformException, RuntimeError) as exc:
            self.publish(header, [], str(exc))
            self.publish_boundary(msg, np.full(msg.width, np.nan))
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)

    def publish_boundary(self, source, rows):
        image = np.zeros((source.height, source.width), dtype=np.uint8)
        columns = np.flatnonzero(np.isfinite(rows))
        image[rows[columns].astype(int), columns] = 255
        self.boundary_pub.publish(Image(header=source.header, height=source.height, width=source.width,
                                       encoding='mono8', step=source.width, data=image.tobytes()))

    def publish(self, header, segments, status):
        with self.output_lock:
            self._publish(header, segments, status)

    def _publish(self, header, segments, status):
        markers = [Marker(header=header, action=Marker.DELETEALL)]
        if len(segments):
            marker = Marker(header=header, ns='shoreline', id=0,
                            type=Marker.LINE_LIST, action=Marker.ADD)
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.p['marker_width_m']
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.5, 0.0, 1.0
            marker.lifetime = Duration(seconds=self.p['input_timeout_s']).to_msg()
            marker.points = [Point(x=float(x), y=float(y), z=float(z))
                             for segment in segments for x, y, z in segment]
            markers.append(marker)
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.json_pub.publish(String(data=json.dumps(dict(
            frame_id=header.frame_id,
            stamp=dict(sec=header.stamp.sec, nanosec=header.stamp.nanosec),
            status=status, segments=segments), allow_nan=False)))
        self.active = bool(len(segments))
        if self.active:
            self.last_success = time.monotonic()

    def expire(self):
        with self.output_lock:
            if (self.active and self.last_success is not None
                    and time.monotonic() - self.last_success > self.p['input_timeout_s']):
                self.publish(Header(stamp=self.get_clock().now().to_msg(), frame_id=self.p['water_frame']),
                             [], 'input timeout')

    def destroy_node(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ShorelineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
