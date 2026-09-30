"""ROS adapter for per-cloud polar boundaries and disconnected polygon markers."""
import copy
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException
from .core import Config, estimate, contiguous_runs, transform_points, rotation_matrix, extend_no_returns


class FreeSpaceNode(Node):
    def __init__(self):
        super().__init__('lidar_free_space')
        defaults = dict(input_topic='/wamv/sensors/lidars/lidar_wamv_sensor/points',
                        target_frame='gymkhana_task', base_frame='wamv/wamv/base_link',
                        assumed_clear_range=100.0, enable_assumed_clear=True,
                        scan_topic='/wamv/sensors/lidars/lidar_wamv_sensor/scan',
                        water_level=0.0, self_mask_min=[-3.2, -1.8, -1.0],
                        self_mask_max=[3.2, 1.8, 2.5], marker_lifetime=0.5)
        defaults.update(vars(Config()))
        self.p = {k: self.declare_parameter(k, v, ParameterDescriptor(read_only=True)).value
                  for k, v in defaults.items()}
        if not 0 < self.p['assumed_clear_range'] <= self.p['max_range']:
            raise ValueError('assumed_clear_range must be positive and <= max_range')
        self.scan = None
        self.scan_sub = self.create_subscription(LaserScan, self.p['scan_topic'], self.on_scan, qos_profile_sensor_data)
        self.config = Config(**{k: self.p[k] for k in vars(Config())})
        self.lower, self.upper = (np.asarray(self.p[k]) for k in ('self_mask_min', 'self_mask_max'))
        if (self.lower.shape != (3,) or self.upper.shape != (3,)
                or not np.isfinite([self.lower, self.upper]).all()
                or not (self.lower < self.upper).all()):
            raise ValueError('Self mask must have finite, increasing three-dimensional bounds')
        if not np.isfinite(self.p['water_level']) or not 0 < self.p['marker_lifetime'] < 60:
            raise ValueError('Invalid water level or marker lifetime')
        if not self.p['target_frame'] or not self.p['base_frame']:
            raise ValueError('Frame names must not be empty')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.boundary_pub = self.create_publisher(PointCloud2, '~/boundary', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '~/polygons', 1)
        self.subscription = self.create_subscription(
            PointCloud2, self.p['input_topic'], self.queue_cloud, qos_profile_sensor_data)
        self.pending = None
        self.retry_timer = self.create_timer(0.02, self.process_pending)
        self.get_logger().info(f"Estimating boundary in {self.p['target_frame']}; unknown bins remain NaN")

    def on_scan(self, msg):
        self.scan = msg

    def queue_cloud(self, msg):
        self.pending_since = time.monotonic()
        self.pending = msg  # bounded queue: prefer the freshest cloud

    def process_pending(self):
        if self.pending is None:
            return
        matching = self.scan is not None and self.scan.header.stamp == self.pending.header.stamp
        if self.p['enable_assumed_clear'] and not matching and time.monotonic() - self.pending_since < 0.08:
            return
        if self.on_cloud(self.pending):
            self.pending = None

    def on_cloud(self, msg):
        try:
            stamp = Time.from_msg(msg.header.stamp)
            world_tf = self.buffer.lookup_transform(self.p['target_frame'], msg.header.frame_id, stamp)
            base_tf = self.buffer.lookup_transform(self.p['base_frame'], msg.header.frame_id, stamp)
            raw = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=False)
            raw = np.asarray(raw).reshape(-1, 3)
            raw = raw[np.isfinite(raw).all(axis=1)]
            base = transform_points(raw, base_tf.transform)
            self_hits = ((base >= self.lower) & (base <= self.upper)).all(axis=1)
            points = transform_points(raw[~self_hits], world_tf.transform)
            t = world_tf.transform.translation
            origin = np.array([t.x, t.y, t.z])
            angles, ranges = estimate(points, origin, self.p['water_level'], self.config)
        except (TransformException, ValueError, AssertionError) as exc:
            self.get_logger().warning(f'Skipping cloud: {exc}', throttle_duration_sec=5.0)
            return
        header = copy.deepcopy(msg.header)
        header.frame_id = self.p['target_frame']
        status = np.isfinite(ranges).astype(np.float32)
        scan = self.scan
        # Require the same acquisition and sensor frame; never reuse stale scans.
        if (self.p['enable_assumed_clear'] and scan is not None
                and scan.header.frame_id == msg.header.frame_id
                and scan.header.stamp == msg.header.stamp
                and np.isfinite(scan.angle_min) and np.isfinite(scan.angle_increment)
                and scan.angle_increment > 0):
            theta = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
            directions = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(len(theta))))
            directions = directions @ rotation_matrix(world_tf.transform.rotation).T
            ranges, status = extend_no_returns(ranges, scan.ranges, directions,
                                               scan.range_max, self.p['assumed_clear_range'])
        valid = np.isfinite(ranges)
        xy = origin[:2] + ranges[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        # Status: 0 UNKNOWN, 1 OBSERVED, 2 ASSUMED_CLEAR. Every angular bin has a row, even unknown.
        data = np.column_stack((xy, np.full(len(ranges), self.p['water_level']),
                                angles, ranges, status)).astype(np.float32)
        fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                  for i, name in enumerate(('x', 'y', 'z', 'theta', 'range', 'status'))]
        self.boundary_pub.publish(point_cloud2.create_cloud(header, fields, data))
        markers = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        origin_point = Point(x=float(origin[0]), y=float(origin[1]), z=float(self.p['water_level']))
        vertices = [Point(x=float(x), y=float(y), z=float(self.p['water_level'])) for x, y in xy]
        lifetime = Duration(seconds=self.p['marker_lifetime']).to_msg()

        def marker(ns, idx, kind, color):
            m = Marker(header=header, ns=ns, id=idx, type=kind, action=Marker.ADD)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = color
            m.lifetime = lifetime
            return m

        hits = marker('boundary', 0, Marker.POINTS, (1.0, 0.5, 0.0, 1.0))
        hits.scale.x = hits.scale.y = 0.15
        hits.points = [vertices[i] for i in np.flatnonzero(status == 1)]
        markers.markers.append(hits)
        for idx, run in enumerate(contiguous_runs(valid)):
            if len(run) < 2:
                continue  # A single ray has no supported polygon area.
            complete = len(run) == len(ranges)
            outline = marker('polygon', idx, Marker.LINE_STRIP, (0.0, 1.0, 0.5, 1.0))
            outline.scale.x = 0.06
            outline.points = ([vertices[i] for i in run] + [vertices[run[0]]] if complete
                              else [origin_point] + [vertices[i] for i in run] + [origin_point])
            markers.markers.append(outline)
            fill = marker('observed_space', idx, Marker.TRIANGLE_LIST, (0.0, 0.8, 0.4, 0.18))
            pairs = list(zip(run[:-1], run[1:]))
            if complete:
                pairs.append((run[-1], run[0]))
            observed_pairs = [(a, b) for a, b in pairs if status[a] == status[b] == 1]
            assumed_pairs = [(a, b) for a, b in pairs if status[a] == 2 or status[b] == 2]
            fill.points = [p for a, b in observed_pairs for p in (origin_point, vertices[a], vertices[b])]
            assumed = marker('assumed_clear', idx, Marker.TRIANGLE_LIST, (0.1, 0.45, 1.0, 0.22))
            assumed.points = [p for a, b in assumed_pairs for p in (origin_point, vertices[a], vertices[b])]
            if assumed.points:
                markers.markers.append(assumed)
            markers.markers.append(fill)
        self.marker_pub.publish(markers)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FreeSpaceNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
