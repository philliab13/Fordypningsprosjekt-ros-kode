"""Compare detector footprints with nearby, segmentation-selected LiDAR scans."""

from collections import OrderedDict, deque
from functools import partial
import json
import time

import numpy as np
from geometry_msgs.msg import PoseStamped
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
from .gnss import DEFAULT_FILES, PIREN_LLA, interpolate_pose, load_track, ned_to_lla


class BevNode(Node):
    def __init__(self):
        super().__init__('footprint_bev')
        defaults = dict(
            lidar_topic='/lidar_aft/points', footprint_topic='/obstacles/json',
            mask_topic='/obstacles/boat_mask', camera_info_topic='/obstacles/camera_info',
            rgb_topic='/obstacles/source_image',
            rgb_output_topic='/debug/bev_lidar_overlay',
            output_topic='/debug/footprint_bev', scan_count=3, max_skew_s=0.2,
            shoreline_topic='/nearest_obstacle_shoreline_detector/barrier_json',
            shoreline_fixed_frame='piren',
            scan_wait_s=0.2, range_m=60.0, size_px=900, fixed_frame='',
            **DEFAULT_FILES,
            gnss_enabled=True, gnss_pose_topic='/navigation/pose',
            gnss_body_frame='p0_floor', gnss_pose_axes='ned',
            piren_lla=PIREN_LLA, gnss_max_gap_s=2.0, pose_max_gap_s=0.5,
            gnss_time_offset_s=0.0, gps_utc_offset_s=18.0,
        )
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if (self.p['gnss_pose_axes'] not in ('ned', 'enu')
                or len(self.p['piren_lla']) != 3
                or not np.isfinite(self.p['piren_lla']).all()
                or not -90 <= self.p['piren_lla'][0] <= 90
                or not -180 <= self.p['piren_lla'][1] <= 180
                or not all(np.isfinite(self.p[k]) and self.p[k] > 0
                           for k in ('gnss_max_gap_s', 'pose_max_gap_s'))
                or not all(np.isfinite(self.p[k]) for k in
                           ('gnss_time_offset_s', 'gps_utc_offset_s'))):
            raise ValueError('Invalid GNSS origin, pose axes or timing parameters')
        if (self.p['scan_count'] < 1 or self.p['scan_count'] > 100
                or self.p['size_px'] < 300
                or not all(np.isfinite(self.p[key]) and self.p[key] >= 0
                           for key in ('max_skew_s', 'scan_wait_s'))
                or not np.isfinite(self.p['range_m']) or self.p['range_m'] <= 0):
            raise ValueError('Invalid BEV scan count, time window, size or range')
        if not self.p['shoreline_fixed_frame']:
            raise ValueError('shoreline_fixed_frame must name a stationary TF frame')
        self.clouds = deque(maxlen=100)
        self.poses = deque(maxlen=6000)
        self.gnss_tracks = []
        if self.p['gnss_enabled']:
            for key, label, color in [
                    ('spatial_dual_file', 'Spatial Dual', (180, 30, 180)),
                    ('kbox2_file', 'KBox2', (30, 110, 210)),
                    ('kbox1_file', 'KBox1', (160, 120, 20)),
                    ('ntbox_file', 'NTBox', (40, 140, 40))]:
                if not self.p[key]:
                    continue
                try:
                    track = load_track(self.p[key], self.p['gps_utc_offset_s'])
                    self.gnss_tracks.append((label, color, track))
                    self.get_logger().info(f'{label} GNSS: {track.source}')
                except (OSError, ValueError, SyntaxError) as exc:
                    self.get_logger().warning(f'{label} GNSS unavailable: {exc}')
        self.pending = OrderedDict()
        self.done = deque(maxlen=100)
        self.shorelines = OrderedDict()
        self.held_shoreline = None
        self.rendered_jobs = OrderedDict()
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
        if self.p['shoreline_topic']:
            self.subscriptions_.append(self.create_subscription(
                String, self.p['shoreline_topic'], self.receive_shoreline, 10))
        if self.p['gnss_enabled']:
            self.subscriptions_.append(self.create_subscription(
                PoseStamped, self.p['gnss_pose_topic'], self.receive_pose,
                qos_profile_sensor_data))
        self.timer = self.create_timer(0.05, self.flush)
        if not self.p['fixed_frame']:
            self.get_logger().info(
                'BEV uses calibrated MA2 LiDAR extrinsics without ego-motion '
                'compensation. Set fixed_frame to an available odom/map TF frame '
                'to align scans at the image timestamp.')

    def receive_cloud(self, msg):
        self.clouds.append((stamp_ns(msg), msg))

    def receive_pose(self, msg):
        pos, q = msg.pose.position, msg.pose.orientation
        values = np.array([pos.x, pos.y, pos.z, q.x, q.y, q.z, q.w])
        if np.isfinite(values).all() and np.linalg.norm(values[3:]) > 1e-8:
            self.poses.append((stamp_ns(msg), values))

    def gnss_overlay(self, timestamp, header, camera_to_water):
        """Put horizontal GNSS positions through the existing BEV geometry."""
        tracks = getattr(self, 'gnss_tracks', [])
        if not self.p.get('gnss_enabled', True):
            return [], ''
        pose = interpolate_pose(self.poses, timestamp, self.p['pose_max_gap_s'])
        if pose is None:
            return [], ' | GNSS: no matching pose'
        position, rotation = pose
        if self.p['gnss_pose_axes'] == 'enu':
            ned_from_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            position = ned_from_enu @ position
            rotation = ned_from_enu @ rotation
        try:
            body_from_camera = self.lookup(self.p['gnss_body_frame'],
                                           header.frame_id, header.stamp)
        except Exception as exc:
            self.get_logger().warning(f'GNSS camera transform unavailable: {exc}',
                                      throttle_duration_sec=2.0)
            return [], ' | GNSS: missing camera TF'
        camera_from_body = np.linalg.inv(body_from_camera)
        references = []
        query = timestamp - round(self.p['gnss_time_offset_s'] * 1e9)
        for label, color, track in tracks:
            target = track.at(query, self.p['gnss_max_gap_s'], self.p['piren_lla'])
            if target is None:
                continue
            # Ignore antenna altitude: this is a horizontal reference comparison.
            target[2] = position[2]
            body = rotation.T @ (target - position)
            camera = camera_from_body[:3, :3] @ body + camera_from_body[:3, 3]
            xy, _ = local_geometry([camera], {'features': []}, camera_to_water)
            references.append(dict(label=label, color=color, xy=xy[0]))
        count = len(references)
        # The navigation pose locates the ego body origin in Piren. In body
        # coordinates this is zero, but it is offset from the camera origin.
        ego_xy, _ = local_geometry(
            [camera_from_body[:3, 3]], {'features': []}, camera_to_water)
        camera_ned = rotation @ body_from_camera[:3, 3] + position
        forward_ned = rotation @ body_from_camera[:3, 2]
        heading = (float(np.degrees(np.arctan2(forward_ned[1], forward_ned[0])) % 360)
                   if np.linalg.norm(forward_ned[:2]) > 1e-8 else None)
        references.append(dict(label='Ego MA2', color=(200, 100, 0), xy=ego_xy[0],
                               ego=True, position_ned=position.copy(),
                               position_lla=ned_to_lla(position, self.p['piren_lla']),
                               camera_lla=ned_to_lla(camera_ned, self.p['piren_lla']),
                               camera_heading_deg=heading))
        return references, f' | GNSS {count}/{len(tracks)} + ego'

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

    def receive_shoreline(self, msg):
        try:
            data = json.loads(msg.data)
            stamp = data['stamp']
            if (not isinstance(data['frame_id'], str) or not data['frame_id']
                    or type(stamp['sec']) is not int or type(stamp['nanosec']) is not int
                    or not 0 <= stamp['nanosec'] < 1_000_000_000):
                raise ValueError('Invalid shoreline frame or timestamp')
            timestamp = stamp['sec'] * 1_000_000_000 + stamp['nanosec']
            segments = np.asarray(data['segments'], dtype=float)
            if segments.size == 0:
                segments = np.empty((0, 2, 3))
            if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
                raise ValueError('Expected finite shoreline endpoint pairs in XYZ')
            if not len(segments):
                return  # Missing measurements do not erase the retained outline.
            self.shorelines[timestamp] = (data['frame_id'], segments)
            while len(self.shorelines) > 100:
                self.shorelines.popitem(last=False)
            # Inference can finish after the matching footprint BEV was drawn.
            # Redraw the acquisition frame and motion-compensate into the latest view.
            job = self.rendered_jobs.get(timestamp)
            if job is not None:
                self.publish_frame(timestamp, job)
            if self.rendered_jobs:
                latest_stamp = max(self.rendered_jobs)
                if latest_stamp > timestamp:
                    self.publish_frame(latest_stamp, self.rendered_jobs[latest_stamp])
        except Exception as exc:
            self.get_logger().warning(f'BEV shoreline unavailable: {exc}',
                                      throttle_duration_sec=2.0)

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

    def shoreline_overlay(self, timestamp, header, camera_to_water):
        """Anchor measurements in a stationary frame; project at image time."""
        empty = np.empty((0, 2, 2))
        shorelines = getattr(self, 'shorelines', {})
        candidates = [stamp for stamp in shorelines if stamp <= timestamp]
        shore_stamp = max(candidates) if candidates else None
        held = getattr(self, 'held_shoreline', None)
        fixed = self.p['shoreline_fixed_frame']
        note = ''
        if shore_stamp is not None and (held is None or shore_stamp > held[0]):
            source, segments = shorelines[shore_stamp]
            try:
                fixed_from_shore = self.lookup(
                    fixed, source, Time(nanoseconds=shore_stamp).to_msg())
                world = (segments.reshape(-1, 3) @ fixed_from_shore[:3, :3].T
                         + fixed_from_shore[:3, 3])
                held = (shore_stamp, world.reshape(-1, 2, 3))
                self.held_shoreline = held
            except Exception as exc:
                # Keep the previous world geometry and retry the new measurement later.
                note = ' | Shore update: missing TF'
                self.get_logger().warning(f'BEV shoreline anchor unavailable: {exc}',
                                          throttle_duration_sec=2.0)
        if held is None or held[0] > timestamp:
            return empty, note
        age = (timestamp - held[0]) / 1e9
        try:
            camera_from_fixed = self.lookup(header.frame_id, fixed, header.stamp)
            camera = (held[1].reshape(-1, 3) @ camera_from_fixed[:3, :3].T
                      + camera_from_fixed[:3, 3])
            xy, _ = local_geometry(camera, {'features': []}, camera_to_water)
        except Exception as exc:
            self.get_logger().warning(f'BEV shoreline projection unavailable: {exc}',
                                      throttle_duration_sec=2.0)
            return empty, note + f' | Shore: missing TF (age={age:.1f}s)'
        return xy.reshape(-1, 2, 2), (note + f' | Shore {len(held[1])} segments '
                                    f'age={age:.1f}s (held, ego-compensated)')

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
        references, gnss_status = BevNode.gnss_overlay(
            self, timestamp, mask_msg.header, camera_to_water)
        status += gnss_status
        shoreline_xy, shoreline_status = BevNode.shoreline_overlay(
            self, timestamp, mask_msg.header, camera_to_water)
        status += shoreline_status
        image = render_bev(
            points, rings, status, self.p['size_px'], self.p['range_m'],
            point_colors=np.concatenate(colors) if colors else np.empty((0, 3)),
            scan_legend=legend, gnss_references=references,
            shoreline_segments=shoreline_xy)
        rgb_overlay = render_lidar_overlay(rgb, projections, legend, status)
        for publisher, array in [(self.publisher, image), (self.rgb_publisher, rgb_overlay)]:
            msg = Image()
            msg.header = mask_msg.header
            msg.height, msg.width = array.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = msg.width * 3
            msg.data = array.tobytes()
            publisher.publish(msg)
        if hasattr(self, 'rendered_jobs'):
            self.rendered_jobs[timestamp] = job
            while len(self.rendered_jobs) > 20:
                self.rendered_jobs.popitem(last=False)


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
