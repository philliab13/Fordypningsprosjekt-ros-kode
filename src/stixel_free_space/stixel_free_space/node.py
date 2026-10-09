"""ROS 2 stereo/lidar adapter for FreeSpacePipeline, with a bounded worker queue."""
from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

from .core import (LatestSlot, compute_stereo, nearest_cloud, pipeline_pose,
                   project_lidar, stamp_ns, stereo_calibration, transform_matrix, nearest_pose, stereo_lidar_depth_errors)
from .outputs import cloud_message, marker_message, overlay_image

from .ma2_calibration import (
    H_POINTS_ZED_FROM_LIDAR,
    H_POINTS_PIREN_FROM_PIREN_ENU,
    T_BODY_TO_CAM,
    R_BODY_TO_CAM,
)


PREFIX = '/wamv/sensors/cameras/'
DEFAULTS = dict(
    repo_path=str(Path.home(
    ) / 'Documents/Stixel_based_free_space_estimation/free-space-estimation'),
    left_topic=PREFIX + 'front_left_camera_sensor/optical/image_raw',
    right_topic=PREFIX + 'front_right_camera_sensor/optical/image_raw',
    left_info_topic=PREFIX + 'front_left_camera_sensor/optical/camera_info',
    right_info_topic=PREFIX + 'front_right_camera_sensor/optical/camera_info',
    lidar_topic='/wamv/sensors/lidars/lidar_wamv_sensor/points',
    fixed_frame='gymkhana_task', stereo_slop_s=0.02, lidar_max_skew_s=0.10, ma2_max_skew_s=0.50,
    require_lidar=True, max_input_age_s=0.5, tf_wait_s=0.2, reset_gap_s=1.0,
    num_stixels=128, retina_masks=True, depth_filter=True, max_prop_frames=5,
    camera_height=0.0, water_level=0.0, fastsam_weights='', yolo_weights='', rwps_config='',
    num_disparities=256, block_size=5, publish_images=True, publish_markers=True,
    marker_lifetime_s=5.0, timing_window=100, warmup_frames=3,
    input_mode='vrx',

    ma2_left_topic='/ma2/zed/left/image',
    ma2_depth_topic='/ma2/zed/depth',
    ma2_disparity_topic='/ma2/zed/disparity',
    ma2_info_topic='/ma2/zed/camera_info',

    ma2_lidar_topic='/lidar_aft/points',
    ma2_pose_topic='/senti_parser/SentiPose',
)


class StixelNode(Node):
    def __init__(self, *, backend=None, start_worker=True, **kwargs):
        super().__init__('stixel_free_space', **kwargs)
        self.p = {key: self.declare_parameter(key, value, ParameterDescriptor(read_only=True)).value
                  for key, value in DEFAULTS.items()}
        self._validate()
        self.buffer = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.buffer, self)
        self.lock = threading.Lock()
        self.clouds = deque(maxlen=60)
        self.infos = {}
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.last_stamp = None
        self.last_pose = None
        self.last_frames = None
        self.last_input_stamp = None
        self.input_epoch = 0
        self.processed_input_epoch = 0
        self.started = time.perf_counter()
        self.input_times = deque(maxlen=self.p['timing_window'])
        self.output_times = deque(maxlen=self.p['timing_window'])
        self.samples = deque(maxlen=self.p['timing_window'])
        self.metrics = dict(received_pairs=0, processed_frames=0, skipped_frames=0,
                            errors=0, temporal_resets=0, segmentation_failures=0,
                            input_discontinuities=0, nonmonotonic_input_pairs=0,
                            state='waiting for synchronized images and CameraInfo')
        self.backend = backend
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.p['num_disparities'], blockSize=self.p['block_size'],
            P1=8*self.p['block_size']**2, P2=32*self.p['block_size']**2,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.cloud_pub = self.create_publisher(PointCloud2, '~/stixels', 1)
        self.markers_pub = self.create_publisher(MarkerArray, '~/markers', 1)
        self.overlay_pub = self.create_publisher(Image, '~/overlay', 1)
        self.water_pub = self.create_publisher(Image, '~/water_mask', 1)
        self.depth_pub = self.create_publisher(Image, '~/depth', 1)
        self.diagnostic_pub = self.create_publisher(
            DiagnosticArray, '~/diagnostics', 1)
        if self.p['input_mode'] == 'ma2':
            self.ma2_info_sub = self.create_subscription(
                CameraInfo,
                self.p['ma2_info_topic'],
                self.receive_ma2_info,
                qos_profile_sensor_data
            )

            self.lidar_sub = self.create_subscription(
                PointCloud2,
                self.p['ma2_lidar_topic'],
                self.receive_cloud,
                qos_profile_sensor_data
            )

            self.ma2_image_subs = [
                Subscriber(
                    self,
                    Image,
                    topic,
                    qos_profile=qos_profile_sensor_data
                )
                for topic in (
                    self.p['ma2_left_topic'],
                    self.p['ma2_depth_topic'],
                    self.p['ma2_disparity_topic'],
                )
            ]
            self.ma2_pose_sub = self.create_subscription(
                PoseStamped,
                self.p['ma2_pose_topic'],
                self.receive_ma2_pose,
                qos_profile_sensor_data
            )
            self.poses = deque(maxlen=500)

            self.ma2_sync = ApproximateTimeSynchronizer(
                self.ma2_image_subs,
                queue_size=5,
                slop=0.005
            )

            self.ma2_sync.registerCallback(
                self.receive_ma2_frame
            )

        elif self.p['input_mode'] == 'vrx':
            self.info_subs = [self.create_subscription(
                CameraInfo, self.p[side + '_info_topic'],
                lambda msg, side=side: self.receive_info(side, msg), qos_profile_sensor_data)
                for side in ('left', 'right')]
            self.lidar_sub = self.create_subscription(PointCloud2, self.p['lidar_topic'],
                                                      self.receive_cloud, qos_profile_sensor_data)
            self.image_subs = [Subscriber(self, Image, self.p[side + '_topic'],
                                          qos_profile=qos_profile_sensor_data) for side in ('left', 'right')]
            self.sync = ApproximateTimeSynchronizer(
                self.image_subs, queue_size=5, slop=self.p['stereo_slop_s'])
            self.sync.registerCallback(self.receive_pair)

        else:
            raise ValueError(f"Unsupported input_mode: {self.p['input_mode']}")

        self.timer = self.create_timer(1.0, self.publish_diagnostics)
        self.worker = None
        if start_worker:
            # Import/model loading happens in the worker; callbacks remain responsive.
            self.worker = threading.Thread(
                target=self.work, name='stixel-inference', daemon=True)
            self.worker.start()
        self.get_logger().info(
            'Original stixel pipeline adapter started; see ~/diagnostics for readiness and timings')

    def _validate(self):
        for key in ('stereo_slop_s', 'lidar_max_skew_s', 'max_input_age_s', 'tf_wait_s',
                    'reset_gap_s', 'marker_lifetime_s', 'ma2_max_skew_s'):
            if not np.isfinite(self.p[key]) or self.p[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if self.p['num_disparities'] <= 0 or self.p['num_disparities'] % 16:
            raise ValueError(
                'num_disparities must be a positive multiple of 16')
        if self.p['block_size'] < 3 or self.p['block_size'] % 2 == 0:
            raise ValueError('block_size must be odd and >= 3')
        if (self.p['num_stixels'] < 1 or self.p['timing_window'] < 2
                or self.p['warmup_frames'] < 0 or self.p['max_prop_frames'] < 0):
            raise ValueError(
                'Invalid column count, timing window, warmup or propagation limit')
        if (not np.isfinite(self.p['camera_height']) or self.p['camera_height'] < 0
                or not np.isfinite(self.p['water_level']) or not self.p['fixed_frame']):
            raise ValueError(
                'Invalid camera height, water level or fixed frame')

    @staticmethod
    def image_msg_to_numpy(msg):
        if msg.encoding == 'bgr8':
            dtype = np.uint8
            channels = 3

        elif msg.encoding == 'mono8':
            dtype = np.uint8
            channels = 1

        elif msg.encoding == '32FC1':
            dtype = np.float32
            channels = 1

        else:
            raise ValueError(
                f'Unsupported image encoding: {msg.encoding}'
            )

        array = np.frombuffer(
            msg.data,
            dtype=dtype
        )

        row_elements = msg.step // np.dtype(dtype).itemsize

        array = array.reshape(
            msg.height,
            row_elements
        )

        if channels == 1:
            return np.ascontiguousarray(
                array[:, :msg.width]
            )

        array = array[:, :msg.width * channels]

        return np.ascontiguousarray(
            array.reshape(
                msg.height,
                msg.width,
                channels
            )
        )

    @staticmethod
    def numpy_to_image_msg(array, encoding, header):
        array = np.ascontiguousarray(array)

        msg = Image()
        msg.header = header
        msg.height = array.shape[0]
        msg.width = array.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False

        if array.ndim == 3:
            msg.step = (
                array.shape[1]
                * array.shape[2]
                * array.dtype.itemsize
            )
        else:
            msg.step = (
                array.shape[1]
                * array.dtype.itemsize
            )

        msg.data = array.tobytes()

        return msg

    def receive_ma2_info(self, msg):
        with self.lock:
            self.infos['ma2'] = msg
            
    def receive_ma2_pose(self, msg):
        with self.lock:
            self.poses.append(msg)

    def receive_ma2_frame(
        self,
        left,
        depth,
        disparity
    ):
        arrived = time.perf_counter()
        timestamp = stamp_ns(left)

        with self.lock:
            self.metrics['received_pairs'] += 1
            self.input_times.append(arrived)

            if self.last_input_stamp is not None:
                delta_s = (
                    timestamp - self.last_input_stamp
                ) / 1e9

                if (
                    -self.p['reset_gap_s']
                    <= delta_s
                    <= 0
                ):
                    self.metrics[
                        'nonmonotonic_input_pairs'
                    ] += 1
                    return

                if (
                    delta_s < -self.p['reset_gap_s']
                    or delta_s > self.p['reset_gap_s']
                ):
                    self.input_epoch += 1
                    self.metrics[
                        'input_discontinuities'
                    ] += 1

            self.last_input_stamp = timestamp

            info = self.infos.get('ma2')
            input_epoch = self.input_epoch

        self.slot.put(
            (
                left,
                depth,
                disparity,
                info,
                arrived,
                input_epoch
            )
        )

    def receive_info(self, side, msg):
        with self.lock:
            self.infos[side] = msg

    def receive_cloud(self, msg):
        with self.lock:
            self.clouds.append(msg)

    def receive_pair(self, left, right):
        arrived = time.perf_counter()
        timestamp = stamp_ns(left)
        with self.lock:
            self.metrics['received_pairs'] += 1
            self.input_times.append(arrived)
            if self.last_input_stamp is not None:
                delta_s = (timestamp - self.last_input_stamp) / 1e9
                if -self.p['reset_gap_s'] <= delta_s <= 0:
                    # ApproximateTimeSynchronizer can emit a duplicate or slightly
                    # reordered pair. It is stale, but it is not a clock reset.
                    self.metrics['nonmonotonic_input_pairs'] += 1
                    return
                if delta_s < -self.p['reset_gap_s'] or delta_s > self.p['reset_gap_s']:
                    self.input_epoch += 1
                    self.metrics['input_discontinuities'] += 1
            self.last_input_stamp = timestamp
            infos = self.infos.copy()
            input_epoch = self.input_epoch
        self.slot.put((left, right, infos, arrived, input_epoch))

    def update_metrics(self, **values):
        with self.lock:
            self.metrics.update(values)

    def skip(self, reason):
        with self.lock:
            self.metrics['skipped_frames'] += 1
            self.metrics['state'] = reason
        self.get_logger().warning(reason, throttle_duration_sec=5.0)

    def work(self):
        try:
            from .backend import PipelineBackend
            self.backend = PipelineBackend(
                self.p) if self.backend is None else self.backend
            self.update_metrics(**getattr(self.backend, 'runtime_info', {}))
        except Exception as exc:
            self.update_metrics(state=f'backend unavailable: {exc}', errors=1)
            self.get_logger().error(str(exc))
            return
        while not self.stop_event.is_set():
            job = self.slot.take()
            if job is None:
                return
            try:
                self.process_job(job)
            except Exception as exc:
                # The upstream no-mask failure is deliberately visible, not silently patched.
                self.backend.reset()
                self.last_stamp = self.last_pose = None
                with self.lock:
                    self.metrics['errors'] += 1
                    self.metrics['state'] = f'processing error: {type(exc).__name__}: {exc}'
                self.get_logger().error(
                    f'Frame failed; temporal state reset: {exc}', throttle_duration_sec=5.0)

    def lookup(self, left, right, cloud):
        deadline = time.perf_counter() + self.p['tf_wait_s']
        while True:
            try:
                stamp = Time.from_msg(left.header.stamp)
                world = self.buffer.lookup_transform(
                    self.p['fixed_frame'], left.header.frame_id, stamp)
                stereo = self.buffer.lookup_transform(
                    left.header.frame_id, right.header.frame_id, stamp)
                lidar = None
                if cloud is not None:
                    lidar = self.buffer.lookup_transform_full(
                        left.header.frame_id, stamp, cloud.header.frame_id,
                        Time.from_msg(cloud.header.stamp), self.p['fixed_frame'])
                return (transform_matrix(world.transform), transform_matrix(stereo.transform),
                        None if lidar is None else transform_matrix(lidar.transform))
            except TransformException:
                if time.perf_counter() >= deadline or self.stop_event.wait(0.01):
                    raise

    def process_job(self, job):
        if self.p['input_mode'] == 'ma2':
            return self.process_ma2_job(job)

        return self.process_vrx_job(job)

    def process_ma2_job(self, job):
        (
            left_msg,
            depth_msg,
            disparity_msg,
            info,
            arrived,
            input_epoch,
        ) = job

        started = time.perf_counter()
        timestamp = stamp_ns(left_msg)

        if info is None:
            self.skip('Waiting for MA2 CameraInfo')
            return

        # --------------------------------------------------
        # Convert ZED ROS images to NumPy
        # --------------------------------------------------

        left = self.image_msg_to_numpy(
            left_msg
        )

        depth = self.image_msg_to_numpy(
            depth_msg
        ).astype(
            np.float32,
            copy=False
        )

        disparity = self.image_msg_to_numpy(
            disparity_msg
        ).astype(
            np.float32,
            copy=False
        )

        shape = (
            int(info.height),
            int(info.width),
        )

        if left.shape[:2] != shape:
            raise ValueError(
                'MA2 left image does not match CameraInfo'
            )

        if depth.shape != shape:
            raise ValueError(
                'MA2 depth image does not match CameraInfo'
            )

        if disparity.shape != shape:
            raise ValueError(
                'MA2 disparity image does not match CameraInfo'
            )

        # --------------------------------------------------
        # Camera intrinsics
        # --------------------------------------------------

        k = np.array(
            info.k,
            dtype=np.float64
        ).reshape(3, 3)

        # ZED stereo baseline:
        # better to publish this later from the SVO node,
        # but for now use the known calibrated ZED baseline.
        baseline = 0.12
        with self.lock:
            cloud = nearest_cloud(
                tuple(self.clouds),
                timestamp,
                self.p['ma2_max_skew_s']
            )

            pose_msg = nearest_pose(
                tuple(self.poses),
                timestamp,
                max_skew_s=0.5
            )
        if pose_msg is None:
            self.skip(
                'No MA2 pose near camera timestamp'
            )
            return

        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation

        pos_enu = np.array([
            pos.x,
            pos.y,
            pos.z,
        ])

        pos_ned = (
            H_POINTS_PIREN_FROM_PIREN_ENU
            @ np.array([
                pos_enu[0],
                pos_enu[1],
                pos_enu[2],
                1.0,
            ])
        )[:3]

        pose = np.array([
            pos_ned[0],
            pos_ned[1],
            pos_ned[2],
            ori.x,
            ori.y,
            ori.z,
            ori.w,
        ])
        uv = np.empty((0, 2))
        xyz = np.empty((0, 3))

        if cloud is not None:
            raw = point_cloud2.read_points_numpy(
                cloud,
                field_names=('x', 'y', 'z'),
                skip_nans=False
            )

            valid = np.isfinite(raw).all(axis=1)
            raw = raw[valid]

            ones = np.ones(
                (raw.shape[0], 1),
                dtype=np.float64
            )

            raw_h = np.hstack([
                raw.astype(np.float64),
                ones
            ])

            xyz_camera_h = (
                H_POINTS_ZED_FROM_LIDAR
                @ raw_h.T
            ).T

            xyz = xyz_camera_h[:, :3]

            forward = xyz[:, 2] > 0
            xyz = xyz[forward]

            projected = (
                k @ xyz.T
            ).T

            uv = projected[:, :2] / projected[:, 2:3]

            in_image = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < shape[0])
            )

            uv = uv[in_image]
            xyz = xyz[in_image]
            
            depth_errors = stereo_lidar_depth_errors(
                depth,
                uv,
                xyz,
                max_depth_m=20.0,
            )
            if len(depth_errors) > 0:
                stereo_lidar_mae = float(
                    np.mean(np.abs(depth_errors))
                )

                stereo_lidar_rmse = float(
                    np.sqrt(np.mean(depth_errors ** 2))
                )

                stereo_lidar_bias = float(
                    np.mean(depth_errors)
                )

                stereo_lidar_pairs = len(depth_errors)

            else:
                stereo_lidar_mae = float('nan')
                stereo_lidar_rmse = float('nan')
                stereo_lidar_bias = float('nan')
                stereo_lidar_pairs = 0
        
        camera_height = self.p['camera_height']

        if camera_height <= 0:
            camera_height = 1.6
            
        configured = time.perf_counter()

        self.backend.configure(
            k,
            baseline,
            shape,
            camera_height,
            t_body_to_cam=T_BODY_TO_CAM,
            R_body_to_cam=R_BODY_TO_CAM,
        )

        setup_ms = (
            time.perf_counter()
            - configured
        ) * 1000

        self.backend.synchronize()

        pipeline_start = time.perf_counter()

        result, stixels = self.backend.run(
            timestamp=timestamp,
            left_img=left,
            disparity_img=disparity,
            depth_img=depth,
            lidar_uv=uv,
            lidar_xyz=xyz,
            pose=pose
        )

        self.backend.synchronize()

        pipeline_ms = (
            time.perf_counter()
            - pipeline_start
        ) * 1000
        self.last_stamp = timestamp
        self.last_pose = pose
        self.last_frames = (left_msg.header.frame_id,)
        self.processed_input_epoch = input_epoch

        publishing = time.perf_counter()

        self.cloud_pub.publish(
            cloud_message(
                left_msg.header,
                stixels,
                result.stixel_footprints
            )
        )

        if self.p['publish_markers']:
            self.markers_pub.publish(
                marker_message(
                    left_msg.header,
                    stixels,
                    k,
                    self.p['marker_lifetime_s']
                )
            )

        if self.p['publish_images']:
            for publisher, array, encoding in (
                (
                    self.overlay_pub,
                    overlay_image(left, result, stixels),
                    'bgr8'
                ),
                (
                    self.water_pub,
                    (result.water_mask * 255).astype(np.uint8),
                    'mono8'
                ),
                (
                    self.depth_pub,
                    depth,
                    '32FC1'
                ),
            ):
                msg = self.numpy_to_image_msg(
                    array,
                    encoding,
                    left_msg.header
                )
                publisher.publish(msg)

        finished = time.perf_counter()

        frame_info = getattr(
            self.backend,
            'last_frame_info',
            {}
        )

        metrics = dict(
            queue_ms=(started - arrived) * 1000,
            setup_ms=setup_ms,
            stereo_ms=0.0,
            pipeline_ms=pipeline_ms,
            upstream_runtime_ms=float(result.runtime_ms),
            publish_ms=(finished - publishing) * 1000,
            worker_ms=(finished - started) * 1000,
            receive_to_publish_ms=(finished - arrived) * 1000,
            lidar_skew_ms=(
                abs(stamp_ns(cloud) - timestamp) / 1e6
                if cloud is not None
                else float('nan')
            ),
            lidar_points=len(xyz),
            valid_stixels=int(
                np.count_nonzero(
                    stixels.stixel_validity
                )
            ),
            water_mask_failure=bool(
                result.water_mask_failure
            ),
            baseline_m=baseline,
            camera_height_m=camera_height,
            image_width=shape[1],
            image_height=shape[0],
            final_water_ratio=float(
                np.mean(result.water_mask)
            ),
            stereo_lidar_pairs=stereo_lidar_pairs,
            stereo_lidar_mae_m=stereo_lidar_mae,
            stereo_lidar_rmse_m=stereo_lidar_rmse,
            stereo_lidar_bias_m=stereo_lidar_bias,
            **frame_info,
        )

        with self.lock:
            self.metrics['processed_frames'] += 1
            self.metrics['segmentation_failures'] += int(
                result.water_mask_failure
            )
            self.metrics.update(metrics)
            self.metrics['state'] = (
                'segmentation rejected'
                if result.water_mask_failure
                else 'running'
            )
            self.output_times.append(finished)

            if (
                self.metrics['processed_frames']
                > self.p['warmup_frames']
            ):
                self.samples.append(metrics)
        

    def process_job_vrx(self, job):
        left_msg, right_msg, infos, arrived, *epoch = job
        input_epoch = epoch[0] if epoch else self.input_epoch
        started = time.perf_counter()
        timestamp = stamp_ns(left_msg)
        age_s = (self.get_clock().now().nanoseconds - timestamp) / 1e9
        if age_s > self.p['max_input_age_s'] or age_s < -self.p['stereo_slop_s']:
            self.skip(
                f'Image timestamp outside current ROS clock window: age={age_s:.3f}s')
            return
        if set(infos) != {'left', 'right'}:
            self.skip('Waiting for both CameraInfo topics')
            return
        for side, msg in (('left', left_msg), ('right', right_msg)):
            if not msg.header.frame_id or msg.header.frame_id != infos[side].header.frame_id:
                raise ValueError(
                    f'{side} Image and CameraInfo must share a nonempty optical frame')
        with self.lock:
            cloud = nearest_cloud(tuple(self.clouds),
                                  timestamp, self.p['lidar_max_skew_s'])
        if cloud is None and self.p['require_lidar']:
            self.skip(
                'No lidar scan within lidar_max_skew_s; waiting for matched sensor data')
            return
        try:
            world, stereo, lidar_tf = self.lookup(left_msg, right_msg, cloud)
        except TransformException as exc:
            self.skip(f'Timestamped TF unavailable: {exc}')
            return
        k, baseline, shape = stereo_calibration(
            infos['left'], infos['right'], stereo, self.p['num_stixels'])
        left = self.image_msg_to_numpy(left_msg)
        right = self.image_msg_to_numpy(right_msg)
        if left.shape[:2] != shape or right.shape != left.shape:
            raise ValueError('Images do not match CameraInfo dimensions')
        if shape[1] <= self.p['num_disparities'] + self.p['block_size']:
            raise ValueError(
                'Image too narrow for configured StereoSGBM disparity search')
        pose = pipeline_pose(world)
        frames = (left_msg.header.frame_id, right_msg.header.frame_id)
        if self.last_frames is not None and frames != self.last_frames:
            raise ValueError('Camera frames changed; restart the node')
        # Preserve upstream behavior within continuous runs, reset after gaps/clock jumps.
        # A slow latest-frame worker naturally processes timestamps several seconds
        # apart. Only reset for a gap observed at the sensor input, a clock jump, or
        # the original planar heading discontinuity.
        reset = self.last_stamp is not None and (
            timestamp <= self.last_stamp or input_epoch != self.processed_input_epoch)
        if self.last_pose is not None:
            from scipy.spatial.transform import Rotation
            angles = Rotation.from_quat(
                [self.last_pose[3:], pose[3:]]).as_euler('xyz')[:, 2]
            reset = reset or abs(angles[1] - angles[0]) > np.pi
        if reset:
            self.backend.reset()
            with self.lock:
                self.metrics['temporal_resets'] += 1
        camera_height = self.p['camera_height'] or float(
            world[2, 3] - self.p['water_level'])
        if camera_height <= 0:
            raise ValueError('Camera must be above water_level')
        configured = time.perf_counter()
        self.backend.configure(k, baseline, shape, camera_height)
        setup_ms = (time.perf_counter() - configured) * 1000
        uv, xyz = np.empty((0, 2)), np.empty((0, 3))
        if cloud is not None:
            raw = point_cloud2.read_points_numpy(
                cloud, field_names=('x', 'y', 'z'), skip_nans=False)
            uv, xyz = project_lidar(raw, lidar_tf, k, shape)
        stereo_start = time.perf_counter()
        disparity, depth = compute_stereo(
            self.matcher, left, right, k[0, 0], baseline)
        stereo_ms = (time.perf_counter() - stereo_start) * 1000
        self.backend.synchronize()
        pipeline_start = time.perf_counter()
        result, stixels = self.backend.run(
            timestamp=timestamp, left_img=left, disparity_img=disparity, depth_img=depth,
            lidar_uv=uv, lidar_xyz=xyz, pose=pose)
        self.backend.synchronize()
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        self.last_stamp, self.last_pose, self.last_frames = timestamp, pose, frames
        self.processed_input_epoch = input_epoch
        publishing = time.perf_counter()
        self.cloud_pub.publish(cloud_message(
            left_msg.header, stixels, result.stixel_footprints))
        if self.p['publish_markers']:
            self.markers_pub.publish(marker_message(
                left_msg.header, stixels, k, self.p['marker_lifetime_s']))
        if self.p['publish_images']:
            for publisher, array, encoding in (
                (self.overlay_pub, overlay_image(left, result, stixels), 'bgr8'),
                (self.water_pub, (result.water_mask * 255).astype(np.uint8), 'mono8'),
                (self.depth_pub, depth, '32FC1'),
            ):
                msg = self.numpy_to_image_msg(array, encoding, left_msg.header)
                msg.header = left_msg.header
                publisher.publish(msg)
        finished = time.perf_counter()
        frame_info = getattr(self.backend, 'last_frame_info', {})
        metrics = dict(
            queue_ms=(started-arrived)*1000, setup_ms=setup_ms, stereo_ms=stereo_ms,
            pipeline_ms=pipeline_ms, upstream_runtime_ms=float(
                result.runtime_ms),
            publish_ms=(finished-publishing)*1000, worker_ms=(finished-started)*1000,
            receive_to_publish_ms=(finished-arrived)*1000,
            output_age_ros_ms=(
                self.get_clock().now().nanoseconds-timestamp)/1e6,
            stereo_skew_ms=abs(stamp_ns(right_msg)-timestamp)/1e6,
            lidar_skew_ms=abs(stamp_ns(cloud)-timestamp) /
            1e6 if cloud is not None else float('nan'),
            lidar_points=len(xyz), valid_stixels=int(np.count_nonzero(stixels.stixel_validity)),
            water_mask_failure=bool(result.water_mask_failure), baseline_m=baseline,
            camera_height_m=camera_height, image_width=shape[1], image_height=shape[0],
            final_water_ratio=float(np.mean(result.water_mask)), **frame_info,
        )
        with self.lock:
            self.metrics['processed_frames'] += 1
            self.metrics['segmentation_failures'] += int(
                result.water_mask_failure)
            self.metrics.update(metrics)
            self.metrics['state'] = 'segmentation rejected' if result.water_mask_failure else 'running'
            self.output_times.append(finished)
            if self.metrics['processed_frames'] > self.p['warmup_frames']:
                self.samples.append(metrics)

    def publish_diagnostics(self):
        with self.lock:
            values = self.metrics.copy()
            samples = list(self.samples)
            input_times, output_times = tuple(
                self.input_times), tuple(self.output_times)
        now = time.perf_counter()
        values['queue_overwrites'] = self.slot.dropped
        for name, times in (('input_pairs_hz_wall', input_times), ('output_hz_wall', output_times)):
            values[name] = ((len(times)-1) / max(now-times[0],
                            1e-9)) if len(times) > 1 else 0.0
        values['last_output_age_wall_s'] = now - \
            output_times[-1] if output_times else float('nan')
        values['timing_samples'] = len(samples)
        for name in ('stereo_ms', 'pipeline_ms', 'worker_ms', 'receive_to_publish_ms'):
            if samples:
                for percentile in (50, 95):
                    values[f'{name}_p{percentile}'] = float(
                        np.percentile([s[name] for s in samples], percentile))
        healthy = values['state'] == 'running' and output_times and now - \
            output_times[-1] < 5.0
        failed = str(values['state']).startswith(
            ('backend unavailable:', 'processing error:'))
        status = DiagnosticStatus(
            level=DiagnosticStatus.ERROR if failed else DiagnosticStatus.OK if healthy else DiagnosticStatus.WARN,
            name='stixel_free_space', hardware_id='original FreeSpacePipeline',
            message=str(values['state']),
            values=[KeyValue(key=str(k), value=str(v)) for k, v in values.items()])
        msg = DiagnosticArray(status=[status])
        msg.header.stamp = self.get_clock().now().to_msg()
        self.diagnostic_pub.publish(msg)
        self.get_logger().info(
            f"{values['state']}; output={values['output_hz_wall']:.2f} Hz, "
            f"pipeline={values.get('pipeline_ms', 0):.0f} ms, "
            f"valid={values.get('valid_stixels', 0)}, dropped={values['queue_overwrites']}",
            throttle_duration_sec=5.0)

    def destroy_node(self):
        self.stop_event.set()
        self.slot.close()
        if self.worker is not None:
            self.worker.join()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = StixelNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
