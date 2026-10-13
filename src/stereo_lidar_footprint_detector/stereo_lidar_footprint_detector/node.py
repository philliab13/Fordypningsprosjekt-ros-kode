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
import threading
import cv2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from stixel_free_space.core import nearest_cloud, project_lidar, stamp_ns, transform_matrix, LatestSlot
from .backend import PipelineBackend
from .images import image_array
from stixel_free_space.ma2_calibration import (
    H_POINTS_ZED_FROM_LIDAR,
)


class FootprintNode(Node):
    def __init__(self):
        super().__init__('stereo_lidar_footprint_detector')
        defaults = dict(repo_path='', baseline_m=0.0, weights='', device='',
                        left_topic='/ma2/zed/left/image',
                        depth_topic='/ma2/zed/depth', camera_info_topic='/ma2/zed/camera_info',
                        lidar_topic='/lidar_aft/points', output_topic='/obstacles/json',
                        water_frame='', max_stereo_depth_m=20.0, lidar_max_skew_s=0.5,
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
        self.cloud_lock = threading.Lock()
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.publisher = self.create_publisher(String, self.p['output_topic'], 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/obstacles/markers',
            10,
        )
        self.cloud_sub = self.create_subscription(PointCloud2, self.p['lidar_topic'],
                                                  self.receive_cloud, qos_profile_sensor_data)
        self.inputs = [message_filters.Subscriber(self, kind, self.p[name], qos_profile=qos_profile_sensor_data)
                       for kind, name in [(Image, 'left_topic'), (Image, 'depth_topic'),
                                          (CameraInfo, 'camera_info_topic')]]
        self.sync = message_filters.ApproximateTimeSynchronizer(self.inputs, 5, 0.03)
        self.sync.registerCallback(self.receive_frame)
        self.yolo_image_pub = self.create_publisher(
            Image,
            '/debug/yolo_overlay',
            10,
        )

        self.worker_thread = threading.Thread(
            target=self.work,
            name="footprint_worker",
            daemon=True,
        )

        self.worker_thread.start()
        

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
            if self.clouds:
                camera_t = stamp_ns(image)
                closest = min(
                    self.clouds,
                    key=lambda msg: abs(stamp_ns(msg) - camera_t)
                )

                skew_ms = (
                    stamp_ns(closest) - camera_t
                ) / 1e6

                self.get_logger().info(
                    f"Nearest LiDAR timestamp difference: {skew_ms:.1f} ms"
                )
            with self.cloud_lock:
                clouds_snapshot = tuple(self.clouds)

            cloud = nearest_cloud(
                clouds_snapshot,
                stamp_ns(image),
                self.p["lidar_max_skew_s"],
            )
            uv, xyz = np.empty((0, 2)), np.empty((0, 3))
            
            if cloud is not None:
                points = point_cloud2.read_points_numpy(
                    cloud,
                    field_names=('x', 'y', 'z'),
                    skip_nans=True
                )

                # Same calibrated LiDAR-to-ZED transform
                # as the original MA2 stixel pipeline.
                uv, xyz = project_lidar(
                    points,
                    H_POINTS_ZED_FROM_LIDAR,
                    k,
                    left.shape[:2],
                )
            elif self.p['require_lidar']:
                raise ValueError('No LiDAR scan within the allowed timestamp skew')

            self.get_logger().info(
                f"LiDAR points in camera image: {len(xyz)}"
            )
            
            water_frame = self.p['water_frame']
            transform = self.lookup(water_frame, optical_frame, image.header.stamp) if water_frame else None
            result = self.backend.run(timestamp=stamp_ns(image), left_img=left, disparity_img=None,
                                      depth_img=depth, lidar_uv=uv, lidar_xyz=xyz,
                                      camera_to_water=transform)
            if result.annotated_image is not None:
                annotated = np.ascontiguousarray(result.annotated_image.copy())

                for feature in result.obstacles["features"]:
                    props = feature["properties"]

                    # Links this footprint back to its YOLO detection
                    idx = props["detection_index"]
                    x1, y1, x2, y2 = result.boat_boxes[idx].astype(int)

                    depth = props["depth_m"]
                    sources = props["depth_sources"]

                    if len(sources) == 2:
                        source = "Fused"
                    elif "lidar_depth_m" in sources:
                        source = "LiDAR"
                    else:
                        source = "Stereo"

                    label = f"{depth:.1f}m ({source})"

                    stereo = props.get("stereo_depth_m")
                    lidar = props.get("lidar_depth_m")

                    if stereo is not None:
                        label += f" | S:{stereo:.1f}m"

                    if lidar is not None:
                        label += f" | L:{lidar:.1f}m"

                    cv2.putText(
                        annotated,
                        label,
                        (x1, min(annotated.shape[0] - 10, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                msg = Image()
                msg.header = image.header
                msg.height, msg.width = annotated.shape[:2]
                msg.encoding = 'bgr8'
                msg.is_bigendian = False
                msg.step = annotated.shape[1] * 3
                msg.data = annotated.tobytes()

                self.yolo_image_pub.publish(msg)
            # The default is a synthetic local plane, not a TF optical frame.
            frame_id = water_frame or optical_frame + '/right_forward_plane'
            collection = self.backend.outputs.obstacle_collection(result.obstacles, stamp_ns(image), frame_id)
            self.publisher.publish(String(data=json.dumps(collection, allow_nan=False)))
            self.publish_obstacle_markers(
                result.obstacles,
                image.header,
                frame_id,
            )
            self.get_logger().info(
                f"YOLO detections: {len(result.boat_boxes)} | "
                f"Footprints: {len(result.obstacles['features'])} | "
                f"LiDAR image points: {len(xyz)} | "
                f"Pipeline: {result.runtime_ms:.1f} ms"
            )
        except Exception as exc:
            self.get_logger().error(f'Footprint frame rejected: {exc}')
            
    def receive_cloud(self, msg):
        with self.cloud_lock:
            self.clouds.append(msg)


    def receive_frame(self, image, depth_msg, info):
        # Keep the ROS callback lightweight.
        # The worker handles conversions, LiDAR and YOLO.
        self.slot.put((image, depth_msg, info))
        
    def publish_obstacle_markers(self, obstacles, header, frame_id):
        markers = MarkerArray()

        # Remove polygons from previous frames.
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for feature in obstacles["features"]:
            props = feature["properties"]
            ring = feature["geometry"]["coordinates"][0]
            idx = props["detection_index"]

            marker = Marker()
            marker.header.stamp = header.stamp
            marker.header.frame_id = frame_id

            marker.ns = "obstacle_footprints"
            marker.id = int(idx)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.15  # Line width in meters

            marker.color.r = 1.0
            marker.color.g = 0.4
            marker.color.b = 0.0
            marker.color.a = 1.0

            for x, y in ring:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                marker.points.append(p)

            markers.markers.append(marker)

            # Add a depth label above the polygon.
            label = Marker()
            label.header = marker.header
            label.ns = "obstacle_depth"
            label.id = int(idx)
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.pose.orientation.w = 1.0
            label.pose.position.x = float(
                sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
            )
            label.pose.position.y = float(
                sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
            )
            label.pose.position.z = 1.0

            label.scale.z = 0.8
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0

            label.text = f'{props["depth_m"]:.1f} m'

            markers.markers.append(label)

        self.marker_pub.publish(markers)


    def work(self):
        self.get_logger().info(
            "Footprint worker started"
        )

        while not self.stop_event.is_set():
            job = self.slot.take()

            if job is None:
                break

            image, depth_msg, info = job

            self.process(image, depth_msg, info)

        self.get_logger().info(
            "Footprint worker stopped"
        )
    def destroy_node(self):
        self.stop_event.set()
        self.slot.close()

        if hasattr(self, "worker_thread"):
            self.worker_thread.join(timeout=5.0)

            if self.worker_thread.is_alive():
                self.get_logger().warning(
                    "Footprint worker did not stop within 5 seconds"
                )

        return super().destroy_node()


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
