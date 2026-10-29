"""First-test ROS adapter: registered optical depth + rectified image + LiDAR."""

import json
import threading
from copy import deepcopy
from collections import deque

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
from stixel_free_space.core import (
    LatestSlot,
    nearest_cloud,
    project_lidar,
    stamp_ns,
    transform_matrix,
)
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
from .transforms import camera_from_lidar

from .backend import PipelineBackend
from .images import image_array
from .visualization import (
    publish_obstacle_markers,
    publish_obstacle_overlay,
    publish_yolo_overlay,
)


class FootprintNode(Node):
    def __init__(self):
        super().__init__("stereo_lidar_footprint_detector")
        defaults = dict(  # noqa: C408
            baseline_m=0.0,
            weights="",
            device="",
            left_topic="/zed/zed_node/rgb/color/rect/image", #"/ma2/zed/left/image", #Subscribe from SVO_player
            depth_topic="/zed/zed_node/depth/depth_registered", #"/ma2/zed/depth", #Subscribe from SVO_player
            camera_info_topic="/zed/zed_node/rgb/color/rect/camera_info", #/ma2/zed/camera_info", #Subscribe from SVO_player
            lidar_topic="/front_lidar/ouster/points",# or "/back_lidar/ouster/points" Sim ->  #"/lidar_aft/points", #Subscribe from rosbag
            output_topic="/obstacles/json", #Output obstacles here
            water_frame="ma2_water_plane", #Has a publisher publishing these transforms
            max_stereo_depth_m=20.0,
            lidar_max_skew_s=0.5,
            require_lidar=False,
            fixed_frame="", # Stationary TF frame for scan-to-image motion compensation
        )
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if not np.isfinite(self.p["baseline_m"]) or self.p["baseline_m"] <= 0:
            raise ValueError("Set baseline_m to the actual stereo baseline in meters")
        self.backend = PipelineBackend(
            yolo_weights=self.p["weights"] or None,
            device=self.p["device"] or None,
            max_stereo_depth=self.p["max_stereo_depth_m"],
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.clouds = deque(maxlen=100)
        self.cloud_lock = threading.Lock()
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.publisher = self.create_publisher(String, self.p["output_topic"], 10)
        self.mask_pub = self.create_publisher(Image, "/obstacles/boat_mask", 10)
        self.source_image_pub = self.create_publisher(Image, "/obstacles/source_image", 10)
        self.bev_info_pub = self.create_publisher(CameraInfo, "/obstacles/camera_info", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/obstacles/markers",
            10,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.p["lidar_topic"],
            self.receive_cloud,
            qos_profile_sensor_data,
        )
        self.inputs = [
            message_filters.Subscriber(
                self, kind, self.p[name], qos_profile=qos_profile_sensor_data
            )
            for kind, name in [
                (Image, "left_topic"),
                (Image, "depth_topic"),
                (CameraInfo, "camera_info_topic"),
            ]
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(self.inputs, 5, 0.03)
        self.sync.registerCallback(self.receive_frame)
        self.yolo_image_pub = self.create_publisher(
            Image,
            "/debug/yolo_overlay",
            10,
        )
        self.obstacle_image_pub = self.create_publisher(
            Image, '/debug/obstacle_overlay', 10)

        self.worker_thread = threading.Thread(
            target=self.work,
            name="footprint_worker",
            daemon=True,
        )

        self.worker_thread.start()

    def lookup(self, target, source, stamp):
        return transform_matrix(
            self.tf_buffer.lookup_transform(
                target, source, Time.from_msg(stamp), timeout=Duration(seconds=0.0)
            ).transform
        )

    def process(self, image, depth_msg, info):
        try:
            optical_frame = image.header.frame_id
            if (
                not optical_frame
                or depth_msg.header.frame_id != optical_frame
                or info.header.frame_id != optical_frame
            ):
                raise ValueError(
                    "Image, depth and CameraInfo must share a registered optical frame"
                )
            if depth_msg.encoding != "32FC1":
                raise ValueError("Expected 32FC1 optical Z depth in meters")
            left = image_array(image)
            depth = image_array(depth_msg)
            # P describes the rectified image; raw/distorted streams must be rectified upstream.
            k = np.asarray(info.p, dtype=float).reshape(3, 4)[:, :3]
            if (info.height, info.width) != left.shape[:2]:
                raise ValueError("CameraInfo resolution differs from the image")
            if (
                info.binning_x > 1
                or info.binning_y > 1
                or info.roi.x_offset
                or info.roi.y_offset
            ):
                raise ValueError("Cropped/binned streams require adjusted CameraInfo")
            self.backend.configure(k, self.p["baseline_m"], left.shape[:2])
            # Take a thread-safe snapshot of the LiDAR buffer
            with self.cloud_lock:
                clouds_snapshot = tuple(self.clouds)

            camera_t = stamp_ns(image)

            # Log timestamp difference using the snapshot
            if clouds_snapshot:
                closest = min(
                    clouds_snapshot, key=lambda msg: abs(stamp_ns(msg) - camera_t)
                )

                skew_ms = (stamp_ns(closest) - camera_t) / 1e6

                self.get_logger().info(
                    f"Nearest LiDAR timestamp difference: {skew_ms:.1f} ms"
                )

            # Find the closest LiDAR scan within the allowed tolerance
            cloud = nearest_cloud(
                clouds_snapshot,
                camera_t,
                self.p["lidar_max_skew_s"],
            )
            uv, xyz = np.empty((0, 2)), np.empty((0, 3))

            if cloud is not None:
                points = point_cloud2.read_points_numpy(
                    cloud, field_names=("x", "y", "z"), skip_nans=True
                )

                uv, xyz = project_lidar(
                    points,
                    camera_from_lidar(
                        self.tf_buffer, image.header, cloud.header,
                        self.p["fixed_frame"],
                    ),
                    k,
                    left.shape[:2],
                )
            elif self.p["require_lidar"]:
                raise ValueError("No LiDAR scan within the allowed timestamp skew")

            self.get_logger().info(f"LiDAR points in camera image: {len(xyz)}")

            water_frame = self.p["water_frame"]
            transform = (
                self.lookup(water_frame, optical_frame, image.header.stamp)
                if water_frame
                else None
            )
            result = self.backend.run(
                timestamp=stamp_ns(image),
                left_img=left,
                disparity_img=None,
                depth_img=depth,
                lidar_uv=uv,
                lidar_xyz=xyz,
                camera_to_water=transform,
            )
            if transform is not None and result.obstacles["features"]:
                # Reconstruct the same bottom corners used by create_obstacles,
                # retaining their height before projection onto water-frame z=0.
                inverse_k = np.linalg.inv(k)
                heights = []
                height, width = left.shape[:2]
                for feature in result.obstacles["features"]:
                    props = feature["properties"]
                    index = props["detection_index"]
                    x1, _, x2, y2 = np.clip(
                        result.boat_boxes[index],
                        [0, 0, 0, 0], [width, height, width, height])
                    rays = np.array([[x1, y2, 1.0], [x2, y2, 1.0]]) @ inverse_k.T
                    camera_points = rays * (props["depth_m"] / rays[:, 2:3])
                    water_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
                    heights.append(
                        f"detection {index}: left={water_points[0, 2]:+.3f} m, "
                        f"right={water_points[1, 2]:+.3f} m "
                        f"(depth={props['depth_m']:.2f} m)")
                self.get_logger().info(
                    f"Bottom-corner heights in {water_frame} before flattening to z=0 "
                    "(negative = below plane): " + "; ".join(heights),
                    throttle_duration_sec=1.0,
                )
            publish_yolo_overlay(
                self.yolo_image_pub,
                result,
                image.header,
            )
            # The default is a synthetic local plane, not a TF optical frame.
            frame_id = water_frame or optical_frame + "/right_forward_plane"
            collection = self.backend.outputs.obstacle_collection(
                result.obstacles, stamp_ns(image), frame_id
            )
            # Publish the actual segmentation used by this result, rather than
            # reconstructing a mask from an annotated RGB image or bounding box.
            mask = np.ascontiguousarray(result.boat_mask > 0, dtype=np.uint8)
            mask_msg = Image()
            mask_msg.header = image.header
            mask_msg.height, mask_msg.width = mask.shape
            mask_msg.encoding = "mono8"
            mask_msg.step = mask_msg.width
            mask_msg.data = mask.tobytes()
            matched_info = deepcopy(info)
            matched_info.header = image.header
            self.mask_pub.publish(mask_msg)
            self.source_image_pub.publish(image)
            self.bev_info_pub.publish(matched_info)
            self.publisher.publish(String(data=json.dumps(collection, allow_nan=False)))
            publish_obstacle_overlay(
                self.obstacle_image_pub, left, result.obstacles,
                result.boat_boxes, k, image.header, transform)
            publish_obstacle_markers(
                self.marker_pub,
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
            self.get_logger().error(f"Footprint frame rejected: {exc}")

    def receive_cloud(self, msg):
        with self.cloud_lock:
            self.clouds.append(msg)

    def receive_frame(self, image, depth_msg, info):
        # Keep the ROS callback lightweight.
        # The worker handles conversions, LiDAR and YOLO.
        self.slot.put((image, depth_msg, info))

    def work(self):
        self.get_logger().info("Footprint worker started")

        while not self.stop_event.is_set():
            job = self.slot.take()

            if job is None:
                break

            image, depth_msg, info = job

            self.process(image, depth_msg, info)

        self.get_logger().info("Footprint worker stopped")

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
