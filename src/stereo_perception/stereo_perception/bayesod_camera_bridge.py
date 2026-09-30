"""Forward the latest Gazebo camera frame to the BayesOD container."""
import threading
import time
import urllib.error
import urllib.request

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class BayesODCameraBridge(Node):
    """Send frames in a worker so inference never blocks ROS callbacks."""

    def __init__(self):
        super().__init__('bayesod_camera_bridge')
        topic = self.declare_parameter(
            'image_topic',
            '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw'
        ).value
        self.endpoint = self.declare_parameter(
            'endpoint', 'http://127.0.0.1:8081/frame'
        ).value
        self.send_fps = float(self.declare_parameter('send_fps', 10.0).value)
        if not 0.0 < self.send_fps <= 60.0:
            raise ValueError('send_fps must be in (0, 60]')
        self.bridge = CvBridge()
        self.condition = threading.Condition()
        self.latest = None
        self.stopping = threading.Event()
        self.subscription = self.create_subscription(
            Image, topic, self.receive, qos_profile_sensor_data
        )
        self.worker = threading.Thread(target=self.send_frames, daemon=True)
        self.worker.start()
        self.get_logger().info(f'Forwarding {topic} to {self.endpoint}')

    def receive(self, message):
        """Replace queued frames instead of accumulating delay."""
        with self.condition:
            self.latest = message
            self.condition.notify()

    def send_frames(self):
        last_warning = float('-inf')
        while not self.stopping.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.latest is not None or self.stopping.is_set()
                )
                if self.stopping.is_set():
                    break
                message, self.latest = self.latest, None
            started = time.monotonic()
            try:
                frame = self.bridge.imgmsg_to_cv2(message, 'bgr8')
                ok, jpeg = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                request = urllib.request.Request(
                    self.endpoint, data=jpeg.tobytes(),
                    headers={'Content-Type': 'image/jpeg'}, method='POST'
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    response.read()
            except Exception as exc:
                if time.monotonic() - last_warning >= 5:
                    self.get_logger().warning(f'Camera forwarding failed: {exc}')
                    last_warning = time.monotonic()
            self.stopping.wait(max(
                0.0, 1.0 / self.send_fps - (time.monotonic() - started)
            ))

    def close(self):
        """Stop and join the sender before destroying the ROS node."""
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        self.worker.join()


def main(args=None):
    rclpy.init(args=args)
    node = BayesODCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
