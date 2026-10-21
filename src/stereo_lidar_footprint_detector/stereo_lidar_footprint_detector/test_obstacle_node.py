"""Publish a synthetic boat through the detector's existing output interfaces."""

import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, String
from visualization_msgs.msg import MarkerArray

from .visualization import publish_obstacle_markers


class TestObstacleNode(Node):
    def __init__(self):
        super().__init__('test_obstacle_publisher')
        defaults = dict(
            output_topic='/obstacles/json',
            frame_id='ma2_water_plane',
            distance_m=10.0,
            side_length_m=2.0,
            forward_yaw_deg=180.0,
            publish_rate_hz=5.0,
        )
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        for name in ('distance_m', 'side_length_m', 'publish_rate_hz'):
            if not math.isfinite(self.p[name]) or self.p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.p['forward_yaw_deg']):
            raise ValueError('forward_yaw_deg must be finite')
        if not self.p['frame_id'].strip():
            raise ValueError('frame_id must not be empty')
        self.publisher = self.create_publisher(String, self.p['output_topic'], 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacles/markers', 10)
        self.timer = self.create_timer(1.0 / self.p['publish_rate_hz'], self.publish)
        self.get_logger().info(
            f"Publishing test boat on {self.p['output_topic']}: "
            f"centre {self.p['distance_m']} m ahead, side {self.p['side_length_m']} m, "
            f"frame {self.p['frame_id']}, heading {self.p['forward_yaw_deg']} deg")

    def publish(self):
        now = self.get_clock().now()
        yaw = math.radians(self.p['forward_yaw_deg'])
        forward = (math.cos(yaw), math.sin(yaw))
        left = (-forward[1], forward[0])
        half = self.p['side_length_m'] / 2.0
        distance = self.p['distance_m']
        ring = [
            [(distance + along * half) * forward[i] + across * half * left[i]
             for i in range(2)]
            for along, across in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
        ]
        collection = dict(
            type='FeatureCollection', timestamp_ns=now.nanoseconds,
            frame_id=self.p['frame_id'], units='m',
            features=[dict(
                type='Feature',
                properties=dict(
                    id='obstacle_0000', Type='detected obstacle', detection_index=0,
                    side_length_m=self.p['side_length_m'],
                    area_m2=self.p['side_length_m'] ** 2,
                    depth_m=distance, depth_variance_m2=None,
                    depth_sources=['synthetic'], class_id=8,
                    class_name='boat', confidence=1.0,
                ),
                geometry=dict(type='Polygon', coordinates=[ring]),
            )],
        )
        self.publisher.publish(String(data=json.dumps(collection, allow_nan=False)))
        header = Header(stamp=now.to_msg(), frame_id=self.p['frame_id'])
        publish_obstacle_markers(self.marker_pub, collection, header, header.frame_id)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TestObstacleNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
