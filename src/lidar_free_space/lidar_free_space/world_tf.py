"""Preserve Gazebo entity names and simulation timestamps for the WAM-V TF."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from gz.transport13 import Node as GazeboNode
from gz.msgs10.pose_v_pb2 import Pose_V
from .core import rotation_matrix


class WorldTF(Node):
    def __init__(self):
        super().__init__('lidar_world_tf')
        self.world = self.declare_parameter('world_name', 'gymkhana_task').value
        self.frame = self.declare_parameter('target_frame', self.world).value
        self.model = self.declare_parameter('model_name', 'wamv').value
        self.base_entity = self.declare_parameter('base_entity', 'wamv/base_link').value
        self.base_frame = self.declare_parameter('base_frame', 'wamv/wamv/base_link').value
        self.broadcaster = TransformBroadcaster(self)
        self.gz = GazeboNode()
        self.topic = f'/world/{self.world}/dynamic_pose/info'
        if not self.gz.subscribe(Pose_V, self.topic, self.on_pose):
            raise RuntimeError(f'Cannot subscribe to {self.topic}')

    def on_pose(self, msg):
        poses = {p.name: p for p in msg.pose}
        if self.model not in poses or self.base_entity not in poses:
            return
        model, base = poses[self.model], poses[self.base_entity]
        # Gazebo's model pose is world-relative; its link pose is model-relative.
        pos = rotation_matrix(model.orientation) @ np.array(
            [base.position.x, base.position.y, base.position.z])
        pos += [model.position.x, model.position.y, model.position.z]
        a, b = model.orientation, base.orientation
        av, bv = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z])
        xyz = a.w * bv + b.w * av + np.cross(av, bv)
        w = a.w * b.w - np.dot(av, bv)
        tf = TransformStamped()
        tf.header.stamp.sec = msg.header.stamp.sec
        tf.header.stamp.nanosec = msg.header.stamp.nsec
        tf.header.frame_id = self.frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pos)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z = map(float, xyz)
        tf.transform.rotation.w = float(w)
        self.broadcaster.sendTransform(tf)

    def destroy_node(self):
        self.gz.unsubscribe(self.topic)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WorldTF()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
