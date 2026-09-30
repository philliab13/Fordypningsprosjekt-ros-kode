from types import SimpleNamespace
import math
import pytest
from gz.msgs10.pose_v_pb2 import Pose_V
from lidar_free_space.world_tf import WorldTF


def test_world_pose_composition_preserves_simulation_stamp():
    messages = []
    adapter = SimpleNamespace(model='wamv', base_entity='wamv/base_link',
                              frame='gymkhana_task', base_frame='wamv/wamv/base_link',
                              broadcaster=SimpleNamespace(sendTransform=messages.append))
    msg = Pose_V()
    msg.header.stamp.sec = 42
    msg.header.stamp.nsec = 123
    model = msg.pose.add()
    model.name = 'wamv'
    model.position.x = 10
    model.orientation.z = math.sin(math.pi / 4)
    model.orientation.w = math.cos(math.pi / 4)
    base = msg.pose.add()
    base.name = 'wamv/base_link'
    base.position.x = 2
    base.orientation.w = 1
    WorldTF.on_pose(adapter, msg)
    tf = messages[0]
    assert tf.header.stamp.sec == 42
    assert tf.header.stamp.nanosec == 123
    assert tf.header.frame_id == 'gymkhana_task'
    assert tf.child_frame_id == 'wamv/wamv/base_link'
    assert tf.transform.translation.x == pytest.approx(10)
    assert tf.transform.translation.y == pytest.approx(2)
    assert tf.transform.rotation.z == pytest.approx(math.sin(math.pi / 4))
    WorldTF.on_pose(adapter, Pose_V())
    assert len(messages) == 1
