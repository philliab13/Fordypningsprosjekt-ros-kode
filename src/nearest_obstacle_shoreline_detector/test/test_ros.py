from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from nearest_obstacle_shoreline_detector.node import ShorelineNode, mask_array


class Capture:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def test_mask_decode_respects_padded_rows():
    msg = Image(height=2, width=2, step=3, encoding='mono8', data=bytes([0, 255, 99, 255, 0, 99]))
    np.testing.assert_array_equal(mask_array(msg), [[0, 255], [255, 0]])


def test_empty_barrier_clears_markers_and_geometry():
    import json
    from visualization_msgs.msg import Marker
    fake = SimpleNamespace(marker_pub=Capture(), json_pub=Capture(), active=True)
    ShorelineNode._publish(fake, Header(frame_id='water'), [], 'rejected')
    assert fake.marker_pub.messages[0].markers[0].action == Marker.DELETEALL
    assert json.loads(fake.json_pub.messages[0].data)['segments'] == []
    assert not fake.active
