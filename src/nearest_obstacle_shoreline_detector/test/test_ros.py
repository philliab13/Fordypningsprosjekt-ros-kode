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
    fake = SimpleNamespace(marker_pub=Capture(), json_pub=Capture(), active=True,
                           p=dict(hold_last_boundary=False))
    ShorelineNode._publish(fake, Header(frame_id='water'), [], 'rejected')
    assert fake.marker_pub.messages[0].markers[0].action == Marker.DELETEALL
    assert json.loads(fake.json_pub.messages[0].data)['segments'] == []
    assert not fake.active


def test_hold_last_barrier_ignores_empty_results_and_timeout():
    fake = SimpleNamespace(marker_pub=Capture(), json_pub=Capture(), active=False,
        p=dict(hold_last_boundary=True, marker_width_m=.15, input_timeout_s=5.))
    header = Header(frame_id='water')
    segments = [[[0., 1., 0.], [2., 1., 0.]]]
    ShorelineNode._publish(fake, header, segments, 'ok')
    marker = fake.marker_pub.messages[-1].markers[-1]
    assert marker.lifetime.sec == 0 and marker.lifetime.nanosec == 0
    for status in ('no supported segments', 'No timestamp-matched LiDAR scan', 'input timeout'):
        ShorelineNode._publish(fake, header, [], status)
    assert len(fake.marker_pub.messages) == 1
    assert len(fake.json_pub.messages) == 1
    assert fake.active
    ShorelineNode._publish(fake, header, [[[1., 1., 0.], [3., 1., 0.]]], 'ok')
    assert len(fake.marker_pub.messages) == 2
