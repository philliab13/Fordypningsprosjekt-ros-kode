"""Frame synchronization, pinned selections, bounded memory and exports."""

import json
import time

import cv2
import numpy as np

from stereo_lidar_footprint_detector.frame_review import (
    PANELS, ReviewBuffer, ReviewFrame, save_frame,
)


def complete(buffer, stamp, data=b'1234'):
    for panel in PANELS:
        buffer.add(stamp, panel, data, frame_id='camera')


def test_exact_match_and_out_of_order_arrivals():
    buffer = ReviewBuffer(1000)
    buffer.add(10, 'camera', b'a')
    for panel in PANELS[1:]:
        buffer.add(11, panel, b'b')
    assert buffer.view()[0] is None
    buffer.add(11, 'camera', b'c')
    assert buffer.view()[0].timestamp_ns == 11
    complete(buffer, 9)
    assert buffer.view()[0].timestamp_ns == 11


def test_pause_step_and_wait_for_exactly_one_new_result():
    buffer = ReviewBuffer(1000)
    complete(buffer, 1)
    buffer.toggle_pause()
    complete(buffer, 2)
    assert buffer.view()[0].timestamp_ns == 1
    buffer.step(1)
    assert buffer.view()[0].timestamp_ns == 2
    buffer.step(1)
    assert buffer.view()[1]['waiting_next']
    complete(buffer, 3)
    complete(buffer, 4)
    assert buffer.view()[0].timestamp_ns == 3
    assert not buffer.view()[1]['waiting_next']
    buffer.step(-1)
    assert buffer.view()[0].timestamp_ns == 2
    buffer.set_live()
    assert buffer.view()[0].timestamp_ns == 4


def test_pinned_selection_survives_eviction_and_live_continues():
    buffer = ReviewBuffer(48)
    complete(buffer, 1)
    buffer.toggle_pause()
    for stamp in range(2, 15):
        complete(buffer, stamp)
        assert buffer.view()[1]['memory_bytes'] <= 48
        assert buffer.view()[0].timestamp_ns == 1
    buffer.set_live()
    assert buffer.view()[0].timestamp_ns == 14
    for stamp in range(15, 25):
        complete(buffer, stamp)
        assert buffer.view()[0].timestamp_ns == stamp
        assert buffer.view()[1]['memory_bytes'] <= 48


def test_timeout_pending_limit_and_oversized_records():
    buffer = ReviewBuffer(100, timeout_s=1, max_pending=2)
    now = time.monotonic()
    buffer.add(1, 'camera', b'a', now=now - 2)
    buffer.add(2, 'camera', b'b', now=now)
    assert buffer.view()[1]['last_missing'] == 'yolo, lidar, bev'
    for stamp in (3, 4, 5):
        buffer.add(stamp, 'camera', b'a')
    assert buffer.view()[1]['pending'] == 2
    complete(buffer, 6, b'x' * 101)
    assert buffer.view()[0] is None
    assert buffer.view()[1]['memory_bytes'] <= 100


def test_export_all_panels_and_timestamp_without_overwriting(tmp_path):
    image = np.full((100, 200, 3), [30, 80, 130], dtype=np.uint8)
    success, jpeg = cv2.imencode('.jpg', image)
    assert success
    frame = ReviewFrame(1234567890, {p: jpeg.tobytes() for p in PANELS},
                        {p: 'camera' for p in PANELS})
    topics = {p: '/' + p for p in PANELS}
    first = save_frame(frame, tmp_path, topics, 90)
    second = save_frame(frame, tmp_path, topics, 90)
    assert first != second
    metadata = json.loads((first / 'metadata.json').read_text())
    assert metadata['timestamp_ns'] == frame.timestamp_ns
    assert metadata['topics'] == topics
    assert metadata['buffer_encoding'] == 'JPEG'
    for panel in PANELS:
        assert cv2.imread(str(first / f'{panel}.png')).shape == image.shape
    assert cv2.imread(str(first / 'comparison.png')).shape == (960, 1280, 3)
