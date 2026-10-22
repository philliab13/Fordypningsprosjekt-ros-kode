"""Exercise real ROS image decoding and Qt controls without a display server."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('PyQt5')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5.QtWidgets import QApplication
from sensor_msgs.msg import Image

from stereo_lidar_footprint_detector import frame_review_node
from stereo_lidar_footprint_detector.frame_review import PANELS, ReviewBuffer, decode_panel


def test_opencv_qt_environment_is_replaced(monkeypatch):
    from PyQt5.QtCore import QCoreApplication, QLibraryInfo

    monkeypatch.setenv('QT_QPA_PLATFORM_PLUGIN_PATH', '/venv/site-packages/cv2/qt/plugins')
    monkeypatch.setenv('QT_QPA_FONTDIR', '/venv/site-packages/cv2/qt/fonts')
    frame_review_node.configure_qt_plugins()
    assert os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] == QLibraryInfo.location(QLibraryInfo.PluginsPath)
    assert 'QT_QPA_FONTDIR' not in os.environ
    assert all('/cv2/qt/' not in path for path in QCoreApplication.libraryPaths())
    assert os.environ['QT_QPA_PLATFORM'] == 'offscreen'


def test_ros_panels_and_window_controls(monkeypatch, tmp_path):
    buffer = ReviewBuffer(1_000_000)
    node = SimpleNamespace(
        p=dict(jpeg_quality=90, memory_limit_mb=1, snapshot_directory=str(tmp_path)),
        buffer=buffer, topics={p: '/' + p for p in PANELS}, last_error='')
    for stamp in (1, 2):
        for panel in PANELS:
            msg = Image(height=20, width=40, encoding='rgb8', step=120)
            msg.header.stamp.sec = stamp
            msg.header.frame_id = 'camera'
            msg.data = np.tile(np.uint8([150, 80, 30]), (20, 40, 1)).tobytes()
            frame_review_node.FrameReviewNode.receive(node, panel, msg)
    image = decode_panel(buffer.view()[0].images['camera'])
    np.testing.assert_allclose(image[0, 0], [30, 80, 150], atol=3)
    monkeypatch.setattr(frame_review_node.rclpy, 'ok', lambda: True)
    frame_review_node.configure_qt_plugins()
    app = QApplication.instance() or QApplication([])
    window = frame_review_node.create_window(node)
    window.show()
    app.processEvents()
    window.pause.click()
    assert not buffer.view()[1]['live']
    window.previous.click()
    window.refresh()
    assert window.displayed == (1_000_000_000, 'lidar')
    window.camera_choice.setCurrentIndex(1)
    window.refresh()
    assert window.displayed == (1_000_000_000, 'yolo')
    window.next.click()
    window.refresh()
    assert window.displayed == (2_000_000_000, 'yolo')
    window.export()
    saved = window.saving.result(timeout=5)
    window.refresh()
    assert (saved / 'comparison.png').exists()
    window.live_button.click()
    assert buffer.view()[1]['live']
    window.close()
