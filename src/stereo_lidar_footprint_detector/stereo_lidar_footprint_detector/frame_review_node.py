"""ROS subscriber and Qt window for reviewing completed detector frames."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .frame_review import PANELS, ReviewBuffer, decode_panel, save_frame
from .images import image_array


def configure_qt_plugins():
    """Use PyQt's Qt plugins after cv2's wheel has modified the environment.

    opencv-python bundles a separate Qt build for HighGUI. Its plugin path and
    font directory must not be used by this PyQt application.
    """
    from PyQt5.QtCore import QCoreApplication, QLibraryInfo

    plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins
    font_directory = os.environ.get('QT_QPA_FONTDIR', '')
    if '/cv2/qt/' in font_directory.replace('\\', '/'):
        os.environ.pop('QT_QPA_FONTDIR', None)
    other_paths = [p for p in QCoreApplication.libraryPaths()
                   if '/cv2/qt/' not in p.replace('\\', '/') and p != plugins]
    QCoreApplication.setLibraryPaths([plugins, *other_paths])


class FrameReviewNode(Node):
    def __init__(self):
        super().__init__('frame_review')
        defaults = dict(
            camera_topic='/obstacles/source_image', yolo_topic='/debug/yolo_overlay',
            lidar_topic='/debug/bev_lidar_overlay', bev_topic='/debug/footprint_bev',
            memory_limit_mb=512, incomplete_timeout_s=10.0, max_pending_frames=30,
            jpeg_quality=90, snapshot_directory=str(Path.cwd() / 'frame_snapshots'))
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if (not 1 <= self.p['jpeg_quality'] <= 100
                or not np.isfinite(self.p['incomplete_timeout_s'])):
            raise ValueError('Invalid JPEG quality or incomplete frame timeout')
        self.buffer = ReviewBuffer(self.p['memory_limit_mb'] * 1024 * 1024,
                                   self.p['incomplete_timeout_s'], self.p['max_pending_frames'])
        self.topics = {panel: self.p[f'{panel}_topic'] for panel in PANELS}
        self.last_error = ''
        self.subscriptions_ = [self.create_subscription(
            Image, topic, partial(self.receive, panel), qos_profile_sensor_data)
            for panel, topic in self.topics.items()]

    def receive(self, panel, msg):
        try:
            image = image_array(msg)
            if image.ndim != 3:
                raise ValueError('Expected a color review image')
            success, encoded = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.p['jpeg_quality']])
            if not success:
                raise ValueError('JPEG encoding failed')
            timestamp = int(msg.header.stamp.sec) * 1_000_000_000 + msg.header.stamp.nanosec
            self.buffer.add(timestamp, panel, encoded.tobytes(), msg.header.frame_id)
        except (ValueError, cv2.error) as exc:
            self.last_error = f'{panel}: {exc}'
            self.get_logger().warning(self.last_error, throttle_duration_sec=2.0)


def create_window(node):
    # Keep Qt out of the storage/ROS module import path for headless tests.
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
        QShortcut, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
    )

    class ImagePanel(QLabel):
        def __init__(self):
            super().__init__('Waiting for a complete timestamp-matched frame…')
            self.source = None
            self.setMinimumSize(160, 120)
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet('background: #191919; color: #dddddd;')

        def display(self, bgr):
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            self.source = QPixmap.fromImage(image.copy())
            self.fit()

        def fit(self):
            if self.source is not None:
                self.setPixmap(self.source.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.fit()

    class ReviewWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Frame review — camera / BEV')
            self.resize(1500, 850)
            self.displayed = None
            self.current_frame = None
            self.save_pool = ThreadPoolExecutor(max_workers=1)
            self.saving = None
            self.export_message = ''
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            controls = QHBoxLayout()
            layout.addLayout(controls)
            self.pause = QPushButton('Pause [Space]')
            self.previous = QPushButton('Previous [←]')
            self.next = QPushButton('Next [→]')
            self.live_button = QPushButton('Live [L]')
            self.save_button = QPushButton('Save [S]')
            self.camera_choice = QComboBox()
            self.camera_choice.addItems(['LiDAR overlay', 'YOLO overlay', 'Original camera'])
            for widget in (self.pause, self.previous, self.next, self.live_button,
                           self.save_button, self.camera_choice):
                controls.addWidget(widget)
            controls.addStretch()
            self.pause.clicked.connect(node.buffer.toggle_pause)
            self.previous.clicked.connect(lambda: node.buffer.step(-1))
            self.next.clicked.connect(lambda: node.buffer.step(1))
            self.live_button.clicked.connect(lambda: node.buffer.set_live())
            self.save_button.clicked.connect(self.export)
            self.shortcuts = []
            for key, callback in [('Space', node.buffer.toggle_pause),
                                  ('Left', lambda: node.buffer.step(-1)),
                                  ('Right', lambda: node.buffer.step(1)),
                                  ('L', lambda: node.buffer.set_live()), ('S', self.export)]:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(callback)
                self.shortcuts.append(shortcut)
            self.left, self.right = ImagePanel(), ImagePanel()
            split = QSplitter(Qt.Horizontal)
            split.addWidget(self.left)
            split.addWidget(self.right)
            layout.addWidget(split, 1)
            self.status = QLabel()
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            self.details = QLabel()
            self.details.setWordWrap(True)
            layout.addWidget(self.details)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)
            self.refresh()

        def refresh(self):
            if not rclpy.ok():
                self.close()
                return
            frame, state = node.buffer.view()
            panel = ('lidar', 'yolo', 'camera')[self.camera_choice.currentIndex()]
            if frame is not None and self.displayed != (frame.timestamp_ns, panel):
                self.left.display(decode_panel(frame.images[panel]))
                self.right.display(decode_panel(frame.images['bev']))
                self.displayed = (frame.timestamp_ns, panel)
                self.current_frame = frame
            self.pause.setText('Pause [Space]' if state['live'] else 'Resume live [Space]')
            self.save_button.setEnabled(frame is not None and self.saving is None)
            mode = 'LIVE' if state['live'] else ('WAITING FOR NEXT' if state['waiting_next'] else 'PAUSED')
            timestamp = str(frame.timestamp_ns) if frame else '—'
            self.status.setText(
                f'{mode} | source timestamp: {timestamp} ns | '
                f'frame {state["index"]}/{state["count"]} | '
                f'buffer {state["memory_bytes"] / 1024**2:.1f}/{node.p["memory_limit_mb"]} MiB | '
                f'pending {state["pending"]} | discarded {state["dropped"]}')
            if self.saving is not None and self.saving.done():
                try:
                    self.export_message = f'Saved: {self.saving.result()}'
                except Exception as exc:
                    self.export_message = f'Snapshot failed: {exc}'
                self.saving = None
            missing = f'Last incomplete frame missing: {state["last_missing"]}' if state['last_missing'] else ''
            self.details.setText(' | '.join(filter(None, [missing, node.last_error, self.export_message])))

        def export(self):
            frame = self.current_frame
            if frame is not None and self.saving is None:
                self.export_message = 'Saving snapshot…'
                self.saving = self.save_pool.submit(
                    save_frame, frame, node.p['snapshot_directory'], node.topics, node.p['jpeg_quality'])

        def closeEvent(self, event):
            self.timer.stop()
            self.save_pool.shutdown(wait=True)
            super().closeEvent(event)

    return ReviewWindow()


def main(args=None):
    configure_qt_plugins()
    from PyQt5.QtWidgets import QApplication

    rclpy.init(args=args)
    node = None
    executor = None
    worker = None
    try:
        app = QApplication([sys.argv[0]])
        node = FrameReviewNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        worker = threading.Thread(target=executor.spin, daemon=True)
        worker.start()
        window = create_window(node)
        window.show()
        app.exec_()
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if worker is not None:
            worker.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
