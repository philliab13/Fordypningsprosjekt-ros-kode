import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from stereo_lidar_footprint_detector.images import image_array
from stereo_lidar_footprint_detector.backend import PipelineBackend
from stereo_lidar_footprint_detector.node import FootprintNode


class AdapterTests(unittest.TestCase):
    def test_image_stride_color_and_endianness(self):
        msg = SimpleNamespace(encoding='rgb8', height=1, width=1, step=4,
                              is_bigendian=False, data=bytes([1, 2, 3, 0]))
        np.testing.assert_array_equal(image_array(msg), [[[3, 2, 1]]])
        msg = SimpleNamespace(encoding='32FC1', height=1, width=1, step=8,
                              is_bigendian=True, data=np.array([5., 0.], dtype='>f4').tobytes())
        np.testing.assert_array_equal(image_array(msg), [[5.]])

    def test_registered_ros_images_to_json(self):
        repo = Path(__file__).resolve().parents[4] / 'Fordypningsprosjekt/stereo-lidar-fusion-footprint-detector'
        detector = SimpleNamespace(detect=lambda image: (np.array([[2., 2., 8., 8.]]),
                                                         None, np.array([8]), np.array([0.9])))
        backend = PipelineBackend(repo, detector=detector)
        publisher, logger = Mock(), Mock()
        node = SimpleNamespace(backend=backend, p=dict(baseline_m=0.12, lidar_max_skew_s=0.1,
                                                      require_lidar=False, water_frame=''),
                               clouds=[], publisher=publisher, get_logger=lambda: logger)
        image = Image(height=10, width=10, encoding='bgr8', step=30,
                      data=np.zeros((10, 10, 3), dtype=np.uint8).tobytes())
        depth = Image(height=10, width=10, encoding='32FC1', step=40,
                      data=np.full((10, 10), 5., dtype=np.float32).tobytes())
        info = CameraInfo(height=10, width=10)
        info.p = [100., 0., 5., 0., 0., 100., 5., 0., 0., 0., 1., 0.]
        for msg in (image, depth, info):
            msg.header.frame_id = 'camera_optical'
            msg.header.stamp.sec = 1
        FootprintNode.process(node, image, depth, info)
        logger.error.assert_not_called()
        result = json.loads(publisher.publish.call_args.args[0].data)
        self.assertEqual(result['frame_id'], 'camera_optical/right_forward_plane')
        self.assertEqual(result['timestamp_ns'], 1000000000)
        self.assertAlmostEqual(result['features'][0]['properties']['side_length_m'], 0.3)


if __name__ == '__main__':
    unittest.main()
