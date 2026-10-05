import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from stixel_free_space.node import StixelNode


class Capture:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class Backend:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def configure(self, k, baseline, shape, camera_height):
        self.calibration = k, baseline, shape, camera_height

    def synchronize(self):
        pass

    def reset(self):
        self.resets += 1

    def run(self, **kwargs):
        self.calls.append(kwargs)
        stixels = SimpleNamespace(
            stixel_width=10, height_base_list=np.tile([20, 60], (32, 1)),
            stixel_validity=np.arange(32) % 2 == 0,
            stixel_fused_depths=np.full(32, 10.), stixel_fused_depths_var=np.full(32, 0.1),
            dynamic_stixel_list=np.zeros(32, dtype=bool), using_prop_depth=np.zeros(32, dtype=bool),
            stixel_has_measurement=np.ones(32, dtype=bool))
        result = SimpleNamespace(stixel_footprints=np.column_stack((np.full(32, 10.), np.zeros(32))),
                                 water_mask=np.zeros((80, 320), dtype=np.uint8),
                                 water_mask_failure=False, runtime_ms=1.0)
        return result, stixels


@pytest.fixture
def node():
    rclpy.init()
    backend = Backend()
    node = StixelNode(backend=backend, start_worker=False, parameter_overrides=[
        Parameter('num_stixels', value=32), Parameter('num_disparities', value=16),
        Parameter('warmup_frames', value=0), Parameter('tf_wait_s', value=0.01)])
    for attr in ('cloud_pub', 'markers_pub', 'overlay_pub', 'water_pub', 'depth_pub', 'diagnostic_pub'):
        setattr(node, attr, Capture())
    # Left optical origin at world (0,0,1.5), looking +X; right camera is +optical X.
    optical_rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    world = TransformStamped()
    world.header.frame_id, world.child_frame_id = 'gymkhana_task', 'left'
    world.transform.translation.z = 1.5
    q = Rotation.from_matrix(optical_rotation).as_quat()
    world.transform.rotation.x, world.transform.rotation.y, world.transform.rotation.z, world.transform.rotation.w = map(float, q)
    node.buffer.set_transform_static(world, 'test')
    for frame, offset in [('right', 0.2), ('lidar', 0.0)]:
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'left', frame
        tf.transform.translation.x = offset
        tf.transform.rotation.w = 1.
        node.buffer.set_transform_static(tf, 'test')
    node.matcher = SimpleNamespace(compute=lambda a, b: np.full((80, 320), 160, dtype=np.int16))
    yield node
    node.destroy_node()
    rclpy.shutdown()


def job(node, lidar=True, lidar_offset_ns=0):
    stamp = node.get_clock().now().to_msg()
    images, infos = [], {}
    for side in ('left', 'right'):
        msg = node.bridge.cv2_to_imgmsg(np.zeros((80, 320, 3), dtype=np.uint8), 'bgr8')
        msg.header = Header(stamp=stamp, frame_id=side)
        images.append(msg)
        info = CameraInfo(header=msg.header, height=80, width=320)
        info.k = [100., 0., 160., 0., 100., 40., 0., 0., 1.]
        infos[side] = info
    if lidar:
        from rclpy.time import Time
        lidar_stamp = Time(nanoseconds=node.get_clock().now().nanoseconds + lidar_offset_ns).to_msg()
        cloud = point_cloud2.create_cloud_xyz32(Header(stamp=lidar_stamp, frame_id='lidar'),
                                                [[1., 0., 10.], [0., 0., -10.]])
        node.receive_cloud(cloud)
    return (*images, infos, time.perf_counter())


def test_adapter_inputs_outputs_and_timing(node):
    sample = job(node)
    node.process_job(sample)
    call = node.backend.calls[0]
    np.testing.assert_allclose(call['lidar_uv'], [[170., 40.]])
    np.testing.assert_allclose(call['lidar_xyz'], [[1., 0., 10.]])
    np.testing.assert_allclose(call['depth_img'], 2.)
    assert node.backend.calibration[1] == 0.2
    assert node.backend.calibration[3] == 1.5
    cloud = node.cloud_pub.messages[-1]
    assert cloud.header == sample[0].header and cloud.width == 32
    rows = point_cloud2.read_points(cloud)
    assert np.isnan(rows['x'][1]) and rows['valid'][1] == 0
    markers = node.markers_pub.messages[-1].markers
    assert markers[0].action == Marker.DELETEALL
    assert len(markers) == 17  # only the 16 valid columns plus clear
    assert node.water_pub.messages[-1].header == sample[0].header
    node.publish_diagnostics()
    values = {v.key: v.value for v in node.diagnostic_pub.messages[-1].status[0].values}
    assert values['processed_frames'] == '1' and values['valid_stixels'] == '16'
    assert 'pipeline_ms_p95' in values and float(values['stereo_ms']) >= 0


def test_missing_lidar_skips_by_default(node):
    node.process_job(job(node, lidar=False))
    assert not node.backend.calls and not node.cloud_pub.messages
    assert node.metrics['skipped_frames'] == 1


def test_stale_lidar_is_not_reused(node):
    node.process_job(job(node, lidar_offset_ns=-1_000_000_000))
    assert not node.backend.calls and node.metrics['skipped_frames'] == 1


def test_lidar_is_compensated_between_scan_and_image_timestamps(node):
    from rclpy.time import Time
    from tf2_ros import Buffer
    sample = job(node, lidar=False)
    camera_time = Time.from_msg(sample[0].header.stamp)
    scan_time = Time(nanoseconds=camera_time.nanoseconds-50_000_000)
    node.buffer = Buffer()
    q = Rotation.from_matrix(np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])).as_quat()
    for stamp, x in [(scan_time, 0.), (camera_time, 2.)]:
        tf = TransformStamped()
        tf.header = Header(stamp=stamp.to_msg(), frame_id='gymkhana_task')
        tf.child_frame_id = 'left'
        tf.transform.translation.x, tf.transform.translation.z = x, 1.5
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
        node.buffer.set_transform(tf, 'test')
    for frame, offset in [('right', 0.2), ('lidar', 0.)]:
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'left', frame
        tf.transform.translation.x = offset
        tf.transform.rotation.w = 1.
        node.buffer.set_transform_static(tf, 'test')
    node.receive_cloud(point_cloud2.create_cloud_xyz32(
        Header(stamp=scan_time.to_msg(), frame_id='lidar'), [[0., 0., 10.]]))
    node.process_job(sample)
    np.testing.assert_allclose(node.backend.calls[0]['lidar_xyz'], [[0., 0., 8.]], atol=1e-8)


def test_optional_lidar_passes_empty_arrays(node):
    node.p['require_lidar'] = False
    node.process_job(job(node, lidar=False))
    call = node.backend.calls[0]
    assert call['lidar_uv'].shape == (0, 2) and call['lidar_xyz'].shape == (0, 3)


def test_missing_tf_skips_without_assumed_pose(node):
    sample = job(node)
    sample[0].header.frame_id = 'unknown'
    sample[2]['left'].header.frame_id = 'unknown'
    node.process_job(sample)
    assert not node.backend.calls and node.metrics['skipped_frames'] == 1


def test_old_image_is_rejected_before_inference(node):
    sample = job(node)
    sample[0].header.stamp.sec -= 2
    node.process_job(sample)
    assert not node.backend.calls and node.metrics['skipped_frames'] == 1


def test_nonincreasing_timestamp_resets_temporal_state(node):
    sample = job(node)
    node.process_job(sample)
    node.process_job(sample)
    assert node.backend.resets == 1
    assert node.metrics['temporal_resets'] == 1
