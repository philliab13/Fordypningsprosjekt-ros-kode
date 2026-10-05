"""Geometry and buffering helpers independent of ROS and inference packages."""
import threading

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class LatestSlot:
    """One pending frame in addition to the frame being processed."""

    def __init__(self):
        self.condition = threading.Condition()
        self.item = None
        self.closed = False
        self.dropped = 0

    def put(self, item):
        with self.condition:
            if self.closed:
                return
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.condition.notify()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.item is not None or self.closed)
            if self.closed:
                return None
            item, self.item = self.item, None
            return item

    def close(self):
        with self.condition:
            self.closed = True
            self.item = None
            self.condition.notify_all()


def transform_matrix(transform):
    q, t = transform.rotation, transform.translation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite TF transform')
    return matrix


def stereo_calibration(left_info, right_info, left_from_right, num_stixels):
    """Accept aligned, undistorted pinhole pairs, including default VRX cameras.

    These are mono CameraInfo messages: derive baseline from TF, not right P[3].
    Rectification for arbitrary real camera rigs belongs upstream of this node.
    """
    shape = (left_info.height, left_info.width)
    if shape != (right_info.height, right_info.width):
        raise ValueError('Stereo image dimensions differ')
    h, w = shape
    if h <= 50 or w < 12 or num_stixels <= 0 or w % num_stixels:
        raise ValueError('Image width must be divisible by num_stixels; height must exceed 50')
    k = np.asarray(left_info.k, dtype=float).reshape(3, 3)
    kr = np.asarray(right_info.k, dtype=float).reshape(3, 3)
    if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k, kr, atol=1e-3)
            or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6):
        raise ValueError('Expected matching calibrated pinhole CameraInfo intrinsics')
    for info in (left_info, right_info):
        if not np.isfinite(info.d).all() or np.any(np.abs(info.d) > 1e-8):
            raise ValueError('Distorted images are unsupported; provide rectified images and calibration')
        if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
            raise ValueError('Binned/cropped CameraInfo requires adjusted intrinsics')
    if (not np.isfinite(left_from_right).all()
            or not np.allclose(left_from_right[:3, :3], np.eye(3), atol=1e-3)
            or not np.allclose(left_from_right[1:3, 3], 0, atol=1e-3)):
        raise ValueError('Stereo cameras must be aligned with a horizontal optical-frame baseline')
    baseline = float(left_from_right[0, 3])
    if baseline <= 0:
        raise ValueError('Right camera must have positive optical x relative to left camera')
    return k, baseline, shape


def pipeline_pose(world_from_camera):
    """Adapt ROS camera pose to the original planar, backwards-heading convention.

    Reflect world Y so (forward,right) is a right-handed 2-D basis. Upstream
    make_homog rotates by -(yaw + pi); supply camera_heading - pi. Use the
    camera origin, so its lever arm is already included in the measured motion.
    This preserves the original planar approximation (no roll/pitch prediction).
    """
    forward = world_from_camera[:3, 2]
    if np.linalg.norm(forward[:2]) < 1e-6:
        raise ValueError('Camera forward axis is vertical')
    heading = np.arctan2(forward[1], forward[0])
    quat = Rotation.from_euler('z', heading - np.pi).as_quat()
    x, y, z = world_from_camera[:3, 3]
    return np.r_[x, -y, -z, quat]


def project_lidar(points, camera_from_lidar, k, shape):
    points = np.asarray(points).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    xyz = points @ camera_from_lidar[:3, :3].T + camera_from_lidar[:3, 3]
    xyz = xyz[xyz[:, 2] > 0]
    uvw = xyz @ k.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    h, w = shape
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return np.ascontiguousarray(uv[keep]), np.ascontiguousarray(xyz[keep])


def compute_stereo(matcher, left, right, fx, baseline):
    disparity = matcher.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16
    valid = np.isfinite(disparity) & (disparity > 0)
    disparity[~valid] = 0  # Same invalid-disparity convention as upstream SVOCamera.
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[valid] = fx * baseline / disparity[valid]
    return disparity, depth


def nearest_cloud(clouds, timestamp, max_skew_s):
    if not clouds:
        return None
    cloud = min(clouds, key=lambda msg: abs(stamp_ns(msg) - timestamp))
    return cloud if abs(stamp_ns(cloud) - timestamp) <= max_skew_s * 1e9 else None


FIELDS = ('x', 'y', 'z', 'index', 'top_px', 'base_px', 'depth', 'depth_variance',
          'valid', 'dynamic', 'propagated', 'lidar_measured')


def stixel_rows(stixels, footprints):
    """Publish every column; invalid positions are NaN, raw depth is retained."""
    n = len(footprints)
    xyz = np.column_stack((footprints[:, 1], np.zeros(n), footprints[:, 0]))
    valid = np.asarray(stixels.stixel_validity, dtype=bool) & np.isfinite(xyz).all(axis=1)
    xyz[~valid] = np.nan
    return np.column_stack((xyz, np.arange(n), stixels.height_base_list,
                            stixels.stixel_fused_depths, stixels.stixel_fused_depths_var,
                            valid, stixels.dynamic_stixel_list, stixels.using_prop_depth,
                            stixels.stixel_has_measurement)).astype(np.float32)
