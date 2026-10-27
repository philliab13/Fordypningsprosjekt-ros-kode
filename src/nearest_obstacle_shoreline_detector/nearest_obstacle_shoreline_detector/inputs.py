"""Small input helpers without a dependency on the stixel ROS package."""
import numpy as np
from scipy.spatial.transform import Rotation


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def subtract_boats(water, boat_msg, image_msg):
    """Use only the footprint segmentation from the exact source image."""
    if (stamp_ns(boat_msg) != stamp_ns(image_msg)
            or boat_msg.header.frame_id != image_msg.header.frame_id
            or (boat_msg.height, boat_msg.width) != water.shape):
        raise ValueError('Boat mask must match the exact image timestamp, optical frame and dimensions')
    if boat_msg.encoding not in ('mono8', '8UC1'):
        raise ValueError('Boat mask must use mono8 or 8UC1')
    if boat_msg.step < boat_msg.width or len(boat_msg.data) < boat_msg.step * boat_msg.height:
        raise ValueError('Invalid boat mask stride or buffer')
    boats = np.ndarray(water.shape, dtype=np.uint8, buffer=boat_msg.data,
                       strides=(boat_msg.step, 1))
    result = water.copy()
    result[boats != 0] = 0
    return result


def nearest_cloud(clouds, timestamp, max_skew_s):
    if not clouds:
        return None
    cloud = min(clouds, key=lambda msg: abs(stamp_ns(msg) - timestamp))
    return cloud if abs(stamp_ns(cloud) - timestamp) <= max_skew_s * 1e9 else None


def transform_matrix(transform):
    q, t = transform.rotation, transform.translation
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [t.x, t.y, t.z]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite TF transform')
    return result


def image_array(msg):
    formats = {'bgr8': (np.dtype('u1'), 3), 'rgb8': (np.dtype('u1'), 3),
               '32FC1': (np.dtype('>f4' if msg.is_bigendian else '<f4'), 1)}
    if msg.encoding not in formats:
        raise ValueError(f'Unsupported input encoding: {msg.encoding}')
    dtype, channels = formats[msg.encoding]
    pixel = dtype.itemsize * channels
    if (msg.height <= 0 or msg.width <= 0 or msg.step < msg.width * pixel
            or len(msg.data) < msg.step * msg.height):
        raise ValueError('Invalid input image dimensions or buffer')
    data = np.ndarray((msg.height, msg.width, channels), dtype=dtype, buffer=msg.data,
                      strides=(msg.step, pixel, dtype.itemsize))
    if channels == 1:
        return data[:, :, 0].astype(np.float32)
    return np.ascontiguousarray(data[:, :, ::-1] if msg.encoding == 'rgb8' else data)
