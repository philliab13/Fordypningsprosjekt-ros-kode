"""Decode supported ROS Image encodings without compiled cv_bridge bindings."""
import numpy as np


def image_array(msg):
    channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, '32FC1': 1}
    if msg.encoding not in channels:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    count = channels[msg.encoding]
    dtype = np.dtype(('>' if msg.is_bigendian else '<') + 'f4') if count == 1 else np.dtype('u1')
    pixel_size = count * dtype.itemsize
    if msg.height <= 0 or msg.width <= 0 or msg.step < msg.width * pixel_size:
        raise ValueError('Invalid image dimensions or row stride')
    if len(msg.data) < msg.step * msg.height:
        raise ValueError('Truncated image buffer')
    array = np.ndarray((msg.height, msg.width, count), dtype=dtype, buffer=msg.data,
                       strides=(msg.step, pixel_size, dtype.itemsize))
    if count == 1:
        return array[:, :, 0].astype(np.float32)
    array = array[:, :, :3]
    if msg.encoding in ('rgb8', 'rgba8'):
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)
