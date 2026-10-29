"""Resolve LiDAR points into the image optical frame using TF."""

from rclpy.duration import Duration
from rclpy.time import Time
from stixel_free_space.core import transform_matrix


def camera_from_lidar(buffer, image_header, cloud_header, fixed_frame=''):
    """Use image-time mounting TF, optionally compensating scan-to-image motion.

    Without a stationary fixed frame, points are treated as if captured at the
    image timestamp. Missing TF is an error; no vehicle-specific fallback exists.
    """
    if not image_header.frame_id or not cloud_header.frame_id:
        raise ValueError('Camera and LiDAR messages must have nonempty frame_id')
    target_time = Time.from_msg(image_header.stamp)
    if fixed_frame:
        tf = buffer.lookup_transform_full(
            image_header.frame_id, target_time,
            cloud_header.frame_id, Time.from_msg(cloud_header.stamp),
            fixed_frame, timeout=Duration(seconds=0))
    else:
        tf = buffer.lookup_transform(
            image_header.frame_id, cloud_header.frame_id, target_time,
            timeout=Duration(seconds=0))
    return transform_matrix(tf.transform)
