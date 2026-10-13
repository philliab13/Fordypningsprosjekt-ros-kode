"""ROS output conversion without inference dependencies."""
import cv2
import numpy as np
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from .core import FIELDS, stixel_rows


def cloud_message(header, stixels, footprints):
    fields = [PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(FIELDS)]
    return point_cloud2.create_cloud(header, fields, stixel_rows(stixels, footprints))


def color(stixels, i):
    if stixels.dynamic_stixel_list[i]:
        return (0, 0, 255)
    if stixels.using_prop_depth[i]:
        return (0, 220, 255)
    return (255, 160, 0)


def overlay_image(left, result, stixels):
    overlay = left.copy()
    h, w = left.shape[:2]
    for i, (top, base) in enumerate(stixels.height_base_list):
        if not stixels.stixel_validity[i]:
            continue
        x0, x1 = i * stixels.stixel_width, (i + 1) * stixels.stixel_width - 1
        top, base = np.clip([top, base], 0, h - 1)
        if base > top:
            cv2.rectangle(overlay, (x0, int(top)), (min(x1, w-1), int(base)), color(stixels, i), -1)
    out = cv2.addWeighted(left, 0.5, overlay, 0.5, 0)
    if result.water_mask_failure:
        cv2.putText(out, 'SEGMENTATION REJECTED - STIXELS INVALID', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def marker_message(header, stixels, k, lifetime):
    # Separate stixels; never connect across missing columns or imply a free polygon.
    clear = Marker(header=header, action=Marker.DELETEALL)
    markers = [clear]
    for i, (top, base) in enumerate(stixels.height_base_list):
        z = stixels.stixel_fused_depths[i]
        if not stixels.stixel_validity[i] or not np.isfinite(z) or z <= 0:
            continue
        marker = Marker(header=header, ns='stixels', id=i, type=Marker.LINE_LIST, action=Marker.ADD)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06
        b, g, r = color(stixels, i)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r/255, g/255, b/255, 1.0
        marker.lifetime = Duration(seconds=lifetime).to_msg()
        u = (i + 1) * stixels.stixel_width - stixels.stixel_width // 2
        marker.points = [Point(x=float(z * (u - k[0, 2]) / k[0, 0]),
                               y=float(z * (v - k[1, 2]) / k[1, 1]), z=float(z))
                         for v in (top, base)]
        markers.append(marker)
    return MarkerArray(markers=markers)
