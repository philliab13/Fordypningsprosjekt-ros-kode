"""ROS2 visualization helpers for YOLO depth overlays and water-plane footprints."""

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray


def publish_obstacle_overlay(publisher, image, obstacles, boxes, k, header,
                             camera_to_water=None):
    """Project the JSON footprint rings onto the matching rectified image.

    A configured water frame uses z=0. The synthetic right/forward plane
    omits height, so recover it from each detection's bottom edge and depth.
    """
    import cv2
    import numpy as np

    annotated = np.ascontiguousarray(image.copy())
    water_to_camera = (np.linalg.inv(camera_to_water)
                       if camera_to_water is not None else None)
    inverse_k = np.linalg.inv(k)
    height, width = annotated.shape[:2]
    for feature in obstacles['features']:
        props = feature['properties']
        ring = np.asarray(feature['geometry']['coordinates'][0], dtype=float)
        if water_to_camera is None:
            x1, _, x2, y2 = np.clip(
                boxes[props['detection_index']],
                [0, 0, 0, 0], [width, height, width, height])
            ray = inverse_k @ np.array([(x1 + x2) / 2, y2, 1.0])
            camera_y = ray[1] * props['depth_m'] / ray[2]
            points = np.column_stack((ring[:, 0],
                                      np.full(len(ring), camera_y), ring[:, 1]))
        else:
            plane = np.column_stack((ring, np.zeros(len(ring)), np.ones(len(ring))))
            points = (plane @ water_to_camera.T)[:, :3]
        if not np.isfinite(points).all():
            continue
        for start, end in zip(points[:-1], points[1:]):
            start, end = start.copy(), end.copy()
            # Clip to the camera near plane before perspective division.
            near = 0.05
            if start[2] < near and end[2] < near:
                continue
            if start[2] < near:
                start += (end - start) * ((near - start[2]) / (end[2] - start[2]))
            if end[2] < near:
                end += (start - end) * ((near - end[2]) / (start[2] - end[2]))
            pixels = np.array([start, end]) @ k.T
            pixels = np.clip(pixels[:, :2] / pixels[:, 2:3], -1e7, 1e7).astype(int)
            visible, a, b = cv2.clipLine((0, 0, width, height),
                                         tuple(pixels[0]), tuple(pixels[1]))
            if visible:
                cv2.line(annotated, a, b, (0, 165, 255), 2, cv2.LINE_AA)

    msg = Image()
    msg.header = header
    msg.height, msg.width = annotated.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = annotated.tobytes()
    publisher.publish(msg)


def publish_yolo_overlay(publisher, result, header):
    """Publish annotated YOLO image with estimated obstacle depths."""
    import cv2
    import numpy as np

    if result.annotated_image is None:
        return

    annotated = np.ascontiguousarray(result.annotated_image.copy())

    for feature in result.obstacles["features"]:
        props = feature["properties"]

        # Match footprint to its original YOLO detection.
        idx = props["detection_index"]
        x1, y1, x2, y2 = result.boat_boxes[idx].astype(int)

        depth = props["depth_m"]
        sources = props["depth_sources"]

        if len(sources) == 2:
            source = "Fused"
        elif "lidar_depth_m" in sources:
            source = "LiDAR"
        else:
            source = "Stereo"

        label = f"{depth:.1f}m ({source})"

        stereo = props.get("stereo_depth_m")
        lidar = props.get("lidar_depth_m")

        if stereo is not None:
            label += f" | S:{stereo:.1f}m"

        if lidar is not None:
            label += f" | L:{lidar:.1f}m"

        cv2.putText(
            annotated,
            label,
            (x1, min(annotated.shape[0] - 10, y2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    msg = Image()
    msg.header = header
    msg.height, msg.width = annotated.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = annotated.shape[1] * 3
    msg.data = np.ascontiguousarray(annotated).tobytes()

    publisher.publish(msg)


def publish_obstacle_markers(publisher, obstacles, header, frame_id):
    """Publish water-plane obstacle polygons and depth labels for RViz."""
    markers = MarkerArray()

    # Clear obstacles from the previous frame.
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    for feature in obstacles["features"]:
        props = feature["properties"]
        ring = feature["geometry"]["coordinates"][0]
        idx = int(props["detection_index"])

        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = frame_id
        marker.ns = "obstacle_footprints"
        marker.id = idx
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15

        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.0
        marker.color.a = 1.0

        for x, y in ring:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            marker.points.append(point)

        markers.markers.append(marker)

        # Display the estimated depth above the footprint.
        label = Marker()
        label.header = marker.header
        label.ns = "obstacle_depth"
        label.id = idx
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD

        label.pose.orientation.w = 1.0
        label.pose.position.x = float(sum(p[0] for p in ring[:-1]) / (len(ring) - 1))
        label.pose.position.y = float(sum(p[1] for p in ring[:-1]) / (len(ring) - 1))
        label.pose.position.z = 1.0

        label.scale.z = 0.8
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0

        label.text = f"{props['depth_m']:.1f} m"

        markers.markers.append(label)

    publisher.publish(markers)
