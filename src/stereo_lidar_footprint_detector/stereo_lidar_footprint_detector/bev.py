"""Local metric BEV geometry and rendering, independent of ROS and inference."""

import cv2
import numpy as np


def closest_scans(clouds, timestamp_ns, count, max_skew_s):
    """Select distinct scan timestamps by absolute distance to the SVO frame."""
    unique = {stamp: cloud for stamp, cloud in clouds
              if abs(stamp - timestamp_ns) <= max_skew_s * 1e9}
    stamps = sorted(unique, key=lambda stamp: (abs(stamp - timestamp_ns), stamp))
    return [(stamp, unique[stamp]) for stamp in stamps[:count]]


def mask_selection(uv, xyz, mask):
    """Return a selection aligned with the original projected point arrays."""
    uv, xyz = np.asarray(uv), np.asarray(xyz)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
    indices = np.flatnonzero(valid)
    pixels = np.floor(uv[valid]).astype(int)
    h, w = mask.shape
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    pixels, indices = pixels[inside], indices[inside]
    selected = np.zeros(len(uv), dtype=bool)
    selected[indices] = mask[pixels[:, 1], pixels[:, 0]] > 0
    return selected


def mask_points(uv, xyz, mask):
    """Keep projected points inside the segmentation, never just its box."""
    return np.asarray(xyz)[mask_selection(uv, xyz, mask)]


def scan_color(index):
    """Shared BGR colours for scans ranked by timestamp proximity."""
    palette = [(220, 80, 30), (190, 40, 190), (30, 160, 100),
               (40, 40, 220), (180, 140, 30), (100, 70, 160)]
    if index < len(palette):
        return palette[index]
    hsv = np.uint8([[[int((index * 68.75) % 180), 210, 220]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_scan_legend(image, legend, x=15, y=72):
    """Compact shared legend; bound its height even for large scan counts."""
    columns = max(1, min(3, image.shape[1] // 280))
    for index, (label, color) in enumerate(legend[:6]):
        left = x + (index % columns) * ((image.shape[1] - 2 * x) // columns)
        top = y + (index // columns) * 20
        cv2.circle(image, (left + 4, top - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (left + 14, top), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
    return 20 * int(np.ceil(min(len(legend), 6) / columns))


def render_lidar_overlay(rgb, scans, legend, status):
    """Draw the exact BEV scan projections: faint background, bright boat dots."""
    image = np.ascontiguousarray(rgb.copy())
    background = image.copy()
    for scan in scans:
        for point in np.floor(scan['uv'][~scan['boat']]).astype(int):
            cv2.circle(background, tuple(point), 1, scan['color'], -1, cv2.LINE_AA)
    image = cv2.addWeighted(background, 0.3, image, 0.7, 0)
    # Draw selected points last so background scans cannot hide boat returns.
    for scan in scans:
        for point in np.floor(scan['uv'][scan['boat']]).astype(int):
            cv2.circle(image, tuple(point), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(point), 3, scan['color'], -1, cv2.LINE_AA)
    columns = max(1, min(3, image.shape[1] // 280))
    panel_height = 60 + 20 * int(np.ceil(min(len(legend), 6) / columns))
    cv2.rectangle(image, (0, 0), (image.shape[1], panel_height), (25, 25, 25), -1)
    cv2.putText(image, 'BEV scans: bright = boat points; faint = background',
                (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, status, (15, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (230, 230, 230), 1, cv2.LINE_AA)
    draw_scan_legend(image, legend)
    return image


def local_geometry(camera_xyz, obstacles, camera_to_water=None):
    """Flatten in the footprint frame, then express XY relative to the camera.

    For a true water frame, drop water Z before rotating into a horizontal
    camera-right/forward basis. The fallback matches the detector's synthetic
    optical X/Z plane exactly, which is not a gravity-aligned water plane.
    """
    rings = [np.asarray(f['geometry']['coordinates'][0], dtype=float)
             for f in obstacles['features']]
    camera_xyz = np.asarray(camera_xyz, dtype=float).reshape(-1, 3)
    if camera_to_water is None:
        return camera_xyz[:, [0, 2]], rings
    transform = np.asarray(camera_to_water)
    forward = transform[:2, 2].copy()
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError('Camera forward axis is vertical in the water frame')
    forward /= norm
    # Derive right from the optical frame: MA2 water frames may be Z-down.
    # Assuming a Z-up cross product here would mirror the entire comparison.
    right = transform[:2, 0].copy()
    right -= np.dot(right, forward) * forward
    if np.linalg.norm(right) < 1e-8:
        raise ValueError('Camera right axis has no independent horizontal component')
    right /= np.linalg.norm(right)
    basis = np.column_stack((right, forward))
    origin = transform[:2, 3]
    # Orthogonal flattening preserves water-frame X/Y; no ray/plane intersection.
    water_xyz = camera_xyz @ transform[:3, :3].T + transform[:3, 3]
    return ((water_xyz[:, :2] - origin) @ basis,
            [(ring[:, :2] - origin) @ basis for ring in rings])


def render_bev(points, rings, status, size_px=900, range_m=60.0,
               point_colors=None, scan_legend=None, gnss_references=None,
               shoreline_segments=None):
    """Equal metre scales, camera origin and filled outlines, like plot_bev."""
    image = np.full((size_px, size_px, 3), 250, dtype=np.uint8)
    legend = scan_legend or []
    columns = max(1, min(3, size_px // 280))
    legend_height = 20 * int(np.ceil(min(len(legend), 6) / columns))
    margin = min(65 + legend_height, (size_px - 100) // 2)
    span = size_px - 2 * margin
    scale = span / range_m
    xmin, ymin = -range_m / 2, -range_m * 0.1

    def pixel(xy):
        xy = np.asarray(xy)
        out = np.column_stack((margin + (xy[:, 0] - xmin) * scale,
                               size_px - margin - (xy[:, 1] - ymin) * scale))
        return np.clip(out, -1e6, 1e6).astype(np.int32)

    def text(label, pos, color=(60, 60, 60), scale=0.45):
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    step = 10 ** np.floor(np.log10(range_m / 6))
    if range_m / step > 12:
        step *= 5
    for value in np.arange(np.ceil(xmin / step) * step, xmin + range_m, step):
        x = int(pixel([[value, 0]])[0, 0])
        cv2.line(image, (x, margin), (x, size_px - margin), (225, 225, 225))
        text(f'{value:g}', (x - 12, size_px - margin + 20))
    for value in np.arange(np.ceil(ymin / step) * step, ymin + range_m, step):
        y = int(pixel([[0, value]])[0, 1])
        cv2.line(image, (margin, y), (size_px - margin, y), (225, 225, 225))
        text(f'{value:g}', (15, y + 5))
    overlay = image.copy()
    for ring in rings:
        if len(ring) >= 3 and np.isfinite(ring).all():
            cv2.fillPoly(overlay, [pixel(ring)], (255, 255, 0))
    image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
    for index, ring in enumerate(rings):
        if len(ring) >= 3 and np.isfinite(ring).all():
            cv2.polylines(image, [pixel(ring)], True, (0, 140, 255), 2, cv2.LINE_AA)
            center = pixel([ring.mean(axis=0)])[0]
            if margin <= center[0] < size_px - margin and margin <= center[1] < size_px - margin:
                text(f'Footprint {index}', tuple(center), (0, 100, 200))
    points = np.asarray(points).reshape(-1, 2)
    finite = np.isfinite(points).all(axis=1)
    colors = (np.tile((180, 40, 20), (len(points), 1)) if point_colors is None
              else np.asarray(point_colors).reshape(-1, 3))
    points, colors = points[finite], colors[finite]
    visible = ((points[:, 0] >= xmin) & (points[:, 0] <= xmin + range_m)
               & (points[:, 1] >= ymin) & (points[:, 1] <= ymin + range_m))
    for point, color in zip(pixel(points[visible]), colors[visible]):
        cv2.circle(image, tuple(point), 2, tuple(int(c) for c in color), -1, cv2.LINE_AA)
    for reference in gnss_references or []:
        xy = np.asarray(reference['xy'])
        if (not np.isfinite(xy).all() or not xmin <= xy[0] <= xmin + range_m
                or not ymin <= xy[1] <= ymin + range_m):
            continue
        location = pixel([xy])[0]
        cv2.drawMarker(image, tuple(location), reference['color'],
                       cv2.MARKER_DIAMOND if reference.get('ego') else cv2.MARKER_CROSS,
                       20, 3, cv2.LINE_AA)
        text(reference['label'], (int(location[0]) + 12, int(location[1]) - 8),
             reference['color'])
    for segment in (() if shoreline_segments is None else shoreline_segments):
        segment = np.asarray(segment)
        if segment.shape == (2, 2) and np.isfinite(segment).all():
            endpoints = pixel(segment)
            visible, start, end = cv2.clipLine(
                (margin, margin, span + 1, span + 1),
                tuple(endpoints[0]), tuple(endpoints[1]))
            if visible:
                cv2.line(image, start, end, (0, 0, 220), 3, cv2.LINE_AA)
    cv2.drawMarker(image, tuple(pixel([[0, 0]])[0]), (0, 150, 0),
                   cv2.MARKER_TRIANGLE_UP, 16, 2)
    text('LiDAR by scan | Footprints (orange) | Camera (green) | GNSS (+)', (15, 25))
    text(status, (15, 46), scale=0.4)
    if shoreline_segments is not None and len(shoreline_segments):
        text('Shoreline (red)', (size_px - 145, size_px - 35), (0, 0, 220))
    for reference in gnss_references or []:
        if reference.get('ego'):
            lat, lon = reference['position_lla'][:2]
            text(f'Ego MA2 (navigation pose): {lat:.7f}, {lon:.7f} (lat, lon)',
                 (15, size_px - 55), reference['color'], scale=0.4)
            lat, lon = reference['camera_lla'][:2]
            heading = reference['camera_heading_deg']
            direction = f'{heading:.1f} deg true' if heading is not None else 'undefined'
            text(f'BEV camera origin: {lat:.7f}, {lon:.7f} | Forward: {direction}',
                 (15, size_px - 35), reference['color'], scale=0.35)
    draw_scan_legend(image, legend)
    text('Right [m]', (size_px // 2 - 30, size_px - 15))
    text('Forward [m]', (margin, margin + 15))
    text(f'{len(points)} boat points ({np.count_nonzero(visible)} in view)',
         (margin + 5, size_px - margin - 10))
    return image
