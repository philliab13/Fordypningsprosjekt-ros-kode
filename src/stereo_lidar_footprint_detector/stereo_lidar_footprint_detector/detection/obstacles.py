"""Square obstacle footprints in a local metric water plane."""

import numpy as np


def create_obstacles(
    boxes,
    depth_img,
    lidar_uv,
    lidar_xyz,
    K,
    segmentation_mask=None,
    camera_to_water=None,
    max_depth=100.0,
    instance_masks=None,
    baseline=None,
    max_stereo_depth=10.0,
):
    """Return maze-style Polygon features, with one square per valid detection.

    ``boxes`` is an (N, 4) array of image xyxy coordinates. Stereo depth and
    lidar_xyz[:, 2] must be optical-axis Z in meters, registered to this image.
    Stereo uses median valid depth; LiDAR uses the reference lower 30th
    percentile. With a baseline, fusion uses the reference inverse variances
    (sigma_lidar=0.01 m; sigma_stereo=0.5*z**2/(fx*baseline)) and stereo
    cutoff (10 m by default). Without a baseline, equal weights are used.
    Missing depth skips the detection. This is a surface-distance estimate, not a measured hull.

    The box's bottom edge is back-projected at that depth. Its horizontal
    width defines the square side; the square extends away from the camera.
    camera_to_water optionally maps optical XYZ to a frame whose XY axes span
    the water plane. By default XY means camera-right/camera-forward, assuming
    a level camera. Coordinates are local meters, NOT geographic longitude and
    latitude. Feature IDs are frame-local detection indices, not tracking IDs.
    """
    depth = np.asarray(depth_img, dtype=float)
    K = np.asarray(K, dtype=float)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    uv = np.asarray(lidar_uv, dtype=float).reshape(-1, 2)
    xyz = np.asarray(lidar_xyz, dtype=float).reshape(-1, 3)
    if depth.ndim != 2 or K.shape != (3, 3):
        raise ValueError("Expected a 2D depth image and a 3x3 intrinsic matrix")
    if len(uv) != len(xyz):
        raise ValueError("LiDAR pixel and XYZ arrays must have matching lengths")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    if not np.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("max_depth must be finite and positive")
    if baseline is not None and (not np.isfinite(baseline) or baseline <= 0):
        raise ValueError("baseline must be positive meters")
    if not np.isfinite(max_stereo_depth) or max_stereo_depth <= 0:
        raise ValueError("max_stereo_depth must be finite and positive")
    if instance_masks is not None and len(instance_masks) != len(boxes):
        raise ValueError("Expected one instance mask per detection")
    mask = None if segmentation_mask is None else np.asarray(segmentation_mask) > 0
    if mask is not None and mask.shape != depth.shape:
        raise ValueError("Segmentation mask must match the depth image")
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if camera_to_water is not None:
        transform = np.asarray(camera_to_water, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_water must be a finite 4x4 rigid transform")
        if (
            not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
            )
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("camera_to_water must be a rigid transform")

    height, width = depth.shape
    valid_lidar = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(xyz).all(axis=1)
        & (xyz[:, 2] > 0)
        & (xyz[:, 2] <= max_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    uv, xyz = uv[valid_lidar], xyz[valid_lidar]
    if mask is not None:
        on_mask = mask[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        uv, xyz = uv[on_mask], xyz[on_mask]

    if not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Expected pinhole intrinsics with final row [0, 0, 1]")
    inverse_K = np.linalg.inv(K)
    features = []
    for index, box in enumerate(boxes):
        if not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if x2 <= x1 or y2 <= y1:
            continue
        # Pixel samples inside the continuous box, with an exclusive upper edge.
        left, top, right, bottom = np.ceil([x1, y1, x2, y2]).astype(int)
        samples = depth[top:bottom, left:right]
        valid = np.isfinite(samples) & (samples > 0) & (samples <= max_depth)
        object_mask = mask
        if instance_masks is not None:
            object_mask = np.asarray(instance_masks[index]) > 0
            if object_mask.shape != depth.shape:
                raise ValueError("Instance masks must match the depth image")
        if object_mask is not None:
            valid &= object_mask[top:bottom, left:right]
        stereo = samples[valid]
        inside = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
        if object_mask is not None:
            inside &= object_mask[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        lidar = xyz[inside, 2]
        estimates = {}
        if stereo.size:
            median = float(np.median(stereo))
            if median <= max_stereo_depth:
                estimates["stereo_depth_m"] = median
        if lidar.size:
            estimates["lidar_depth_m"] = float(
                np.sort(lidar)[int(0.3 * (len(lidar) - 1))]
            )
        if not estimates:
            continue
        variances = (
            {
                source: (
                    0.01**2
                    if source == "lidar_depth_m"
                    else (0.5 * value**2 / (K[0, 0] * baseline)) ** 2
                )
                for source, value in estimates.items()
            }
            if baseline is not None
            else {source: 1.0 for source in estimates}
        )
        precision = sum(1.0 / max(var, 1e-20) for var in variances.values())
        variance = 1.0 / precision
        distance = float(
            variance
            * sum(
                value / max(variances[source], 1e-20)
                for source, value in estimates.items()
            )
        )
        rays = np.array([[x1, y2, 1.0], [x2, y2, 1.0]]) @ inverse_K.T
        camera_points = rays * (distance / rays[:, 2:3])
        front = (camera_points @ transform[:3, :3].T + transform[:3, 3])[:, :2]
        edge = front[1] - front[0]
        side = float(np.linalg.norm(edge))
        if side <= 1e-9:
            continue
        away = np.array([-edge[1], edge[0]]) / side
        forward = transform[:2, 2]
        alignment = float(away @ forward)
        if abs(alignment) <= 1e-9:
            continue
        if alignment < 0:
            away = -away
        ring = np.array(
            [
                front[0],
                front[1],
                front[1] + side * away,
                front[0] + side * away,
                front[0],
            ]
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"obstacle_{index:04d}",
                    "Type": "detected obstacle",
                    "detection_index": index,
                    "side_length_m": side,
                    "area_m2": side**2,
                    "depth_m": distance,
                    "depth_variance_m2": variance if baseline is not None else None,
                    "depth_sources": list(estimates),
                    **estimates,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
            }
        )
    return {"type": "FeatureCollection", "features": features}
