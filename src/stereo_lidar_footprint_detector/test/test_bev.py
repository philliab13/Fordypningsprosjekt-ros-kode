"""Check frame alignment, segmentation selection and temporal scan selection."""

import numpy as np

from stereo_lidar_footprint_detector.bev import (
    closest_scans, local_geometry, mask_points, render_bev, render_lidar_overlay,
    scan_color,
)


def collection(ring):
    return {'features': [{'geometry': {'coordinates': [ring]}}]}


def test_scan_window_and_absolute_distance():
    scans = [(900_000_000, 'past'), (1_050_000_000, 'future'),
             (1_400_000_000, 'outside'), (800_000_000, 'edge')]
    assert closest_scans(scans, 1_000_000_000, 3, 0.2) == [
        scans[1], scans[0], scans[3]]
    assert closest_scans(scans, 1_000_000_000, 1, 0.2) == [scans[1]]
    assert closest_scans(scans, 0, 3, 0.2) == []


def test_mask_rejects_box_background_and_outside_pixels():
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 2] = 1
    uv = np.array([[2.2, 1.8], [1, 1], [4, 1], [-0.1, 1], [np.nan, 1]])
    xyz = np.arange(15).reshape(-1, 3)
    np.testing.assert_array_equal(mask_points(uv, xyz, mask), xyz[:1])


def test_water_flattening_keeps_xy_despite_different_heights():
    transform = np.eye(4)
    # Optical right = water -Y, optical forward = water +X.
    transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    transform[:3, 3] = [10, 20, 2]
    water = np.array([[12, 19, 1], [12, 19, 5]])
    camera = (water - transform[:3, 3]) @ transform[:3, :3]
    ring = [[12, 19], [13, 19], [13, 18], [12, 19]]
    points, rings = local_geometry(camera, collection(ring), transform)
    np.testing.assert_allclose(points, [[1, 2], [1, 2]])
    np.testing.assert_allclose(rings[0], [[1, 2], [1, 3], [2, 3], [1, 2]])


def test_synthetic_plane_matches_detector_and_render_handles_empty():
    ring = [[1, 4], [2, 4], [2, 5], [1, 4]]
    points, rings = local_geometry([[1, 8, 4]], collection(ring))
    np.testing.assert_allclose(points, [[1, 4]])
    np.testing.assert_allclose(rings[0], ring)
    for xy, polygons in [(points, rings), (np.empty((0, 2)), [])]:
        image = render_bev(xy, polygons, 'test', size_px=400, range_m=20)
        assert image.shape == (400, 400, 3)
        assert image.dtype == np.uint8


def test_z_down_water_frame_does_not_mirror_camera_right():
    transform = np.eye(4)
    transform[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    points, rings = local_geometry(
        [[2, 7, 5]], collection([[5, 2], [6, 2], [5, 2]]), transform)
    np.testing.assert_allclose(points, [[2, 5]])
    np.testing.assert_allclose(rings[0][0], [2, 5])


def test_rgb_overlay_scan_colors_and_faint_background():
    rgb = np.full((400, 600, 3), 120, dtype=np.uint8)
    color = scan_color(0)
    scans = [dict(uv=np.array([[200, 200], [300, 200]]),
                  boat=np.array([True, False]), color=color)]
    overlay = render_lidar_overlay(rgb, scans, [('S1: +10 ms', color)], 'test')
    np.testing.assert_array_equal(overlay[200, 200], color)
    np.testing.assert_allclose(overlay[200, 300],
                               0.3 * np.array(color) + 0.7 * 120, atol=1)
    np.testing.assert_array_equal(rgb, 120)
    bev = render_bev([[0, 5]], [], 'test', size_px=600, range_m=20,
                     point_colors=[color], scan_legend=[('S1: +10 ms', color)])
    # The same exact scan colour is present in both rasters.
    assert np.any(np.all(bev == color, axis=2))


def test_shoreline_segments_remain_open_and_are_clipped_to_plot():
    segments = [[[-100, 5], [-2, 5]], [[2, 5], [100, 5]]]
    image = render_bev([], [], 'test', size_px=400, range_m=20,
                       shoreline_segments=segments)
    # With no scan legend the plot margin is 65, scale is 13.5 px/metre.
    y = round(400 - 65 - (5 + 2) * 13.5)
    np.testing.assert_array_equal(image[y, 140], [0, 0, 220])
    np.testing.assert_array_equal(image[y, 260], [0, 0, 220])
    assert not np.array_equal(image[y, 200], [0, 0, 220])  # unsupported gap
    assert not np.array_equal(image[y, 50], [0, 0, 220])  # outside plot
