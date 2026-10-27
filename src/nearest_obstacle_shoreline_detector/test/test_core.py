import numpy as np
import pytest

from nearest_obstacle_shoreline_detector.core import Settings, barrier, boundary_rows


def scene():
    mask = np.zeros((100, 40), dtype=np.uint8)
    mask[40:] = 255
    k = np.array([[100., 0., 20.], [0., 100., 50.], [0., 0., 1.]])
    # Optical x=right, y=down, z=forward -> water x=forward, y=left, z=up.
    transform = np.array([[0., 0., 1., 2.], [-1., 0., 0., 3.],
                          [0., -1., 0., 1.5], [0., 0., 0., 1.]])
    points = np.array([[(u-20)*z/100, (39-50)*z/100, z]
                       for u in (4., 5., 14., 15., 24., 25., 34., 35.) for z in (10.,)])
    return mask, points, k, transform


def test_bottom_up_boundary_and_unsupported_columns():
    mask, _, _, _ = scene()
    mask[:, 0] = 255  # all water: no edge
    mask[49, 1] = 0  # scan begins blocked
    mask[10, 2] = 255  # detached water above the first non-water
    rows = boundary_rows(mask)
    assert np.isnan(rows[:2]).all()
    np.testing.assert_array_equal(rows[2:], 39)


def test_margin_controls_where_scan_starts():
    mask, _, _, _ = scene()
    mask[90:] = 0
    assert np.isfinite(boundary_rows(mask)).all()
    assert np.isnan(boundary_rows(mask, 0)).all()
    with pytest.raises(ValueError):
        boundary_rows(mask, 100)


def test_measured_boundary_transforms_and_flattens():
    mask, points, k, transform = scene()
    xyz, segments, rows = barrier(mask, points, k, transform)
    assert segments.shape == (3, 2, 3)
    np.testing.assert_allclose(xyz[:, 0], 12.)
    np.testing.assert_allclose(xyz[:, 1], [4.55, 3.55, 2.55, 1.55])
    np.testing.assert_allclose(xyz[:, 2], 0.)
    np.testing.assert_allclose(rows, 39)


def test_missing_strip_and_large_jump_break_barrier():
    mask, points, k, transform = scene()
    xyz, segments, _ = barrier(mask, np.delete(points, [2, 3], axis=0), k, transform)
    assert np.isnan(xyz[1]).all()
    assert segments.shape == (1, 2, 3)  # only strips 2 and 3 connect
    _, segments, _ = barrier(mask, points, k, transform, Settings(max_segment_m=0.5))
    assert segments.shape == (0, 2, 3)


def test_nearer_singleton_does_not_hide_supported_shore():
    mask, points, k, transform = scene()
    outlier = np.array([[(4-20)*3/100, (39-50)*3/100, 3.]])
    xyz, _, _ = barrier(mask, np.vstack([points, outlier]), k, transform)
    np.testing.assert_allclose(xyz[:, 0], 12.)


def test_nearest_supported_surface_wins_over_inland_returns():
    mask, points, k, transform = scene()
    xyz, _, _ = barrier(mask, np.vstack([points * 2, points]), k, transform)
    np.testing.assert_allclose(xyz[:, 0], 12.)


@pytest.mark.parametrize('kind', ['empty', 'behind', 'outside', 'nan', 'far', 'off_boundary'])
def test_unusable_returns_never_create_barrier(kind):
    mask, points, k, transform = scene()
    if kind == 'empty':
        points = np.empty((0, 3))
    elif kind == 'behind':
        points *= -1
    elif kind == 'outside':
        points[:, 0] = 1000
    elif kind == 'nan':
        points[:] = np.nan
    elif kind == 'far':
        points *= 100
    else:
        points[:, 1] = 4  # projects far below edge
    xyz, segments, _ = barrier(mask, points, k, transform)
    assert np.isnan(xyz).all()
    assert len(segments) == 0


@pytest.mark.parametrize('kwargs', [dict(min_points=0), dict(bottom_margin_px=-1),
                                    dict(max_segment_m=float('nan')), dict(column_width_px=0)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
