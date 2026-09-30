import math
from types import SimpleNamespace
import numpy as np
import pytest
from lidar_free_space.core import Config, estimate, contiguous_runs, transform_points


def test_nearest_hit_survives_and_unknown_stays_unknown():
    _, ranges = estimate([[5, 0, 1], [20, 0, 1], [8, 0, 1]], [0, 0, 2], 0, Config())
    assert np.isfinite(ranges).sum() == 1
    assert np.nanmin(ranges) == 5


def test_filters_and_horizontal_distance():
    points = [[float('nan'), 0, 1], [float('inf'), 0, 1], [5, 0, 0],
              [5, 0, 4], [131, 0, 1], [0.01, 0, 1], [3, 4, 2]]
    _, r = estimate(points, [0, 0, 8], 0, Config(max_range=5.0))
    assert np.isfinite(r).sum() == 1
    assert np.nanmin(r) == 5


def test_water_reference_and_origin_translation():
    _, r = estimate([[13, 24, 11]], [10, 20, 12], 10, Config())
    assert np.nanmin(r) == 5


def test_empty_cloud():
    _, r = estimate([], [0, 0, 0], 0, Config())
    assert np.isnan(r).all()


def test_circular_runs_do_not_bridge_gaps():
    assert contiguous_runs([True, False, True, True]) == [[2, 3, 0]]
    assert contiguous_runs([True, False, True, False]) == [[2], [0]]
    assert contiguous_runs([True, True]) == [[0, 1]]
    assert contiguous_runs([False, False]) == []


def test_pi_seam_is_in_bounds():
    _, r = estimate([[-5, 0.0, 1], [-4, -0.0, 1]], [0, 0, 0], 0, Config())
    assert np.isfinite(r).sum() == 1
    assert r[0] == 4


def test_roll_transform():
    tf = SimpleNamespace(translation=SimpleNamespace(x=1, y=2, z=3),
                         rotation=SimpleNamespace(x=math.sin(math.pi/4), y=0, z=0,
                                                  w=math.cos(math.pi/4)))
    np.testing.assert_allclose(transform_points(np.array([[0, 1, 0]]), tf), [[1, 2, 4]])


@pytest.mark.parametrize('kwargs', [dict(angular_resolution=0), dict(angular_resolution=7),
                                    dict(max_range=-1), dict(min_obstacle_height=5),
                                    dict(max_range=float('nan'))])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_assumed_clear_only_for_explicit_no_returns():
    from lidar_free_space.core import extend_no_returns
    angles = np.deg2rad([-135, -45, 45, 135])
    rays = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(4)))
    r, status = extend_no_returns(np.array([np.nan, np.nan, np.nan, 8.]),
                                  [np.inf, np.nan, 30., np.inf], rays, 130., 100.)
    np.testing.assert_allclose(r, [100, np.nan, np.nan, 8], equal_nan=True)
    np.testing.assert_array_equal(status, [2, 0, 0, 1])
    r, _ = extend_no_returns(np.full(4, np.nan), [np.inf]*4, rays, 50., 100.)
    assert np.isnan(r).all()


def test_mixed_scan_bin_vetoes_extension():
    from lidar_free_space.core import extend_no_returns
    r, _ = extend_no_returns(np.full(4, np.nan), [np.inf, 5], [[1,0,0],[1,0,0]], 130, 100)
    assert np.isnan(r).all()
