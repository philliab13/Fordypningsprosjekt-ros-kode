import numpy as np
import pytest

from nearest_obstacle_shoreline_detector.lidar_projection import project_to_water


def test_height_band_and_projection_preserve_xy():
    points = np.array([[1, 2, -.01], [3, 4, 0], [5, 6, 1.999], [7, 8, 2], [9, 10, 3]])
    filtered, projected, heights = project_to_water(points, np.eye(4))
    np.testing.assert_allclose(filtered, points[1:3])
    np.testing.assert_allclose(projected, [[3, 4, 0], [5, 6, 0]])
    np.testing.assert_allclose(heights, [0, 1.999])
    np.testing.assert_allclose(points[2], [5, 6, 1.999])


def test_filter_uses_water_frame_height_after_tilt_and_translation():
    # Water Z is sensor X + 4, not sensor Z. Water elevation is 5.
    transform = np.array([[0, 0, -1, 10], [0, 1, 0, 20],
                          [1, 0, 0, 4], [0, 0, 0, 1]], dtype=float)
    points = np.array([[0, 1, 9], [1, 2, 8], [2, 3, 7], [3, 4, 6]])
    filtered, projected, heights = project_to_water(points, transform, water_level_m=5)
    np.testing.assert_allclose(filtered, [[2, 22, 5], [3, 23, 6]])
    np.testing.assert_allclose(projected, [[2, 22, 5], [3, 23, 5]])
    np.testing.assert_allclose(heights, [0, 1])


def test_nonfinite_and_empty_clouds():
    for points in ([], [[np.nan, 1, 1], [1, np.inf, 1], [1, 1, -np.inf]]):
        filtered, projected, heights = project_to_water(points, np.eye(4))
        assert filtered.shape == projected.shape == (0, 3)
        assert heights.shape == (0,)


def test_configurable_band():
    _, projected, heights = project_to_water([[0, 0, -.5], [0, 0, .5]], np.eye(4),
                                            min_height_m=-1, max_height_m=0)
    np.testing.assert_allclose(projected, [[0, 0, 0]])
    np.testing.assert_allclose(heights, [-.5])


@pytest.mark.parametrize('kwargs', [dict(max_height_m=0), dict(min_height_m=3),
                                    dict(water_level_m=np.nan), dict(max_height_m=np.inf)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        project_to_water([], np.eye(4), **kwargs)
