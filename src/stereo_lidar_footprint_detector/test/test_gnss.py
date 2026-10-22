"""Recorded GNSS timing, geographic axes and interpolation checks."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stereo_lidar_footprint_detector.gnss import (
    PIREN_LLA, Track, interpolate_pose, lla_to_ned, load_track, utc_ns,
)


def test_piren_origin_and_axis_directions():
    np.testing.assert_allclose(lla_to_ned(PIREN_LLA), [0, 0, 0], atol=1e-8)
    north = lla_to_ned([PIREN_LLA[0] + 0.001, *PIREN_LLA[1:]])
    east = lla_to_ned([PIREN_LLA[0], PIREN_LLA[1] + 0.001, PIREN_LLA[2]])
    assert 110 < north[0] < 112
    assert 49 < east[1] < 51
    np.testing.assert_allclose(lla_to_ned([*PIREN_LLA[:2], PIREN_LLA[2] + 1]),
                               [0, 0, -1], atol=1e-8)


def test_spatial_dual_kml_and_untimed_pos_fallback(tmp_path):
    kml = tmp_path / 'track.kml'
    kml.write_text('''<kml><Placemark><TimeStamp>
        <when>2023-07-11T10:21:24Z</when></TimeStamp><Point>
        <coordinates>10.4 63.4</coordinates></Point></Placemark></kml>''')
    track = load_track(kml)
    assert track.stamps[0] == utc_ns('2023-07-11T10:21:24Z')
    np.testing.assert_allclose(track.lla[0, :2], [63.4, 10.4])
    kml.write_text('<kml><LineString><coordinates>10.4,63.4,0</coordinates></LineString></kml>')
    kml.with_suffix('.pos').write_text(
        '%  GPST latitude(deg) longitude(deg) height(m) Q\n'
        '2023/07/11 10:21:42.000 63.4 10.4 36.0 2\n')
    fallback = load_track(kml)
    assert fallback.source.endswith('.pos')
    assert fallback.stamps[0] == track.stamps[0]


def test_no_extrapolation_or_interpolation_across_gaps():
    samples = [(0, PIREN_LLA), (1_000_000_000, [PIREN_LLA[0] + .001, *PIREN_LLA[1:]])]
    track = Track(samples, 'test')
    assert track.at(-1, 2) is None
    assert track.at(1_000_000_001, 2) is None
    assert track.at(500_000_000, .5) is None
    assert 55 < track.at(500_000_000, 2)[0] < 56
    assert track.at(0, 2) is not None


def test_pose_rotation_interpolation_wraps_heading():
    qa = Rotation.from_euler('z', 170, degrees=True).as_quat()
    qb = Rotation.from_euler('z', -170, degrees=True).as_quat()
    samples = [(0, np.r_[[0, 0, 0], qa]), (1_000_000_000, np.r_[[2, 0, 0], qb])]
    position, rotation = interpolate_pose(samples, 500_000_000, 2)
    np.testing.assert_allclose(position, [1, 0, 0])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [-1, 0, 0], atol=1e-10)
    assert interpolate_pose(samples, 2_000_000_000, 2) is None


def test_reject_unsupported_pos_time_system(tmp_path):
    path = tmp_path / 'bad.pos'
    path.write_text('% TOW latitude(deg) longitude(deg) height(m) Q\n'
                    '2023/07/11 10:21:42.000 63.4 10.4 36.0 2\n')
    with pytest.raises(ValueError, match='time header'):
        load_track(path)
