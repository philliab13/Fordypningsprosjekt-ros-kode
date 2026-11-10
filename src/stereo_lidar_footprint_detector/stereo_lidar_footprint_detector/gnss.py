"""Recorded GNSS tracks and timestamp interpolation, without ROS dependencies."""

from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = (Path.home() / 'Documents/Data_For_Stixel_based_Free_space_estimation'
        / '2023-07-11_Multi_ZED_Summer')
DEFAULT_FILES = dict(
    spatial_dual_file=str(ROOT / 'Spatial Dual/SpatialDualLog_23-07-11_10-21-15/GoogleEarthTrack.kml'),
    kbox2_file=str(ROOT / 'KBox2/LOG00054.kml'),
    kbox1_file=str(ROOT / 'KBox1/LOG00030.kml'),
    ntbox_file=str(ROOT / 'NTBox/gnssdump-20230710173603.kml'),
)

OFFSET_LAT = - 0.5e-04   # 1.2091700000382843e-05 # - 0.45e-04
OFFSET_LON = 2.0e-04    #-2.2780000000111045e-05 # 2.0e-04
PIREN_LAT = 63.4389029083 #+ OFFSET_LAT
PIREN_LON = 10.39908278 #+ OFFSET_LON
PIREN_ALT = 39.923
PIREN_LLA = [PIREN_LAT, PIREN_LON, PIREN_ALT]


def lla_to_ned(lla, origin=PIREN_LLA):
    """WGS84 ECEF difference rotated into the Piren tangent frame."""
    def ecef(values):
        lat, lon, height = np.asarray(values, dtype=float).T
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        e2 = 6.6943799901413165e-3
        radius = 6378137.0 / np.sqrt(1 - e2 * np.sin(lat)**2)
        return np.stack(((radius + height) * np.cos(lat) * np.cos(lon),
                         (radius + height) * np.cos(lat) * np.sin(lon),
                         (radius * (1 - e2) + height) * np.sin(lat)), axis=-1)

    lat, lon = np.deg2rad(origin[:2])
    rotation = np.array([
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), -np.sin(lat)]])
    return (ecef(lla) - ecef(origin)) @ rotation.T


def ned_to_lla(ned, origin=PIREN_LLA):
    """Invert the Piren tangent-frame conversion; return lat/lon degrees and metres."""
    ned = np.asarray(ned, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if ned.shape[-1:] != (3,) or origin.shape != (3,) or not (
            np.isfinite(ned).all() and np.isfinite(origin).all()):
        raise ValueError('Expected finite NED positions and a latitude/longitude/height origin')
    a, e2 = 6378137.0, 6.6943799901413165e-3
    lat, lon = np.deg2rad(origin[:2])
    rotation = np.array([
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), -np.sin(lat)]])
    radius = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    origin_ecef = np.array([(radius + origin[2]) * np.cos(lat) * np.cos(lon),
                            (radius + origin[2]) * np.cos(lat) * np.sin(lon),
                            (radius * (1 - e2) + origin[2]) * np.sin(lat)])
    xyz = ned @ rotation + origin_ecef
    x, y, z = np.moveaxis(xyz, -1, 0)
    horizontal = np.hypot(x, y)
    lat = np.arctan2(z, horizontal * (1 - e2))
    for _ in range(8):
        radius = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * radius * np.sin(lat), horizontal)
    height = (horizontal * np.cos(lat) + z * np.sin(lat)
              - a * np.sqrt(1 - e2 * np.sin(lat)**2))
    return np.stack((np.rad2deg(lat), np.rad2deg(np.arctan2(y, x)), height), axis=-1)


def utc_ns(value):
    dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('KML timestamps must include a timezone')
    return round(dt.timestamp() * 1e9)


class Track:
    def __init__(self, samples, source):
        # Sorting also supports KML files whose placemarks are out of order.
        samples = dict(samples)
        if not samples:
            raise ValueError(f'No timestamped GNSS positions in {source}')
        self.stamps = np.array(sorted(samples), dtype=np.int64)
        self.lla = np.array([samples[t] for t in self.stamps])
        if not np.isfinite(self.lla).all():
            raise ValueError(f'Nonfinite GNSS coordinates in {source}')
        self.source = str(source)

    def at(self, timestamp, max_gap_s, origin=PIREN_LLA):
        match = bracket(self.stamps, timestamp, max_gap_s)
        if match is None:
            return None
        left, right, weight = match
        # The comparison is horizontal: KML altitude can be zero or missing.
        lla = self.lla[[left, right]].copy()
        lla[:, 2] = origin[2]
        ned = lla_to_ned(lla, origin)
        return ned[0] * (1 - weight) + ned[1] * weight


def bracket(stamps, timestamp, max_gap_s):
    """Require bracketing samples and never extrapolate across missing data."""
    index = int(np.searchsorted(stamps, timestamp))
    if index < len(stamps) and stamps[index] == timestamp:
        return index, index, 0.0
    if index == 0 or index == len(stamps):
        return None
    left, right = index - 1, index
    gap = int(stamps[right]) - int(stamps[left])
    if gap > max_gap_s * 1e9:
        return None
    return left, right, (timestamp - int(stamps[left])) / gap


def load_track(filename, gps_utc_offset_s=18.0):
    """Read timed KML points, or their sibling RTKLIB .pos when untimed.

    The GPS/UTC offset defaults to the value for this July 2023 dataset.
    RTKLIB GPST and UTC headers are handled explicitly, never inferred from names.
    """
    path = Path(filename).expanduser()
    if path.suffix.lower() == '.kml':
        samples = []
        root = ET.parse(path).getroot()
        for mark in root.iter():
            if mark.tag.split('}')[-1] != 'Placemark':
                continue
            times = [e.text for e in mark.iter()
                     if e.tag.split('}')[-1] == 'when' and e.text]
            points = [e for e in mark.iter() if e.tag.split('}')[-1] == 'Point']
            if len(times) != 1 or len(points) != 1:
                continue
            coordinates = next((e.text for e in points[0].iter()
                                if e.tag.split('}')[-1] == 'coordinates'), None)
            if coordinates:
                values = [float(x) for x in coordinates.replace(',', ' ').split()]
                lon, lat = values[:2]
                samples.append((utc_ns(times[0]), [lat, lon, 0.0]))
        if samples:
            return Track(samples, path)
        path = path.with_suffix('.pos')
    if path.suffix.lower() != '.pos':
        raise ValueError('GNSS files must be KML or RTKLIB .pos')
    samples, time_system = [], None
    for line in path.read_text().splitlines():
        if line.startswith('%'):
            if 'latitude(deg)' in line:
                time_system = line.split()[1]
            continue
        if not line.strip():
            continue
        if time_system not in ('GPST', 'UTC'):
            raise ValueError(f'Unsupported RTKLIB coordinates or time header in {path}')
        fields = line.split()
        if int(fields[5]) == 0:
            continue
        dt = datetime.strptime(' '.join(fields[:2]), '%Y/%m/%d %H:%M:%S.%f')
        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
        if time_system == 'GPST':
            timestamp -= gps_utc_offset_s
        samples.append((round(timestamp * 1e9), list(map(float, fields[2:5]))))
    return Track(samples, path)


def interpolate_pose(samples, timestamp, max_gap_s):
    """Interpolate position and quaternion using source stamps."""
    samples = dict(samples)
    stamps = sorted(samples)
    match = bracket(stamps, timestamp, max_gap_s)
    if match is None:
        return None
    left, right, weight = match
    a, b = samples[stamps[left]], samples[stamps[right]]
    position = a[:3] * (1 - weight) + b[:3] * weight
    rotation = (Rotation.from_quat(a[3:]) if left == right else
                Slerp([0, 1], Rotation.from_quat([a[3:], b[3:]]))([weight])[0])
    return position, rotation.as_matrix()
