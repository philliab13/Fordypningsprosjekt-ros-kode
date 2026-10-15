"""Load shared MA2 scenario files without ROS or ZED dependencies."""

from fractions import Fraction
from pathlib import Path

import yaml


def _clap_value(value):
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError('A clap interval must contain two timestamps')
        return (_clap_value(value[0]) + _clap_value(value[1])) / 2
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('Clap timestamps must be integer nanoseconds')
    return Fraction(value)


def load_scenario(scenario, scenario_dir, data_root):
    """Return config path, config, SVO path and GNSS-minus-ZED offset in ns."""
    path = Path(scenario).expanduser()
    if path.suffix not in ('.yaml', '.yml'):
        path = path.with_suffix('.yaml')
    if not path.is_absolute():
        path = Path(scenario_dir).expanduser() / path
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'{path}: expected a scenario mapping')
    svo_file = config.get('svo_file')
    if not isinstance(svo_file, str) or not svo_file.strip():
        raise ValueError(f'{path}: svo_file must be a nonempty path')
    claps = config.get('clap_timestamps', {})
    if not isinstance(claps, dict):
        raise ValueError(f'{path}: clap_timestamps must be a mapping')
    ma2, svo = claps.get('ma2'), claps.get('svo')
    if (not isinstance(ma2, list) or not isinstance(svo, list)
            or not ma2 or len(ma2) != len(svo)):
        raise ValueError(f'{path}: MA2 and SVO claps must be nonempty lists of equal length')
    # Subtract before averaging; avoid float rounding at epoch-nanosecond scale.
    offset = int(sum((_clap_value(a) - _clap_value(b)
                      for a, b in zip(ma2, svo)), Fraction()) / len(ma2))
    input_path = Path(data_root).expanduser() / Path(svo_file).expanduser()
    return path, config, str(input_path), offset
