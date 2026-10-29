# Pinned pip dependencies

These files pin the packages used by the application, rather than every package
installed in the development venvs. Versions were checked against the local
Python 3.12 environments on 2026-10-29. Transitive dependencies are resolved by
pip; these are not full environment lockfiles.

| Install file | Purpose |
| --- | --- |
| `stixel-pip-pinned.txt` | Stixel adapter, SVO player and external estimation pipeline |
| `footprint-pip-pinned.txt` | Footprint detector, BEV viewer and shared helpers |
| `footprint-gui.txt` | Footprint runtime plus the optional frame-review GUI |
| `build-pinned.txt` | Setuptools and colcon build tools for either venv |
| `footprint-pip-install.txt` | Alias for `footprint-pip-pinned.txt` |

Runtime pins are maintained in each package's `requirements-runtime.txt`; the
workspace files include those canonical lists rather than duplicating versions.

## Why these packages

| Environment | Imported modules / functionality | Pip distributions |
| --- | --- | --- |
| Both | Arrays, image handling, transforms | NumPy, opencv-python, SciPy |
| Both | YOLO/FastSAM inference | Ultralytics, Torch, Torchvision |
| Stixel | Scenario YAML, temporal/stixel algorithms, RWPS | PyYAML, Numba, Open3D |
| Footprint GUI (optional) | Qt frame-review window | PyQt5 |

The stixel audit includes `stixel_free_space/backend.py`'s dynamically loaded
`free_space.pipeline` and its FastSAM, YOLO, RWPS, stixels, utils and temporal
filtering modules in the external `free-space-estimation` checkout. The footprint
audit includes its bundled detection modules and `stixel_free_space.core` helpers.
Torch and Torchvision are pinned as an inference pair even when imported by
Ultralytics rather than the ROS adapter directly.

Matplotlib, NVIDIA wheel packages and other indirect dependencies are not pinned
here individually; pip installs them where required by the selected libraries.
Unused notebooks, offline dataset readers, visualization scripts and their
optional packages are outside these ROS runtime lists. ROS modules and SDK
`pyzed` do not belong in these pip files.

## Install

Run from the workspace root with Python 3.12 and ROS 2 Jazzy already available.
GPU versions target the current Linux x86_64/CUDA 13 setup, not Jetson/aarch64.
Use fresh venvs to avoid retaining old packages; the commands below do not delete
or clean existing environments.

```bash
/usr/bin/python3.12 -m venv "$HOME/venvs/stixel"
"$HOME/venvs/stixel/bin/python" -m pip install \
  -r requirements/stixel-pip-pinned.txt -r requirements/build-pinned.txt
"$HOME/venvs/stixel/bin/python" -m pip check
```

For footprint, preserving the existing ROS/system-package visibility:

```bash
/usr/bin/python3.12 -m venv --system-site-packages .venv-footprint
.venv-footprint/bin/python -m pip install \
  -r requirements/footprint-pip-pinned.txt -r requirements/build-pinned.txt
.venv-footprint/bin/python -m pip check
```

For the optional review GUI, also install:

```bash
.venv-footprint/bin/python -m pip install -r requirements/footprint-gui.txt
```

If system `python3-pyqt5` is already available through `--system-site-packages`,
the optional pip GUI install is unnecessary. SDK `pyzed` still needs linking into
the clean stixel venv as in `scripts/setup_stixel_env.sh`. That script now uses
the same stixel runtime and build requirements. Follow `checklist_Ma1.md` for
activation and package build commands.

## Version checks and limits

Footprint uses the observed NumPy 1.26.4, SciPy 1.11.4 (previously inherited from
Ubuntu), Torch 2.14.1, Torchvision 0.29.1 and Ultralytics 8.4.176. Its observed
OpenCV 5.0.0.93 requires NumPy >=2, so the maintained requirements use OpenCV
4.11.0.86 instead. That release's metadata accepts NumPy 1.26.4 on Python 3.12.
This is an intentional compatibility correction, not an exact copy of the old venv.

Stixel runtime pins match the checked installed versions. Existing venvs were
not modified. Fresh installs, GPU inference and execution on MA1 still need to
be verified; inherited system-package conflicts may also appear in `pip check`.
