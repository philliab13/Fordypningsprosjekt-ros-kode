# Pinned pip environments

Captured from the two testing environments on 2026-10-29, using Python 3.12
on Linux x86_64. These files include every venv-local pip distribution, including
build tools and transitive packages. They exclude inherited system packages,
ROS/workspace packages and the SDK-provided `pyzed` wheel.

| File | Purpose |
| --- | --- |
| `stixel-pip-pinned.txt` | Exact snapshot of `~/venvs/stixel` (150 packages) |
| `footprint-pip-pinned.txt` | Exact snapshot of `.venv-footprint` (41 packages) |
| `footprint-pip-install.txt` | Footprint installation variant with OpenCV 4.11 instead of 5 |

The footprint snapshot contains an existing dependency conflict:
`opencv-python==5.0.0.93` requires NumPy >=2, while the environment contains
`numpy==1.26.4`. The installation variant changes only OpenCV to `4.11.0.86`,
whose NumPy requirements support 1.26.4 on Python 3.12. This variant has not been
installed or tested with the detector yet. Neither existing venv was modified.

## Install on MA1

Run from the workspace root, with Python 3.12, ROS 2 Jazzy, and compatible NVIDIA
drivers already installed. These GPU pins target the current x86_64/CUDA 13
setup; they are not a Jetson/aarch64 environment recipe.

Create the environments only if those paths do not already contain an environment
you want to keep. Creating a venv over an existing one does not remove old packages.

```bash
/usr/bin/python3.12 -m venv "$HOME/venvs/stixel"
"$HOME/venvs/stixel/bin/python" -m pip install \
  -r requirements/stixel-pip-pinned.txt
"$HOME/venvs/stixel/bin/python" -m pip check
```

Reproduce the footprint environment's use of system Python packages:

```bash
/usr/bin/python3.12 -m venv --system-site-packages .venv-footprint
.venv-footprint/bin/python -m pip install \
  -r requirements/footprint-pip-install.txt
.venv-footprint/bin/python -m pip check
```

The footprint pip snapshot does not pin inherited packages such as SciPy,
Matplotlib, PyYAML and PyQt5. Those must be supplied by the system as on the test
machine; pip may otherwise resolve additional dependencies beyond these pins.
This is a snapshot of the pip portion, not a complete machine lock.

After installation, use the activation and build commands in `checklist_Ma1.md`.
The clean stixel environment also needs the SDK's `pyzed` package linked as
described in `scripts/setup_stixel_env.sh`. That older setup script recreates the
venv and installs the shorter runtime requirements; do not run it after installing
this full snapshot if you want to retain these exact pins.

## Validation

The captured stixel environment passes `pip check`. The original footprint
environment reports the OpenCV/NumPy conflict above and a missing system `cffi`
dependency for PyNaCl. System package issues are outside these pip-only files.
Package downloads and installation on the MA1 computer have not been verified.
