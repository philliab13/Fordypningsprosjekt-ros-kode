#!/usr/bin/env bash
set -euo pipefail

# Canonical runtime bootstrap for stixel_free_space + MA2 ZED playback.
#
# Usage:
#   ./scripts/setup_stixel_env.sh
#
# Optional:
#   STIXEL_VENV=/custom/path ./scripts/setup_stixel_env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${STIXEL_VENV:-$HOME/venvs/stixel}"
PYTHON_BIN="/usr/bin/python3"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
REQ="$ROOT/src/stixel_free_space/requirements-runtime.txt"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS 2 Jazzy not found at $ROS_SETUP" >&2
  exit 1
fi

if [[ ! -f "$REQ" ]]; then
  echo "Missing requirements file: $REQ" >&2
  exit 1
fi

PY_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_MINOR" != "3.12" ]]; then
  echo "Expected /usr/bin/python3 to be Python 3.12, found $PY_MINOR" >&2
  exit 1
fi

echo "Creating clean venv at: $VENV"
rm -rf "$VENV"
"$PYTHON_BIN" -m venv "$VENV"

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REQ" -r "$ROOT/requirements/build-pinned.txt"

# ZED SDK installs pyzed outside the venv. Link only the pyzed package, never the
# whole user site-packages directory, so its NumPy and other packages cannot leak in.
PYZED_DIR="$($PYTHON_BIN - <<'PY'
import pathlib
import pyzed
print(pathlib.Path(pyzed.__file__).resolve().parent)
PY
)"

VENV_SITE="$($VENV/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

rm -rf "$VENV_SITE/pyzed"
ln -s "$PYZED_DIR" "$VENV_SITE/pyzed"

echo "Linked pyzed:"
echo "  $VENV_SITE/pyzed -> $PYZED_DIR"

# ROS's Python packages are not inherited by this clean venv. Verify the runtime
# with the ROS Python directory exposed exactly as used in normal sessions.
ROS_PYTHON="/opt/ros/jazzy/lib/python3.12/site-packages"

PYTHONPATH="$ROS_PYTHON" "$VENV/bin/python" - <<'PY'
import cv2
import numba
import numpy
import open3d
import pyzed.sl
import rclpy
import scipy
import torch
import ultralytics

print("Runtime verification OK")
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("numba:", numba.__version__)
print("cv2:", cv2.__version__)
print("open3d:", open3d.__version__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
print("rclpy: OK")
print("pyzed: OK")

if numpy.__version__ != "2.2.6":
    raise SystemExit(f"Unexpected NumPy version: {numpy.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
PY

echo
echo "Environment ready."
echo "Activate with:"
echo "  source $ROOT/scripts/activate_stixel_env.sh"
