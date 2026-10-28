# Fordypningsprosjekt ROS code

ROS 2 Jazzy workspace code for the maritime perception/free-space project.

The current stixel pipeline supports both:

- VRX simulation input; and
- milliAmpere 2 (MA2) 2023 recordings using ZED SVO + ROS bag data.

## Canonical setup

Clone this workspace with its third-party packages:

```bash
git clone --recurse-submodules https://github.com/philliab13/Fordypningsprosjekt-ros-kode.git vrx_ws
cd vrx_ws
```

The VRX and three ZED repositories are Git submodules pinned to specific commits.
For an existing clone, or after pulling changes to the submodule revisions, run:

```bash
git submodule update --init --recursive
```

Cloning without `--recurse-submodules` does not download their source; the update
command above retrieves it. The ZED submodules provide the camera wrapper,
message interfaces, and camera descriptions. ZED SDK/CUDA and ROS dependencies
must still be installed separately on the machine running the camera.

The supported Python runtime is a clean Python 3.12 virtual environment using
NumPy 2.2.6, pip OpenCV, PyTorch/CUDA, Ultralytics, Open3D, Numba, SciPy, ROS 2
Jazzy Python modules, and ZED SDK 5.5 `pyzed`.

Do not use `--system-site-packages` for this environment.

From the repository/workspace root:

```bash
git pull
chmod +x scripts/setup_stixel_env.sh
./scripts/setup_stixel_env.sh
source scripts/activate_stixel_env.sh

colcon build --merge-install --packages-select lidar_free_space stixel_free_space
source install/setup.bash
```

The detailed setup, MA2 playback procedure, VRX procedure, dependency rules,
calibration notes, diagnostics, and cleanup instructions are in:

```text
src/stixel_free_space/README.md
```

This workspace uses **merge install** for the stixel package.
