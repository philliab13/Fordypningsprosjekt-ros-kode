# Stixel free-space ROS 2 adapter

ROS 2 Jazzy adapter around the existing `FreeSpacePipeline` in the separate
`Stixel_based_free_space_estimation/free-space-estimation` checkout.

The adapter supports two input modes:

- **VRX**: live stereo camera + lidar + TF from the VRX simulator.
- **MA2**: July 2023 milliAmpere 2 recordings, using a ZED SVO for images/depth
  and the ROS bag for lidar and vessel pose.

The upstream estimator is imported directly. The adapter does not copy or rewrite
FastSAM, YOLO, RWPS, temporal filtering, or stixel depth fusion.

## Supported runtime

The canonical tested runtime is:

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12
- NumPy 2.2.6
- SciPy 1.13.1
- Numba 0.61.2
- OpenCV 4.11.0 (pip wheel, **not** Ubuntu's NumPy-1-built Python OpenCV)
- Open3D 0.20.0
- PyTorch 2.14.1 + CUDA 13.0 wheel
- torchvision 0.29.1 + CUDA 13.0 wheel
- Ultralytics 8.4.174
- ZED SDK 5.5 `pyzed`
- NVIDIA GPU with a driver new enough for the selected PyTorch CUDA runtime

The original free-space repository was developed against an older Python/Ubuntu
stack. Do **not** install its historical `requirements.txt` into this ROS runtime.

### Why NumPy 2 works here

Earlier failures came from mixing NumPy 2 with Ubuntu/system Python extensions that
had been compiled against the NumPy 1 ABI, especially the system OpenCV/cv_bridge
path. This package no longer uses `cv_bridge`; image conversion is done directly
between `sensor_msgs/Image` and NumPy. The canonical venv installs a NumPy-2-
compatible OpenCV wheel and was verified with the complete stixel + ZED runtime.

Do not create this environment with `--system-site-packages`. That can leak
`~/.local` packages (including another NumPy version) into the venv.

## One-time setup

The workspace is assumed to be:

```text
~/Documents/vrx_ws
```

and the upstream estimator checkout is assumed to be:

```text
~/Documents/Stixel_based_free_space_estimation/free-space-estimation
```

ZED SDK 5.5 must already be installed and this must work with system Python:

```bash
/usr/bin/python3 -c "import pyzed.sl as sl; print('pyzed OK')"
```

From the workspace root:

```bash
cd ~/Documents/vrx_ws
git pull

chmod +x scripts/setup_stixel_env.sh
./scripts/setup_stixel_env.sh
```

The script deliberately creates a **clean** environment at:

```text
~/venvs/stixel
```

It then:

1. installs the pinned runtime from
   `src/stixel_free_space/requirements-runtime.txt`;
2. finds the ZED SDK's existing `pyzed` package;
3. links only `pyzed` into the venv;
4. does not expose the rest of `~/.local/site-packages`;
5. verifies NumPy, SciPy, Numba, OpenCV, Open3D, PyTorch/CUDA,
   Ultralytics, ROS `rclpy`, and `pyzed` together.

### Activate the runtime

For every new terminal that runs the stixel/MA2 Python nodes:

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh
```

This activates `~/venvs/stixel`, sources ROS Jazzy, exposes the ROS Python
packages to the clean venv, and sources the workspace install if it exists.

Check the interpreter before debugging anything else:

```bash
which python
python -c "import numpy, cv2, pyzed.sl, rclpy; print(numpy.__version__, cv2.__version__)"
```

Expected NumPy:

```text
2.2.6
```

## Build

After activating the environment:

```bash
cd ~/Documents/vrx_ws
colcon build --merge-install --packages-select lidar_free_space stixel_free_space
source install/setup.bash
```

This project uses **merge install**, not symlink install.

Rebuild after changing Python files because the installed package is a copied build.

## Run MA2 scenario data

Three terminals are used.

### Terminal 1: stixel pipeline

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh

python -m stixel_free_space.node \
  --ros-args \
  -p input_mode:=ma2
```

A healthy run eventually reports something like:

```text
running; output=2.8 Hz, pipeline=250 ms, valid=...
```

Initial `No MA2 pose near camera timestamp` messages can occur while playback
catches up. They are not a failure if the state subsequently becomes `running`.

### Terminal 2: ZED SVO player

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh

python -m stixel_free_space.ma2_svo_player --ros-args -p scenario:=scen1
```

Set `scenario` to the matching YAML name, such as `scen5`, `scen6`, or
`scen4_2`, and play the corresponding bag in terminal 3. The default is `scen1`.
The player reads `svo_file` and `clap_timestamps` directly from
`~/Documents/Stixel_based_free_space_estimation/free-space-estimation/configs/scenarios`.
Two-frame clap intervals use their midpoint. The mean MA2-minus-SVO clap
difference corrects the published camera timestamps.

Override the config directory or recording root if needed:

```bash
python -m stixel_free_space.ma2_svo_player --ros-args \
  -p scenario:=scen6 \
  -p scenario_dir:=/path/to/configs/scenarios \
  -p data_root:=/path/to/2023-07-11_Multi_ZED_Summer
```

An absolute YAML path also works as `scenario`. The YAML's `start_timestamp`
does not change playback; the player continues to follow the bag's `/clock`.

The MA2 player uses ZED SDK depth mode `NEURAL`.

It publishes:

- `/ma2/zed/left/image`
- `/ma2/zed/depth`
- `/ma2/zed/disparity`
- `/ma2/zed/camera_info`

### Terminal 3: MA2 ROS bag

```bash
source /opt/ros/jazzy/setup.bash

ros2 bag play \
  "$HOME/Documents/Data_For_Stixel_based_Free_space_estimation/2023-07-11_Multi_ZED_Summer/bags/scen1" \
  --clock
```

Do not use `--loop` with the current SVO player because it does not reset its SVO
position when ROS clock jumps backwards.

The MA2 adapter expects:

- lidar: `/lidar_aft/points`
- pose: `/senti_parser/SentiPose`

The SVO timestamps are corrected with the MA2/ZED timestamp offset inside the SVO
player before being published.

## MA2 calibration

MA2 uses the calibration in:

```text
stixel_free_space/ma2_calibration.py
```

This includes:

- lidar -> right ZED camera transform;
- ENU -> NED position transform;
- original MA2 `T_BODY_TO_CAM` and `R_BODY_TO_CAM`;
- expected camera/water-plane height of 1.6 m in the ROS adapter
  (the upstream temporal filter receives `-1.6 m`).

Do not replace these with VRX TF values when running `input_mode:=ma2`.

The current MA2 path still uses a 0.12 m ZED baseline constant. Replacing this with
the exact baseline published from the SVO calibration is a future cleanup item.

## Run against VRX

Start VRX as usual:

```bash
cd ~/Documents/vrx_ws
./start_vrx.sh
```

Then in another terminal:

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh

ros2 launch stixel_free_space vrx.launch.py \
  python_executable:="$VIRTUAL_ENV/bin/python" \
  publish_world_tf:=true \
  rviz:=false
```

The launch attaches to the live VRX topics. The defaults include the front stereo
pair, camera info, WAM-V lidar, TF, and simulation clock.

VRX and MA2 share the upstream estimator but use different input/calibration paths.

## Outputs

The node publishes:

- `/stixel_free_space/overlay`
- `/stixel_free_space/water_mask`
- `/stixel_free_space/depth`
- `/stixel_free_space/markers`
- `/stixel_free_space/stixels`
- `/stixel_free_space/diagnostics`

Useful checks:

```bash
ros2 topic hz /stixel_free_space/overlay
ros2 topic hz /stixel_free_space/stixels
ros2 topic echo /stixel_free_space/diagnostics
```

For image inspection:

```bash
ros2 run rqt_image_view rqt_image_view
```

Use `/stixel_free_space/overlay` and `/stixel_free_space/water_mask`.

## Runtime warnings

These currently do not indicate pipeline failure:

```text
WARNING 'half' is deprecated and will be removed in the future. Use 'quantize' instead.
```

This comes from the newer Ultralytics API used with the unchanged upstream wrappers.

A Numba warning that the installed TBB interface is too old means Numba disables the
TBB threading layer. The pipeline can still run; it is a performance/runtime warning.

## Dependency rules

To keep the runtime reproducible:

1. Do not use `sudo pip`.
2. Do not use `pip install --user` for stixel dependencies.
3. Do not use `--system-site-packages` for the canonical venv.
4. Do not install the original repository's historical requirements into this venv.
5. Do not upgrade NumPy independently.
6. Change dependency versions only in
   `src/stixel_free_space/requirements-runtime.txt`.
7. Re-run `./scripts/setup_stixel_env.sh` after changing pinned dependencies.
8. Keep `pyzed` supplied by the installed ZED SDK; do not try to replace it from
   an unrelated PyPI package.

## Clean rebuild

If the workspace build itself becomes suspect:

```bash
cd ~/Documents/vrx_ws
rm -rf build install log

source scripts/activate_stixel_env.sh
colcon build --merge-install --packages-select lidar_free_space stixel_free_space
source install/setup.bash
```

This does not delete the Python venv.

## Tests

After activating the environment:

```bash
cd ~/Documents/vrx_ws
PYTHONPATH="$PWD/src/stixel_free_space:$PYTHONPATH" \
  python -m pytest src/stixel_free_space/test -q
```

The tests validate adapter geometry/buffering and ROS-facing behavior. They do not
establish segmentation accuracy or GPU throughput.

## Local environment cleanup

Once the canonical `~/venvs/stixel` environment has been created and the MA2 node
has reached `running`, old experiment environments can be removed:

```bash
deactivate 2>/dev/null || true

rm -rf ~/venvs/stixel_numpy2_test
rm -rf ~/venvs/stixel_old
rm -rf ~/venvs/stixel_numpy1_test
```

Only remove names that actually exist. List them first:

```bash
ls -la ~/venvs
```

If your old working environment is also named `~/venvs/stixel`, do not delete it
manually: `setup_stixel_env.sh` recreates that exact directory from scratch.

## Architecture note

The canonical environment intentionally supports both processes:

```text
MA2 SVO/ZED process ──ROS topics──> stixel process
       pyzed                         FastSAM/YOLO/RWPS/stixels
             \___________________________/
                         |
                   one NumPy-2 venv
```

The two processes remain separate ROS nodes, but they no longer require separate
Python environments.
