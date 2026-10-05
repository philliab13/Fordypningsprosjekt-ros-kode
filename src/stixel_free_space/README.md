# Stixel free space in VRX

ROS 2 Jazzy adapter for the existing `FreeSpacePipeline` in the separate
`Stixel_based_free_space_estimation/free-space-estimation` checkout. The adapter
imports that checkout directly; it does not copy or change the estimator, replace
FastSAM/YOLO, add BayesOD, or change missing-stixel/depth-fusion rules.

## Setup

The original Python 3.8 requirements cannot be used unchanged with ROS Jazzy's
Python 3.12. Use a dedicated environment with access to ROS system packages:
Install Ubuntu's `python3.12-venv` package first if `python3 -m venv` reports that
`ensurepip` is missing.

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m venv --system-site-packages .venv-stixels
.venv-stixels/bin/python -m pip install -r src/stixel_free_space/requirements-runtime.txt
colcon build --merge-install --packages-select lidar_free_space stixel_free_space
source install/setup.bash
```

CUDA must be visible to PyTorch. The unchanged FastSAM/YOLO wrappers explicitly
request CUDA and half precision. No ZED SDK or MA2 recordings are required.
First use downloads the original `FastSAM-x.pt` and `yolo11n-seg.pt` weights if
missing; the first few processed frames also include compilation/warmup.

### Ubuntu 24.04 and modern CUDA

The original repository documents its tested Ubuntu 20.04 / Python 3.8 / CUDA 12 /
ZED SDK 4.2 environment. That remains the historical MA2 baseline. Its code,
requirements, and README are not modified by this adapter.

This ROS adapter targets **Ubuntu 24.04, ROS 2 Jazzy, Python 3.12**, and the pinned
PyTorch 2.14.1 / torchvision 0.29.1 **CUDA 13.2** wheels. The wheel's CUDA libraries
are version 13.2.2. These are selected explicitly in `requirements-runtime.txt`;
`nvidia-smi` reports the driver's CUDA capability, not the version PyTorch uses.
See the [PyTorch release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.1).
NVIDIA's standalone toolkit is newer (13.4.2 as of 2026-10-05), but the available
stable PyTorch CUDA wheel used here is 13.2. No standalone compiler/toolkit or
system-driver replacement is required for this inference-only adapter.

NumPy stays at 1.26.4 for Jazzy's system `cv_bridge` ABI; Open3D and Numba use
Python-3.12-compatible wheels. FastSAM, YOLO and the stixel estimation logic remain
the original versions. This profile does not validate the MA2 SVO loader, ZED SDK,
or identical numerical results across library versions.

To upgrade an environment created by the earlier ROS adapter setup:

```bash
.venv-stixels/bin/python -m pip install --upgrade -r src/stixel_free_space/requirements-runtime.txt
.venv-stixels/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The diagnostic topic also reports `torch_version`, `torch_cuda_runtime`,
`gpu_name`, and `upstream_pipeline`, so a recorded run identifies its runtime.

## Run against VRX

Start VRX as usual with `./start_vrx.sh`. With the old lidar free-space launch
disabled, start this adapter's world TF helper in the stixel launch. In another
terminal:

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch stixel_free_space vrx.launch.py \
  python_executable:="$PWD/.venv-stixels/bin/python" \
  publish_world_tf:=true rviz:=false
```

The workspace script already opens RViz, hence `rviz:=false`. If a separate lidar
launch is providing `lidar_free_space/world_tf`, set `publish_world_tf:=false` to
avoid two world TF helpers. Set `world_name` and `fixed_frame` for another world.
Stop `stereo_viewer` and any BayesOD inference when benchmarking to avoid extra
stereo computation/GPU contention.

The launch attaches to live topics; it does not start another simulator. The
default inputs are:

* `/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw`
* `/wamv/sensors/cameras/front_right_camera_sensor/optical/image_raw`
* Matching `/optical/camera_info` topics for both cameras
* `/wamv/sensors/lidars/lidar_wamv_sensor/points`
* `/tf`, `/tf_static`, and `/clock`

The workspace's default gymkhana world stops when its scoring task finishes
(320 simulation seconds). Restart VRX and the stixel node for another run.

All input topics and tolerances are parameters. Copy `config/vrx.yaml` and pass
`params_file:=/absolute/path.yaml` to customize them. Parameters are read-only;
restart to change the configuration. `repo_path` selects the upstream checkout.

## Outputs

* `/stixel_free_space/overlay`: BGR camera image with valid stixels. Blue/orange =
  static, red = boat/dynamic flag, yellow = propagated depth. Rejection is labeled.
* `/stixel_free_space/water_mask`: mono8 final mask (0/255), including on rejected frames.
* `/stixel_free_space/depth`: 32FC1 StereoSGBM depth in metres. Invalid pixels are
  zero, following the input convention of the original loader; upstream can set
  depths above 100 m to NaN.
* `/stixel_free_space/markers`: RViz MarkerArray with separate 3-D vertical image
  columns; missing columns are not connected. Markers expire after 5 seconds of
  simulation time by default, which accommodates the current full-pipeline rate.
* `/stixel_free_space/stixels`: PointCloud2, one row per column, including invalid
  columns. All fields are float32: `x,y,z,index,top_px,base_px,depth,depth_variance,
  valid,dynamic,propagated,lidar_measured`. Boolean fields are 0/1. Invalid XYZ is
  NaN; raw depth/variance remain available for inspection. `lidar_measured` follows
  the original flag, which does **not** count stereo-only measurements.
* `/stixel_free_space/diagnostics`: DiagnosticArray containing status, counters,
  current timings and rolling p50/p95 timings.

Images and geometry retain the left image's acquisition stamp and optical frame.
The cloud represents original `(forward,right)` footprints as optical
`(x=right,y=0,z=forward)`: these are a camera-plane visualization, **not a
water-plane projection**. RViz stixel markers additionally use top/base pixels to
recover optical Y. No navigation polygon or occupancy claim is introduced.

The optional `rviz:=true` opens a prepared layout with the overlay, water mask and
stixel markers, fixed to the default VRX left optical frame. In an existing RViz
window select `gymkhana_task` (or the left optical frame), add MarkerArray on
`/stixel_free_space/markers`, and Image on `/stixel_free_space/overlay`. A PointCloud2
display on `/stixel_free_space/stixels` can color by depth/variance/flags.

## What differs from MA2 input

* Aligned VRX stereo images use OpenCV StereoSGBM instead of ZED NEURAL_PLUS.
  This changes depth quality and contributes an explicitly measured runtime.
  Textureless water can produce poor stereo and frequent RWPS/segmentation rejection.
* Intrinsics come from CameraInfo; baseline and lidar extrinsics come from TF.
  The adapter validates an undistorted, aligned pinhole pair with matching intrinsics.
  Arbitrary distorted/rotated real stereo rigs need upstream rectification first.
* VRX's native 1280-pixel width uses **128 stixels**, because the existing algorithm
  requires the width to divide evenly. The 192-column MA2 setting expects 1920 pixels.
  Angular association uses the measured horizontal field of view instead of MA2's 110°.
* Expected water-plane height comes from camera altitude above `water_level` (0 m
  by default). Set positive `camera_height` to use a fixed height instead.
* Timestamped camera poses are converted to the existing planar, backwards-heading
  convention; the camera origin incorporates the body-to-camera lever arm. The
  original temporal model still ignores roll/pitch and assumes planar motion.
* Lidar is transformed from its acquisition time into the image-time optical frame
  through the fixed world frame. A scan must be within `lidar_max_skew_s` (0.1 s).
  Default `require_lidar=true` skips unmatched images. Set false to pass empty arrays
  and observe the original stereo/propagation behavior; stale scans are never reused.

Stereo pairs have a 20 ms maximum skew. One worker owns all pipeline state and has
only one pending pair; new pairs replace pending pairs when inference is slower
than the cameras. There is no fixed processing rate or frame-rate assumption in
the ROS adapter. The original 3-frame mask smoothing and 5-frame propagation limit
remain **frame-based**, so their duration changes with actual processing frequency.
Temporal state resets after gaps observed in the incoming camera stream (default 1 s),
non-increasing processed timestamps, processing exceptions, or the original
Euler-heading wrap discontinuity. Intentional frame replacement by the latest-frame
worker does not count as an input gap.
Restart the node after resetting/reloading the simulation so all synchronizers and
TF history start fresh.

## Benchmark

```bash
ros2 topic echo /stixel_free_space/diagnostics
ros2 topic hz /stixel_free_space/stixels
ros2 bag record /stixel_free_space/stixels /stixel_free_space/diagnostics
```

`stereo_ms` measures StereoSGBM. `pipeline_ms` measures the original process call
with CUDA synchronization. `setup_ms` includes lazy model initialization on the
first frame. `worker_ms` includes preparation, TF waiting, stereo, inference,
visualization and publication calls. `queue_ms` measures time waiting in the
single pending slot. `receive_to_publish_ms` includes both queue and worker time;
it excludes transport before the synchronized callback and subscriber rendering.
`output_age_ros_ms` measures acquisition-to-publication in ROS/simulation time,
which is **not wall time** if Gazebo runs below real time. The output/input rates
are wall-clock rates; neither is an inference-only FPS estimate.

The first 3 successful process calls are excluded from rolling percentile summaries
(they still publish outputs and individual timings). `queue_overwrites`,
`skipped_frames`, `errors`, `segmentation_failures`, `valid_stixels`, `rwps_succeeded`,
`rwps_plane_height_m`, `plane_height_deviation_m`, `final_water_ratio`, and sensor skew
help distinguish throughput from useful results. Pair counters do not count raw
camera messages discarded by the stereo synchronizer or middleware.

For a comparison with cheaper FastSAM masks, restart with `retina_masks:=false`.
To measure without visualization work, use `publish_images:=false publish_markers:=false`.
The default keeps the original full-resolution FastSAM setting.

Known upstream issues remain observable: a FastSAM no-mask result returns the wrong
tuple length; RWPS failure's dummy plane rejects the contour fallback; the temporal
IoU check compares against the current unfiltered mask. Exceptions are reported and
reset temporal state, rather than publishing a fabricated successful result.

## Verification

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="$PWD/src/stixel_free_space:$PYTHONPATH" \
  python3 -m pytest src/stixel_free_space/test -q
```

Tests exercise projection, stereo calibration/scale, pose conversion, bounded
buffering, missing/stale lidar, invalid outputs, and the ROS adapter with an injected
backend. These do not establish segmentation accuracy or GPU throughput; those
require the original models and a running VRX scene.
