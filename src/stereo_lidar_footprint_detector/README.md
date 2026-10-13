# ROS2 square footprint adapter

This is a first-test adapter for the footprint repository, based on the geometry
and timestamp helpers in `stixel_free_space`. It publishes a maze-style JSON
FeatureCollection using `std_msgs/msg/String` on `/obstacles/json`.

## Inputs

| Parameter | Default | Contract |
| --- | --- | --- |
| `left_topic` | `/ma2/zed/left/image` | Rectified BGR/RGB(A) image |
| `depth_topic` | `/ma2/zed/depth` | Registered `32FC1` optical-Z depth in meters |
| `camera_info_topic` | `/ma2/zed/camera_info` | Matching rectified projection matrix P |
| `lidar_topic` | `/lidar_aft/points` | PointCloud2 with XYZ and a valid TF frame |
| `baseline_m` | 0 (must be set) | Actual stereo baseline, used by depth uncertainty model |
| `repo_path` | empty (must be set) | Footprint repository containing `free_space/pipeline.py` |
| `weights` | repository `weights/yolo11n-seg.pt` | Optional explicit model path |
| `water_frame` | empty | Optional TF frame with XY spanning the water plane |
| `max_stereo_depth_m` | 10 | Reference stereo validity limit |
| `lidar_max_skew_s` | 0.1 | Maximum LiDAR/camera timestamp difference |
| `require_lidar` | false | Reject frames instead of using stereo alone when LiDAR is absent |
| `device` | empty | Ultralytics device selection; `cpu` is supported |

Image/depth/CameraInfo are approximately synchronized within 30 ms. All three
must have the same optical frame and compatible dimensions. LiDAR is transformed
into that optical frame using timestamped TF, then projected with rectified K
from CameraInfo.P. Use the actual LiDAR-to-camera extrinsics; do not label raw
LiDAR XYZ as camera XYZ. An unavailable required TF rejects the frame with an
error log. Input clocks must already be aligned; the node does not calculate
MA2 clap offsets itself. The existing SVO player handles its own timestamps.

With no `water_frame`, coordinates are `[right, forward]` in camera-local meters,
assuming a level camera. The JSON frame ID is `<optical_frame>/right_forward_plane`:
this is a descriptive synthetic plane, not a published TF frame. With a configured
water frame, polygons use that frame's XY axes and include that frame ID. The
vertical coordinate is discarded. No plane fit is performed.

## Build and run

Use the workspace ROS-compatible inference environment, not the unconfigured
system Python. This adapter avoids cv_bridge's NumPy ABI problem by decoding
supported images directly. It depends on the installed `stixel_free_space`
package for pure geometry helpers, but never instantiates its inference backend.

```bash
cd /home/phillip-andre/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source .venv-stixels/bin/activate
export PYTHONNOUSERSITE=1
python -m colcon build --merge-install --packages-select stereo_lidar_footprint_detector
source install/setup.bash

# Set this to the measured baseline from the camera calibration, in meters.
export STEREO_BASELINE_M=0.12  # Example only: replace for your recording.
ros2 run stereo_lidar_footprint_detector footprint_node --ros-args \
  -p repo_path:=/home/phillip-andre/Documents/Fordypningsprosjekt/stereo-lidar-fusion-footprint-detector \
  -p baseline_m:=$STEREO_BASELINE_M
```

The default topics match `stixel_free_space`'s MA2 SVO player. Start that player,
the LiDAR bag and the required sensor TF publishers separately. Remap topics or
set their parameters for another registered-depth producer. The baseline is not
recoverable from the left CameraInfo alone; use calibration rather than guessing.

## Scope and remaining ROS work

- The node consumes depth; it does not yet implement raw VRX stereo-pair matching.
  The existing `stixel_free_space.core.compute_stereo` is available for that stage.
- JSON is frame-local unless `water_frame` supplies a shared frame. IDs are not
  tracks. Consumers must use the JSON timestamp and expire old observations.
- The adapter is synchronous with small subscription queues. It is intended for
  the first recorded-data test, not verified high-rate operation. A latest-frame
  worker, diagnostics, RViz markers and launch integration can be added later.
- LiDAR matching is nearest-time without per-point motion compensation.
- Keep the original stixel node pointed at its original repository. Its backend
  expects stixels, RWPS and temporal state, which this pipeline does not expose.
  Both pipelines use the package name `free_space`; run them in separate processes.

Validation covers ROS Image decoding and a synthetic registered-frame callback
through the actual footprint backend to published JSON. Live sensor/TF playback
has not been validated.
