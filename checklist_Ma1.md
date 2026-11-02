# Checklist MA1 initial test

## Recording data

The first test is to make sure I am able to record data from the rosbag, and optionally the SVO file in addition. Saving SVO while the ZEd is running is done through: 

    ros2 service call /zed/zed_node/start_svo_rec zed_msgs/srv/StartSvoRec "{svo_filename: '/path/to/svo/file/file.svo2', compression_mode: <4 for best detail, 2 if it produces too much data >}"

Then make sure to clap infront of both zed camera and ma1 camera to test if they are synced. Also check the time difference in terminal with command:
    
```bash
python3 - <<'PY'
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
node = rclpy.create_node('camera_stamp_check')
latest = {}

def receive(msg, camera):
    latest[camera] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

subscriptions = [
    node.create_subscription(Image, topic,
                             lambda msg, key=key: receive(msg, key),
                             qos_profile_sensor_data)
    for key, topic in [
        ('zed', '/zed/zed_node/rgb/color/rect/image'),
        ('ma1', '/camera_eo/image_raw'),
    ]
]

def report():
    if len(latest) == 2:
        print(f"Latest image stamps: ZED - MA1 = {(latest['zed'] - latest['ma1']) * 1000:.1f} ms", flush=True)
    else:
        print(f'Waiting for both cameras; received: {list(latest)}', flush=True)

timer = node.create_timer(1.0, report)
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
```

Run in a terminal with ROS sourced. This compares the latest received image
header timestamps, so different frame rates and transport delays affect the
result. It does not measure true exposure synchronization. Use the visible clap
in both recordings to check alignment of the actual images. Ctrl+C stops it.

## Measure frames and/or find out if the frames are already being published

When this is done, we have a baseline recording I can work on. Now I need to see if there are all the frames I need. The frames I need are:

1. **MA1 vessel reference (`base_link`):** find its physical origin and axis
   directions before measuring offsets.
2. **Vessel to LiDAR:** measure X/Y/Z and roll/pitch/yaw from `base_link` to
   the selected LiDAR's mounting/reference frame, such as `os_sensor`.
   Keep the driver's existing `os_sensor -> os_lidar` and `os_sensor -> os_imu`
   transforms. Check the actual point-cloud `header.frame_id`, especially when
   using both front and rear LiDARs; they need distinct frame names.
3. **Vessel to ZED:** measure X/Y/Z and roll/pitch/yaw from `base_link` to
   `zed_camera_link` (confirm its actual name). Keep the wrapper's internal
   transforms connecting it to the image optical frame.
4. **Vessel to waterplane:** confirm the vessel origin is 0.20 m above water.
   With Z upward, a static vessel-to-waterplane translation is `[0, 0, -0.20]`.
   Record the water frame's axis convention. The shoreline/LiDAR projection
   nodes expect Z upward; a static transform follows vessel roll and pitch.
   A level waterplane requires attitude compensation.
5. **Stationary reference, when required:** confirm `piren -> base_link`
   is available for nodes using `fixed_frame=piren`.

Inspect connectivity with `ros2 run tf2_tools view_frames`. Sensor frame names
come from message headers, not topic namespaces. The existing MA2 launch uses
`p0_floor`, `ma2_zed_left_optical`, and `ma2_water_plane`; these do not automatically
connect the live MA1 sensors.

The footprint detector and BEV viewer obtain LiDAR-to-camera calibration from
TF using the sensor message frame IDs. Publish the measured camera mounting and
confirmed LiDAR mounting transforms; no Python calibration matrix needs editing.

Then when I have those, we could try a preliminary test with the wrong frames just to see that the system is running. That should be done with the following commands. Need to make sure all dependencies are installed, you need the two venvs:

1. **Stixel environment:** `~/venvs/stixel`, activated from the workspace with
   `source scripts/activate_stixel_env.sh`.
2. **Footprint environment:** `<workspace>/.venv-footprint`, activated with
   `source .venv-footprint/bin/activate`. See
   `src/stereo_lidar_footprint_detector/requirements-runtime.txt` for dependencies.
   Use separate terminals for the two environments. Build Python packages with
   the intended environment's interpreter; activating a venv does not change an
   already-installed executable's interpreter.

For the footprint detector, in a new terminal:

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source .venv-footprint/bin/activate
source install/setup.bash
ros2 run stereo_lidar_footprint_detector footprint_node --ros-args \
  -p baseline_m:=0.12 \
  -p left_topic:=/zed/zed_node/rgb/color/rect/image \
  -p depth_topic:=/zed/zed_node/depth/depth_registered \
  -p camera_info_topic:=/zed/zed_node/rgb/color/rect/camera_info \
  -p lidar_topic:=/front_lidar/ouster/points \
  -p water_frame:=ma1_water_plane
```

The `0.12` m baseline is for the ZED 2; confirm the connected model. This command
assumes you have published `ma1_water_plane` and connected it to the camera.
Temporary mounting estimates may be used for a startup check, but the resulting
geometry is not calibrated until the TF mounting values are measured correctly.

For the standalone LiDAR-to-water projection, in another sourced terminal:

```bash
ros2 run nearest_obstacle_shoreline_detector lidar_projection_node --ros-args \
  -p lidar_topic:=/front_lidar/ouster/points \
  -p water_frame:=ma1_water_plane \
  -p water_level_m:=0.0
```

This assumes `ma1_water_plane` has Z upward and is connected to the LiDAR frame.
For a live test, use the default `use_sim_time=false`; for bag playback with a
published `/clock`, pass `-p use_sim_time:=true` to each processing node.

And make sure you have the SDK installed for running the SVO recording. 

## Startup procedure

This procedure uses the real MA1, live front Ouster and live ZED 2. Run each
long-running command in its own terminal. Do not start `ma2_svo_player`, MA2
frame publishers, or `ros2 bag play`. Use `use_sim_time=false` throughout.

Before testing, start the vessel's existing navigation and front Ouster drivers
using the onboard procedure. Confirm `/front_lidar/ouster/points` is publishing
and `piren -> base_link` is available for the shoreline overlay. Do not start
duplicate driver or TF publishers if they are already running.

**Calibration still required:** measure the camera mounting and confirm the
water offset/axis directions. Both `footprint_node` and `bev_node` use the TF
mounting chain. Missing sensor TF rejects frames with selected LiDAR scans.
For motion compensation, add `-p fixed_frame:=piren` to both nodes once the
stationary `piren` frame is connected; otherwise scans use image-time mounting
TF without compensating scan-to-image vessel motion.

### 1. Build packages when code or calibration changes

Stixel/shared helpers and LiDAR projection:

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh
python -m colcon build --merge-install \
  --packages-select stixel_free_space nearest_obstacle_shoreline_detector
source install/setup.bash
```

Footprint package, in a separate terminal:

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source .venv-footprint/bin/activate
python -m colcon build --merge-install \
  --packages-select stereo_lidar_footprint_detector
source install/setup.bash
```

Ensure the model file exists at
`src/stereo_lidar_footprint_detector/weights/yolo26n-seg.pt` before building.
Omitting `weights` below selects the installed package's model, avoiding a
working-directory-dependent `weights:=yolo26n-seg.pt` path.

### 2. Start the live ZED wrapper

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh
source install/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed2 \
  publish_tf:=false \
  publish_map_tf:=false
```

The wrapper publishes images, registered depth, camera info, and internal camera
frames. Its tracking TF is disabled because the vessel mounting supplies the
camera's parent; this avoids competing `odom -> camera` and `base_link -> camera`
transforms.

### 3. Publish the MA1 sensor mounting and water frames

In each TF terminal, first run:

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Confirmed front LiDAR mounting (unless already published onboard):

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 2.0 --y 1.0 --z 0.0 \
  --roll 0 --pitch 0 --yaw 0 \
  --frame-id base_link --child-frame-id os_sensor
```

Camera mounting, in another terminal. Set these six variables to the measured
values first (metres and radians); the checks prevent publishing unset values:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x "${CAM_X:?Set measured camera X}" \
  --y "${CAM_Y:?Set measured camera Y}" \
  --z "${CAM_Z:?Set measured camera Z}" \
  --roll "${CAM_ROLL:?Set measured camera roll}" \
  --pitch "${CAM_PITCH:?Set measured camera pitch}" \
  --yaw "${CAM_YAW:?Set measured camera yaw}" \
  --frame-id base_link --child-frame-id zed_camera_link
```

Waterplane, in another terminal. This assumes `base_link` is Z upward with its
origin 0.20 m above water; confirm before use:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z -0.20 \
  --roll 0 --pitch 0 --yaw 0 \
  --frame-id base_link --child-frame-id ma1_water_plane
```

Here `ma1_water_plane` is already Z upward: no separate 180-degree waterplane
flip is needed. If `base_link` uses different axes, adjust this rotation and
translation accordingly. This static plane follows the vessel's roll and pitch.

Check the tree in another sourced terminal:

```bash
ros2 run tf2_tools view_frames
```

### 4. Start the footprint detector

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source .venv-footprint/bin/activate
source install/setup.bash
ros2 run stereo_lidar_footprint_detector footprint_node --ros-args \
  -p use_sim_time:=false \
  -p baseline_m:=0.12 \
  -p max_stereo_depth_m:=20.0 \
  -p lidar_max_skew_s:=0.5 \
  -p left_topic:=/zed/zed_node/rgb/color/rect/image \
  -p depth_topic:=/zed/zed_node/depth/depth_registered \
  -p camera_info_topic:=/zed/zed_node/rgb/color/rect/camera_info \
  -p lidar_topic:=/front_lidar/ouster/points \
  -p water_frame:=ma1_water_plane
```

### 5. Start the LiDAR shoreline projection

```bash
cd ~/Documents/vrx_ws
source scripts/activate_stixel_env.sh
source install/setup.bash
ros2 run nearest_obstacle_shoreline_detector lidar_projection_node --ros-args \
  -r __node:=lidar_water_projection \
  -p use_sim_time:=false \
  -p lidar_topic:=/front_lidar/ouster/points \
  -p water_frame:=ma1_water_plane \
  -p water_level_m:=0.0
```

### 6. Start the footprint BEV visualization

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source .venv-footprint/bin/activate
source install/setup.bash
ros2 run stereo_lidar_footprint_detector bev_node --ros-args \
  -p use_sim_time:=false \
  -p lidar_topic:=/front_lidar/ouster/points \
  -p shoreline_topic:=/lidar_water_projection/barrier_json \
  -p shoreline_fixed_frame:=piren \
  -p gnss_enabled:=false
```

GNSS comparison is disabled to avoid loading the MA2 reference tracks and MA2
body-frame defaults. The shoreline overlay needs the camera and waterplane
connected to `piren`; enable GNSS separately after configuring MA1 inputs.

### 7. Start the frame review window

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
source .venv-footprint/bin/activate
source install/setup.bash
ros2 run stereo_lidar_footprint_detector frame_review_node --ros-args \
  -p use_sim_time:=false
```

The review window needs both the footprint detector and BEV node running to
assemble its camera, YOLO, LiDAR-overlay and BEV panels.

### 8. Optional image and RViz windows

In fresh terminals, source ROS and the workspace as above, then run either:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/debug/yolo_overlay`, `/debug/footprint_bev`, or
`/debug/bev_lidar_overlay` in the image viewer.

```bash
rviz2
```

Use `base_link` as RViz Fixed Frame for a vessel-relative view, or `piren` when
that stationary frame is connected. Add TF and the front LiDAR PointCloud2.

### 9. Record the live test

Start the rosbag recorder with the required camera, LiDAR, navigation and TF
topics; optionally start SVO recording using the service under **Recording
data** above. Neither recording process replaces the live publishers. Stop SVO
recording explicitly before shutting down the ZED wrapper:

```bash
ros2 service call /zed/zed_node/stop_svo_rec std_srvs/srv/Trigger
```

## Clone recursive

```bash
git clone --recurse-submodules <repository-url>
```
