# Camera footprint overlay

Select `/debug/obstacle_overlay` in `rqt_image_view` to see the obstacle
polygons published in `/obstacles/json` projected onto the rectified camera
image as orange outlines. `/debug/yolo_overlay` provides the YOLO annotations.

With `water_frame` configured, polygons are projected from that frame's z=0
plane using the current camera transform. Without it, the JSON's synthetic
camera-right/forward plane has no height; each square is displayed at the
height inferred from its detection's bottom edge and fused depth. These
outlines show estimated square footprints, not measured 3D hulls.

After rebuilding and restarting the node, open the viewer with:

```bash
ros2 run rqt_image_view rqt_image_view /debug/obstacle_overlay
```

## Self-contained footprint detector

The YOLO wrapper, per-frame pipeline, sensor-depth fusion, footprint geometry,
and JSON helpers live in `stereo_lidar_footprint_detector/detection`. The node
no longer imports the separate footprint repository or modifies `sys.path`.
Remove `-p repo_path:=...` from old commands; that parameter is no longer used.
See `DETECTION_SOURCES.md` for the source revision. The `stixel_free_space`
package in this workspace still supplies shared ROS helpers and MA2 calibration.

For example, after building with your chosen footprint environment and sourcing
the workspace, run:

```bash
ros2 run stereo_lidar_footprint_detector footprint_node --ros-args \
  -p use_sim_time:=true \
  -p baseline_m:=0.12 \
  -p max_stereo_depth_m:=20.0 \
  -p lidar_max_skew_s:=0.5 \
  -p water_frame:=ma2_water_plane
```

Empty `weights` selects `weights/yolo26n-seg.pt` in the installed package share
directory. An explicit `-p weights:=/absolute/path/to/model.pt` overrides it;
relative paths are resolved from the process's working directory. Missing files
produce a startup error rather than an implicit model download.

Model files are local assets, excluded from Git by `weights/.gitignore`. The
existing workspace checkpoint has been copied there on this machine. On another
machine, obtain the checkpoint and put it in
`src/stereo_lidar_footprint_detector/weights/yolo26n-seg.pt` before building, or
pass an explicit existing path. Available local checkpoints are installed by
`setup.py`. Runtime Python dependencies are listed in `requirements-runtime.txt`;
the existing `.venv-footprint` contains them. No environment changes are needed
for this migration. Topic schemas, footprint geometry, GNSS handling, and boat
class selection are unchanged.

## Build environment

When building inside `.venv-footprint`, use the workspace's compatible setuptools
version before invoking colcon with that environment's Python:

```bash
python -m pip install setuptools==78.1.0
.venv-footprint/bin/python -m colcon build --merge-install --packages-select stereo_lidar_footprint_detector
```

Setuptools 84.0.0 does not accept `setup.py develop --uninstall`, which colcon
uses when switching an existing symlink build to a regular installation. This
causes `error: option --uninstall not recognized`; pinning the build environment
as above fixes it without deleting the workspace's build or install directories.

Run builds from the workspace root using `.venv-footprint/bin/python -m colcon`.
The build interpreter determines the installed nodes' shebangs. Building with
system Python makes `ros2 run` use system Python even when a virtual environment
is active, bypassing the virtual environment's NumPy, OpenCV and inference dependencies.

## Local BEV comparison

The detector also publishes `/obstacles/boat_mask` (the actual combined boat
segmentation, mono8), `/obstacles/camera_info`, and `/obstacles/source_image`
(the original rectified RGB image), all stamped with the source SVO image
timestamp. The separate `bev_node` subscribes to those, `/obstacles/json`
and `/lidar_aft/points`. It matches detector messages by exact timestamp and
projects the nearest buffered LiDAR scans into that image to select boat points.
It does not run YOLO again or select points just by bounding boxes.

Build and source the workspace, then run alongside the detector and SVO/bag playback:

```bash
.venv-footprint/bin/python -m colcon build --packages-select stereo_lidar_footprint_detector --symlink-install --merge-install
source install/setup.bash
ros2 run stereo_lidar_footprint_detector bev_node --ros-args -p use_sim_time:=true
ros2 run rqt_image_view rqt_image_view /debug/footprint_bev
ros2 run rqt_image_view rqt_image_view /debug/bev_lidar_overlay
```

Restart the rebuilt detector as well, so the mask and calibration topics exist.
Use the same `use_sim_time` setting as playback. Coloured dots show boat LiDAR
returns, with one colour per selected scan; orange outlines with cyan fill show
the published footprints. The green
triangle marks the camera origin, with right/forward axes in metres and equal
scales. This is measured surface support, not an independent complete hull or
ground-truth footprint; the detector may use the same LiDAR returns for depth.

`/debug/bev_lidar_overlay` shows the same selected scans projected onto the
exact RGB frame used for that detector result. Bright points with dark borders
are the boat points used in the BEV; faint dots show the remaining projected
points from those scans. Both views share the scan colours and legends:
`S1`, `S2`, etc. rank scans by timestamp proximity to that image, and each label
shows the signed time offset and selected boat-point count. Colours identify
scans within each frame, not persistent scan or object tracks. Up to six scan
labels are displayed; the status line reports the total scan count.

The RGB and BEV images have the same source timestamp and use identical motion
compensation when enabled. Missing RGB data skips the comparison rather than
overlaying on a different frame. The source image is republished by the detector
so inference delays do not require guessing which live camera frame to use.
Restart both the detector and BEV node after rebuilding to enable this topic.
`rgb_topic` and `rgb_output_topic` configure the input and output image topics.

When the detector has `water_frame` configured (for example `ma2_water_plane`),
the node transforms the selected 3D points into that exact frame, drops Z to zero,
and rotates/translates both points and polygons into a local camera-centered BEV.
This preserves horizontal positions; it does not intersect camera rays with the
water plane. Without `water_frame`, the comparison matches the detector's synthetic
optical X/Z plane, explicitly labelled in the view; configure a water frame for
gravity-aligned flattening.

Parameters (all topic names can also be changed):

| Parameter | Default | Meaning |
|---|---|---|
| `scan_count` | `3` | Maximum nearest scans to combine |
| `max_skew_s` | `0.2` | Maximum absolute scan/image timestamp difference |
| `scan_wait_s` | `0.2` | Wall-clock wait after detector messages start arriving, allowing later scans into the buffer |
| `fixed_frame` | empty | An available stationary TF frame, such as `odom`, for ego-motion compensation |
| `range_m` | `60.0` | Fixed square view width and height in metres |
| `size_px` | `900` | Output image width and height |

The scan buffer contains 100 scans. Selection uses the source timestamps, not
arrival order; the output reports the actual scan count and signed skews in ms.
It uses whichever scans are buffered after the wait, so future scans are not
guaranteed during paused or slow playback. Missing scans produce an empty point
overlay, rather than reusing stale measurements.

### Recorded GNSS references

GNSS is enabled by default for the target-undocking recording
`2023-07-11_12-20-43_28170706_HD1080_FPS15.svo`. Labeled crosses show receiver
positions alongside the LiDAR and footprints. Receiver names are retained because
some receivers moved between boats, obstacles and the quay during data gathering.
These are horizontal antenna reference positions, not hull centers or outlines.

Paths below are relative to
`~/Documents/Data_For_Stixel_based_Free_space_estimation/2023-07-11_Multi_ZED_Summer`:

| Parameter | Default file |
|---|---|
| `spatial_dual_file` | `Spatial Dual/SpatialDualLog_23-07-11_10-21-15/GoogleEarthTrack.kml` |
| `kbox2_file` | `KBox2/LOG00054.kml` |
| `kbox1_file` | `KBox1/LOG00030.kml` |
| `ntbox_file` | `NTBox/gnssdump-20230710173603.kml` |

Each parameter accepts an absolute KML or RTKLIB `.pos` path (and expands `~`).
Set a file parameter to an empty string to omit that receiver. Untimed KML files,
including these KBox/NTBox exports, automatically use the sibling `.pos` file.
Spatial Dual uses timestamped KML Point placemarks directly. Untimed LineStrings
are never treated as simultaneous positions. Startup logs identify the files
actually loaded; missing or invalid files produce a warning and omit that receiver.

| Parameter | Default | Meaning |
|---|---|---|
| `gnss_enabled` | `true` | Enable recorded reference overlays |
| `gnss_pose_topic` | `/navigation/pose` | MA2 `PoseStamped` source |
| `gnss_body_frame` | `p0_floor` | Body frame whose pose the topic describes; TF supplies its camera extrinsics |
| `gnss_pose_axes` | `ned` | Pose position and orientation reference axes, `ned` or `enu` |
| `piren_lla` | `[63.4389029083, 10.39908278, 39.923]` | WGS84 local origin in degrees/degrees/metres |
| `gnss_max_gap_s` | `2.0` | Maximum bracketing GNSS sample interval |
| `pose_max_gap_s` | `0.5` | Maximum bracketing MA2 pose interval |
| `gps_utc_offset_s` | `18.0` | GPST minus UTC for this July 2023 dataset; applies only to GPST `.pos` inputs |
| `gnss_time_offset_s` | `0.0` | Offset added to GNSS timestamps to align them with camera timestamps |

All values are startup parameters. Positions are converted through WGS84 ECEF
into Piren NED and transformed using interpolated MA2 position, quaternion SLERP,
and body-to-camera TF. GNSS altitude is excluded from this horizontal comparison.
GNSS positions then use the same plane/basis as the existing BEV geometry.
The pose position and quaternion must describe `gnss_body_frame` relative to the
selected geographic axes and Piren origin; the topic's frame name alone does not
establish those conventions. The default NED convention follows `/navigation/pose`
for this scenario; the existing stixel node's ENU handling of `/senti_parser/SentiPose`
is a separate input convention.

No GNSS or pose extrapolation is performed. Missing pose/TF leaves the sensor BEV
available with a status message; tracks outside their time coverage are omitted.
For these default files NTBox ends at 10:22:00 UTC. The status reports the number
of time-valid receivers; some may be outside the configured BEV range.

For example, override a file or disable GNSS when testing another scenario:

```bash
ros2 run stereo_lidar_footprint_detector bev_node --ros-args \
  -p use_sim_time:=true -p ntbox_file:=/absolute/path/to/other.pos
ros2 run stereo_lidar_footprint_detector bev_node --ros-args -p gnss_enabled:=false
```

The output retains the camera source timestamp, so saving `/debug/footprint_bev`
and the matching `/debug/bev_lidar_overlay` captures the same detector frame.

The node uses the same MA2 LiDAR-to-camera calibration as the detector. With
`fixed_frame` set, it additionally uses time-dependent camera TF to transform
each scan to the image time. Missing required TF rejects the frame after a short
retry window. Without this setting, scans use static calibration and the image
is labelled `uncompensated`. Only ego motion is compensated; moving target boats
can still smear across scans. Use `scan_count:=1` or a tighter time window to
inspect that effect. The combined mask covers all detected boats; points are
shown as one set, without per-instance association or hidden-surface filtering.

The BEV viewer also subscribes to
`/nearest_obstacle_shoreline_detector/barrier_json` and draws shoreline segments
in red on `/debug/footprint_bev`. Each endpoint is transformed from the JSON
`frame_id` into the stationary `shoreline_fixed_frame` (default `piren`) at the
measurement timestamp. The world coordinates are retained until a new valid
measurement replaces them. For each displayed image, the viewer transforms the
held endpoints into the camera frame at that image's acquisition timestamp,
then projects them into the same local right/forward basis as the footprints.
This compensates vessel translation and rotation while preserving segment gaps.
The status shows the measurement age and labels the shoreline `ego-compensated`.

Empty results and input timeouts do not clear the stored geometry. Missing TF
for a new measurement keeps the previous world outline and retries the update
on later frames. Missing current camera TF hides the overlay for that frame,
retaining the world geometry for retry. There is no camera-relative fallback.
The viewer caches 20 completed frames so late inference can redraw both its
matching frame and the current frame. Measurements newer than the displayed
image are not overlaid. This holds one shoreline estimate; it does not fuse
measurements into a map.

`shoreline_fixed_frame` must be a stationary TF frame connected to the shoreline
source at measurement time and to the camera at display time. This parameter is
independent of `fixed_frame`, which controls the existing LiDAR scan compensation.
The other BEV overlays continue to work without the shoreline detector. Set
`shoreline_topic` to another JSON topic or an empty string to disable the
subscription. Run this viewer with `-p use_sim_time:=true` during bag playback.

## Frame review window

Run the review window alongside the detector, BEV node and playback. It replaces
the rqt image windows for inspecting completed results:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run stereo_lidar_footprint_detector frame_review_node --ros-args \
  -p use_sim_time:=true
```

The window shows a camera panel beside the BEV, with a selector for the original
camera image, YOLO overlay or LiDAR overlay. All four images must have exactly
the same source timestamp before a frame becomes available. The default inputs
are `/obstacles/source_image`, `/debug/yolo_overlay`, `/debug/bev_lidar_overlay`
and `/debug/footprint_bev`. Start both the detector and BEV node to provide them.

| Control | Behavior |
|---|---|
| Space / Pause | Freeze the displayed selection; processing and subscriptions continue |
| Space / Resume live | Jump to the newest completed result and follow incoming results |
| Left / Previous | Pause and select the previous retained result |
| Right / Next | Pause and select the next retained result; at the end, wait for one new complete result |
| L / Live | Jump to the newest completed result |
| S / Save | Export the displayed result, all four panels and metadata |

The status shows the exact timestamp in nanoseconds, position in retained history,
compressed buffer size, pending matches and discarded records. When an incomplete
record expires, the missing panel names are shown. Playback can stay running while
review is paused. History older than the memory budget is evicted, while the paused
selection remains pinned. Stepping can therefore skip evicted results. This reviews
completed detector results; it does not force the detector to process every SVO frame.

Snapshots go into `frame_snapshots` under the directory from which the viewer was
started. Each save creates a unique timestamp-named directory containing
`camera.png`, `yolo.png`, `lidar.png`, `bev.png`, a four-panel `comparison.png`, and
`metadata.json` with source timestamp, topics, frame IDs, dimensions and compression
settings. Saving runs in a background thread. Images in memory are JPEG compressed;
the exported PNGs retain that compression's artifacts and are not lossless originals.

| Parameter | Default | Meaning |
|---|---|---|
| `camera_topic` | `/obstacles/source_image` | Original camera panel |
| `yolo_topic` | `/debug/yolo_overlay` | YOLO panel |
| `lidar_topic` | `/debug/bev_lidar_overlay` | LiDAR camera panel |
| `bev_topic` | `/debug/footprint_bev` | BEV panel |
| `memory_limit_mb` | `512` | Maximum compressed image payload in MiB, including incomplete records |
| `jpeg_quality` | `90` | Buffer JPEG quality, 1–100 |
| `incomplete_timeout_s` | `10.0` | Wall-clock timeout for matching all four panels, independent of paused simulated time |
| `max_pending_frames` | `30` | Maximum incomplete timestamps to retain |
| `snapshot_directory` | `./frame_snapshots` | Snapshot export location |

Qt uses the installed PyQt5 binding (`python3-pyqt5`). GUI display and decoded
images add memory beyond the compressed payload limit. Use a budget large enough
for the selected frame plus incoming frames. For example:

```bash
ros2 run stereo_lidar_footprint_detector frame_review_node --ros-args \
  -p use_sim_time:=true -p memory_limit_mb:=256 \
  -p snapshot_directory:=/tmp/frame_snapshots
```

# Synthetic obstacle for planner testing

Build and run this publisher without starting the detector, cameras, LiDAR,
YOLO, or bag playback:

```bash
source /opt/ros/jazzy/setup.bash
.venv-footprint/bin/python -m colcon build --merge-install --packages-select stereo_lidar_footprint_detector
source install/setup.bash
ros2 run stereo_lidar_footprint_detector test_obstacle_node
```

This publishes one boat (`class_id: 8`, `class_name: boat`, confidence 1.0)
at 5 Hz on the existing `/obstacles/json` (`std_msgs/msg/String`) topic,
with the detector's FeatureCollection/Polygon schema and fresh ROS timestamps.
It also reuses `/obstacles/markers` for RViz. Stop the real detector while using
this publisher to avoid mixing real and synthetic detections.

The 2 m square is centred 10 m ahead of the frame origin, so its nearest edge
is 9 m away. The default frame is `ma2_water_plane`; forward defaults to
negative X (180 degrees), matching the approximate forward direction of the
existing MA2 camera calibration. Its default centre is therefore (-10, 0).
The footprint stays fixed relative to that frame as the vessel moves.
For a frame with positive X forward, set `forward_yaw_deg:=0.0` and the
appropriate existing frame ID. No TF frames are created by this node; retain
the existing TF providers if downstream consumers require transforms.

```bash
ros2 run stereo_lidar_footprint_detector test_obstacle_node --ros-args \
  -p distance_m:=10.0 -p side_length_m:=2.0 -p publish_rate_hz:=5.0
ros2 topic echo /obstacles/json std_msgs/msg/String --once
```

Keep the existing planner bridge/subscribers running. This node replaces the
perception publisher only; the planner bridge is not provided in this workspace.
Use wall time by default when no `/clock` provider is running. For simulation
or playback with an active clock, pass `-p use_sim_time:=true` consistently.
