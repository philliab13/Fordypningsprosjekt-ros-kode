# Shoreline detector experiments

## LiDAR-only projection (no ML)

This separate first-stage experiment takes LiDAR points, transforms them into
the water frame, keeps heights `0 <= h < 2` metres above water, and flattens them
onto the water plane. There is no camera input, model loading, clustering,
polyline extraction, or dependence on the footprint/segmentation nodes.

```bash
colcon build --symlink-install --packages-select nearest_obstacle_shoreline_detector
source install/setup.bash
ros2 launch nearest_obstacle_shoreline_detector lidar_projection.launch.py
```

For bag playback add `use_sim_time:=true`. Parameters are in
`config/lidar_projection.yaml`: `lidar_topic`, `water_frame`, `water_level_m`,
`min_height_m`, and `max_height_m` (exclusive). The defaults are `/lidar_aft/points`,
`ma2_water_plane`, water z=0, and the height interval [0, 2) metres. Filtering is
performed after transforming, so boat tilt and sensor mounting height are
handled by TF rather than by treating raw sensor Z as height above water.
The horizontal water frame must have vertical +Z and be connected to LiDAR TF
at the scan timestamp. This node does not estimate the water plane; provide that
frame and elevation independently. No RWPS or FastSAM is involved.

In RViz set Fixed Frame to `ma2_water_plane` (or another connected frame), add a
**PointCloud2** display, and select:

- `/lidar_water_projection/projected_points`: retained points flattened onto water.
- `/lidar_water_projection/filtered_points`: the same points at their original
  heights, expressed in the water frame, for comparison.

Both clouds carry an extra `height_above_water` field holding the height before
flattening. Use the Intensity color transformer and select this field to see the
original height as color. A top-down view helps inspect point coverage. Use
Decay Time=0 to show the current cloud without accumulation. Only XYZ and this
height field are published; original intensity/ring fields are not retained.
Output stamps match the input scans. Invalid points are removed; missing TF
or invalid input produces empty clouds and a throttled warning. If input stops,
the last cloud remains visible until another message arrives or RViz is reset.

This shows every retained point, including boats, vegetation and points inland.
It does not yet identify the nearest shore or create a barrier.

The projection executable needs only ROS, NumPy, SciPy and TF. The ML runtime
requirements listed below apply to the separate segmentation executable.

## Standalone segmentation-based shoreline barrier node

Runs RWPS + FastSAM water segmentation itself, using the existing implementations
in the `free-space-estimation` checkout. No running stixel node or installed
`stixel_free_space` ROS package is required. No stixels, disparity input, stixel
depth fusion, or pose-message subscription are used.

## Run

```bash
colcon build --symlink-install --packages-select nearest_obstacle_shoreline_detector
source install/setup.bash
ros2 launch nearest_obstacle_shoreline_detector shoreline.launch.py
```

Set `repo_path` in `config/shoreline.yaml` to the upstream repository on the ROS
machine. Set `fastsam_weights` and `rwps_config` if assets are not
in its normal `weights/` and `configs/` directories. Reuse the existing working
CUDA environment: runtime dependencies include PyTorch, Ultralytics, Open3D,
Numba, NumPy, SciPy, and OpenCV. ROS dependencies are in package.xml. The Python
runtime list is provided in requirements-runtime.txt; choose PyTorch matching
your CUDA installation. Model files must already exist; the node does not
implicitly download them.

Standalone mode (default) needs synchronized BGR/RGB left images and registered
`32FC1` optical depth in metres, plus matching CameraInfo, LiDAR, and TF. MA2
input defaults are `/obstacles/source_image`, `/ma2/zed/depth`,
`/ma2/zed/camera_info`, and `/lidar_aft/points`. The MA2 SVO player or another
camera/depth publisher must still run. With remove_boats enabled, run the
footprint detector too: its source-image output selects exactly the frames it
processed, and `/obstacles/boat_mask` supplies the corresponding segmentation. This node does not compute depth from a
raw left/right pair. For VRX, supply a registered depth publisher as well as
changing the topics and frames; the former stixel adapter's internal SGBM is
not used here.

Pass a custom configuration with `params_file:=/path/to/config.yaml`. For bags
and simulation also pass `use_sim_time:=true`. Defaults use `water_frame` =
`ma2_water_plane`, and `fixed_frame` = `map`. These frames must already exist.
`fixed_frame` must be stationary for cross-time motion compensation.
`water_frame` must be horizontal, with water at `water_level_m` (default zero).
Both LiDAR and camera optical frames must be connected to TF at acquisition
time. If MA2 only uses the upstream hardcoded extrinsics, publish their equivalent
TF first. The node does not create frames or publish the RWPS plane as TF.

`camera_height_m=0` derives expected height from camera TF above water_level_m;
set a positive value to override the segmentation height check. TF is still
needed for LiDAR registration and barrier output.

## Implications

- Uses the same RWPS, FastSAM and temporal-filter implementations, but skips all
  stixel processing and does not require image width to divide a stixel count.
  Runtime savings have not been benchmarked.
- CUDA is still required by upstream FastSAM. This node never loads YOLO.
  Running this alongside the full stixel node duplicates inference/model memory.
- `remove_boats=true` uses the footprint detector boat mask after temporal
  water filtering, avoiding a second YOLO instance. The mask must match the exact
  source-image timestamp, frame, and dimensions; it is not resized or reused
  from an older frame. Both 0/1 and 0/255 masks work. An all-zero boat mask is a
  valid no-boats result.
- Image, depth, and boat masks are synchronized before segmentation.
  `sync_queue_size` (default 20) retains input history to accommodate footprint
  inference delay. Increase it if matching fails under load, at the cost of
  memory. Missing masks stop new results; input_timeout_s clears old barriers.
  Malformed or mismatched masks clear the result immediately.
- To run without the footprint detector, set `remove_boats=false` and
  `left_topic=/ma2/zed/left/image`. Boat pixels will then remain in the water mask.
- Failed RWPS fits, missing FastSAM masks, and rejected temporal checks produce
  an empty water mask and clear the barrier. Unlike the full pipeline, this does
  not fall back to unseeded contours after plane failure. Temporal history resets
  after rejected frames, calibration changes, timestamp rewinds, or reset_gap_s.
- A worker handles inference while callbacks retain LiDAR and TF. Only one image
  pair is pending, replacing older pairs under load. Processing rate and output
  latency still depend on segmentation performance.

Optional compatibility mode: set `input_mode=water_mask` to consume
`water_mask_topic` (default `/stixel_free_space/validated_water_mask`) instead of
loading models. Only that mode requires another segmentation publisher. The
validated-mask publisher added to the stixel adapter remains available for it.

## Processing

- Omit the bottom 50 image rows by default, matching the original boundary scan.
  Set `bottom_margin_px=0` to start at the actual bottom. Columns must start in
  water and encounter non-water; blocked starts and all-water columns are unknown.
- Keep LiDAR returns within `boundary_band_px` of the edge. Per strip of
  `column_width_px`, choose the nearest optical-depth cluster with at least
  `min_points` returns within `depth_cluster_m`. Isolated outliers are skipped.
- Back-project the strip median boundary pixel at the measured optical depth,
  transform to the water frame, and flatten z to `water_level_m`. This is an
  approximation of the visible edge using measured range.
- Join only adjacent supported strips closer than `max_segment_m`. Do not close
  the boundary or bridge missing measurements. No range means unknown geometry.
- Match LiDAR by acquisition timestamp, allowing `lidar_max_skew_s` mismatch.
  The scan buffer accommodates segmentation inference delay; increase
  `cloud_buffer_size` if needed. Cross-time TF compensates ego motion but does
  not deskew individual points or compensate moving obstacles.
- Rejected masks, missing inputs/TF, and processing failures clear the barrier.
  Missing-input timeout also clears it. Consumers must enforce freshness using
  the acquisition timestamp and clear old geometry on empty output.

The camera mask and CameraInfo must describe the same undistorted pinhole image.
Boats, docks and vegetation can also form this visible free-space boundary.
The footprint boat mask removes boat pixels when remove_boats is enabled;
this does not classify docks or vegetation as shoreline.

## Outputs

Prefix: `/nearest_obstacle_shoreline_detector`.

| Topic | Type | Contents |
| --- | --- | --- |
| `barrier` | visualization_msgs/MarkerArray | Orange LINE_LIST in water frame, replacing previous result |
| `barrier_json` | std_msgs/String | frame_id, acquisition stamp, status, segments as pairs of XYZ endpoints in metres |
| `water_mask` | sensor_msgs/Image | Accepted standalone fused water mask, or zero on segmentation rejection |
| `boundary_mask` | sensor_msgs/Image | Sparse mono8 edge pixels in the original camera frame |

`segments=[]` clears the barrier. These are geometry outputs; a planner or
costmap consumer must insert the segments as barriers. Marker width is visual.

## Tests

```bash
PYTHONPATH=src/nearest_obstacle_shoreline_detector python -m pytest src/nearest_obstacle_shoreline_detector/test
```

Geometry, input, and mocked segmentation tests run without ROS, CUDA or model weights.
They require NumPy, SciPy, and pytest. Real model inference needs the runtime environment.
ROS tests skip when ROS Python packages are unavailable.
