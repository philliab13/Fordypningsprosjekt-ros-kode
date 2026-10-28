# Shoreline detector experiments

## LiDAR-only projection (no ML)

This LiDAR-only experiment transforms scans into the water frame, keeps heights
`0 <= h < 2` metres above water, and flattens them onto the water plane. It also
extracts a nearest-supported boundary in angular sectors. It needs no camera,
model loading, or footprint/segmentation node.

```bash
colcon build --symlink-install --packages-select nearest_obstacle_shoreline_detector
source install/setup.bash
ros2 launch nearest_obstacle_shoreline_detector lidar_projection.launch.py
```

For bag playback add `use_sim_time:=true`. Parameters are in
`config/lidar_projection.yaml`: `lidar_topic`, `water_frame`, `water_level_m`,
`min_height_m`, and `max_height_m` (exclusive). The supplied config uses `/lidar_aft/points`,
`ma2_water_plane_up`, water z=0, and the height interval [0, 2) metres. Filtering is
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

The original clouds show every retained point, including boats, vegetation and
points inland. Additional outputs under `/lidar_water_projection` are:

- `boundary_points` (PointCloud2): one representative point per supported sector.
- `barrier` (MarkerArray): orange line segments between adjacent supported sectors.
- `barrier_json` (String): timestamped XYZ segment pairs using the same schema as
  the segmentation shoreline output. To view this line in the footprint BEV,
  run that viewer with `-p shoreline_topic:=/lidar_water_projection/barrier_json`.

In RViz add PointCloud2 for `boundary_points` and MarkerArray for `barrier`.
The supplied config requires `ma2_water_plane_up` TF. With `ma2.launch.py` running,
provide it using the following command, unless another launch already publishes it:

```bash
ros2 run tf2_ros static_transform_publisher \
  --roll 3.141592653589793 --pitch 0 --yaw 0 \
  --frame-id ma2_water_plane --child-frame-id ma2_water_plane_up
```

Sectors cover 360 degrees around the LiDAR origin projected into the water frame.
`sector_width_deg=2` sets their nominal width (rounded to an integer sector count).
Within each sector, returns are sorted by horizontal range; the first range window
of width `range_cluster_m=1` containing at least `min_sector_points=2` returns is
selected. Its median XYZ is the representative point, so a nearer isolated return
cannot hide a farther supported group. Only ranges `min_range_m=1` through
`max_range_m=100` are considered. Adjacent supported sectors are connected only
when their endpoints are within `max_segment_m=3`; empty sectors and larger jumps
stay open. The last and first sectors are adjacent across the angular seam.

This is a nearest obstacle boundary, not a semantic classification of shoreline.
Boats and docks can form it too. Outputs are replaced each scan; missing TF clears
the node's boundary outputs. The optional BEV viewer applies its own hold-last
and ego-compensation behavior. Height filtering still assumes the configured
water frame is horizontal; the MA2 static frame follows vessel roll and pitch.

The projection executable needs only ROS, NumPy, SciPy and TF. The ML runtime
requirements listed below apply to the separate segmentation executable.

## Standalone segmentation-based shoreline barrier node

Runs RWPS + FastSAM water segmentation itself, using implementations bundled in
this ROS package under `nearest_obstacle_shoreline_detector/segmentation`.
No separate source checkout, running stixel node or installed
`stixel_free_space` ROS package is required. No stixels, disparity input, stixel
depth fusion, or pose-message subscription are used.

## Run

```bash
colcon build --symlink-install --packages-select nearest_obstacle_shoreline_detector
source install/setup.bash
ros2 launch nearest_obstacle_shoreline_detector shoreline.launch.py
```

The package includes `config/rwps.json` and `weights/FastSAM-x.pt` (about 139 MiB).
Both are installed into the ROS package share directory. Empty `fastsam_weights`
and `rwps_config` parameters select those bundled assets; set explicit paths to
override them. The former `repo_path` parameter is no longer used and should be
removed from custom parameter files. Reuse the existing working
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
and simulation also pass `use_sim_time:=true`. The supplied MA2 configuration uses
`water_frame=ma2_water_plane_up` and `fixed_frame=piren`.
`shoreline.launch.py` publishes a static transform from `ma2_water_plane` to
`ma2_water_plane_up`, with the same origin and a 180-degree rotation about X.
Run `ros2 launch stixel_free_space ma2.launch.py` to supply the original water
frame and camera extrinsics; bag playback supplies the connection to `piren`.
The flipped water frame still follows the boat's roll and pitch, so this is an
approximation of a horizontal water surface. It does not stabilize the frame.
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
- CUDA is still required by FastSAM. This node never loads YOLO.
  Running this alongside the full stixel node duplicates inference/model memory.
- `remove_boats=true` uses the footprint detector boat mask after temporal
  water filtering, avoiding a second YOLO instance. The mask must match the exact
  source-image timestamp, frame, and dimensions; it is not resized or reused
  from an older frame. Both 0/1 and 0/255 masks work. An all-zero boat mask is a
  valid no-boats result.
- Image, depth, and boat masks are synchronized before segmentation.
  `sync_queue_size` (default 20) retains input history to accommodate footprint
  inference delay. Increase it if matching fails under load, at the cost of
  memory. Missing masks stop new results. With `hold_last_boundary=true`
  (default), input gaps and invalid masks retain the last valid barrier.
- To run without the footprint detector, set `remove_boats=false` and
  `left_topic=/ma2/zed/left/image`. Boat pixels will then remain in the water mask.
- Failed RWPS fits, missing FastSAM masks, and rejected temporal checks produce
  an empty water mask; the last valid barrier is retained by default. Unlike the full pipeline, this does
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
- `hold_last_boundary=true` retains the last valid barrier until another nonempty
  measurement replaces it. Rejected masks, missing LiDAR/TF, and input timeouts
  do not publish empty replacements. Markers have no automatic expiry; JSON
  retains the actual measurement timestamp (it is not refreshed or republished
  as a new measurement). Set `hold_last_boundary=false` to restore immediate
  clearing and `input_timeout_s` expiry. Mask and boundary images still describe
  the current processing result, independently of held barrier geometry.

The camera mask and CameraInfo must describe the same undistorted pinhole image.
Boats, docks and vegetation can also form this visible free-space boundary.
The footprint boat mask removes boat pixels when remove_boats is enabled;
this does not classify docks or vegetation as shoreline.

## Outputs

With `log_timing=true` (default), each standalone processing pass logs its source
timestamp, total elapsed milliseconds, model initialization, RWPS, FastSAM,
temporal filtering, combined segmentation, boat removal, and barrier/output time.
Segmentation includes its three named stages; these timings overlap and should
not be added together. Total time starts when the worker dequeues the image pair,
includes initialization on the first pass, and excludes input synchronization and
queue waiting. FastSAM time includes mask selection and GPU-to-CPU mask transfer.
These are elapsed wall times, independent of playback speed. Rejected/error passes
are logged too; stages that did not complete show `n/a`. First-pass model/CUDA
warmup and Numba compilation can make the initial timings much larger. Set
`log_timing=false` to silence these per-pass INFO logs.

Prefix: `/nearest_obstacle_shoreline_detector`.

| Topic | Type | Contents |
| --- | --- | --- |
| `barrier` | visualization_msgs/MarkerArray | Orange LINE_LIST in water frame, replacing previous result |
| `barrier_json` | std_msgs/String | frame_id, acquisition stamp, status, segments as pairs of XYZ endpoints in metres |
| `water_mask` | sensor_msgs/Image | Accepted standalone fused water mask, or zero on segmentation rejection |
| `boundary_mask` | sensor_msgs/Image | Sparse mono8 edge pixels in the original camera frame |

With holding disabled, `segments=[]` clears the barrier. These are geometry outputs; a planner or
costmap consumer must insert the segments as barriers. Marker width is visual.

## Tests

```bash
PYTHONPATH=src/nearest_obstacle_shoreline_detector python -m pytest src/nearest_obstacle_shoreline_detector/test
```

Geometry, input, and mocked segmentation tests run without ROS, CUDA or model weights.
They require NumPy, SciPy, and pytest. Real model inference needs the runtime environment.
ROS tests skip when ROS Python packages are unavailable.
