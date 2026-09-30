# VRX polar lidar free-space estimator

Per-cloud nearest-obstacle boundary from `/wamv/sensors/lidars/lidar_wamv_sensor/points`.
No temporal accumulation, interpolation, or smoothing is applied. Missing, non-finite,
self-masked, and height-rejected cloud measurements alone never become maximum-range clear rays.

## Run

```bash
cd ~/Documents/vrx_ws
source /opt/ros/jazzy/setup.bash
colcon build --merge-install --packages-select lidar_free_space
source install/setup.bash
ros2 launch lidar_free_space free_space.launch.py
```

The default world frame is `gymkhana_task`, matching `start_vrx.sh`'s world selection.
The launch also starts `world_tf`, which subscribes directly to Gazebo Harmonic's
world pose stream using installed `gz.transport13` / `gz.msgs10` Python bindings.
It composes the WAM-V model and base-link poses and publishes timestamped world-to-base
TF. The standard Pose_V-to-TF bridge drops names/timestamps on this particular stream.
Disable the helper with `publish_world_tf:=false` if another node owns that transform.
For a different simulation world set `world_name:=YOUR_WORLD` as well.
Check the simulator TF tree; for another world/frame use
`target_frame:=YOUR_GRAVITY_ALIGNED_FRAME`. Both the cloud-to-world and
cloud-to-`wamv/wamv/base_link` transforms must exist at the cloud timestamp.
Missing TF causes a throttled warning and the cloud is skipped. No assumed static
world transform is substituted. Simulation time is enabled by the launch config.

In RViz add a **MarkerArray** display on `/lidar_free_space/polygons`.
Orange dots are retained obstacle boundaries, green outlines enclose contiguous
observed angular intervals, and translucent green triangles show their approximate
interior. Set RViz's fixed frame to the configured target frame, or a connected frame.
Markers expire after 0.5 simulation seconds and each new cloud clears old polygons.
A paused simulation also pauses marker expiry.

## Outputs

* `/lidar_free_space/boundary`: PointCloud2 with one row per angular bin and float32
  fields `x,y,z,theta,range,status`. Status is 0 UNKNOWN, 1 OBSERVED, or 2 ASSUMED_CLEAR. Unknown ranges
  and x/y coordinates are NaN. Theta is radians in the target frame, from -pi toward
  +pi; ranges are horizontal distances from the transformed lidar origin. Positions
  are absolute target-frame coordinates, projected to `water_level`.
* `/lidar_free_space/polygons`: MarkerArray containing `polygon` LINE_STRIP markers
  (closed polygons), `observed_space` TRIANGLE_LIST markers (fills), and `boundary`
  POINTS. Every disjoint angular run has a separate polygon ID. Single-bin runs only
  produce boundary points. Full-circle coverage closes the boundary without adding
  the origin. Partial runs close through the projected sensor origin. This topic is
  a visualization format, not a custom planner polygon-array interface.

## Parameters

Edit `config/free_space.yaml`, rebuild, and restart, or supply `params_file:=/path/file.yaml`.
Parameters are read-only at runtime to avoid partial configuration changes.

* `min_range`, `max_range`: horizontal limits in metres (defaults 0.1 and 130).
* `min_obstacle_height`, `max_obstacle_height`: metres above `water_level` (0.15 and 3).
* `water_level`: reference plane z in the gravity-aligned target frame (0).
* `angular_resolution`: degrees per bin; must divide 360 (1 degree).
* `self_mask_min`, `self_mask_max`: box bounds in the vessel base frame, in metres.
  Defaults are an initial approximate WAM-V envelope, **not a measured hull model**.
  Inspect and tune them: a large box can also suppress nearby external obstacles.
* `input_topic`, `target_frame`, `base_frame`, `marker_lifetime`: input, transforms,
  and visualization expiry in seconds.

Height filtering compensates for vessel attitude through TF, but a flat water-level
reference does not model local wave elevation. Returns within the height band support
an approximate visibility boundary; vertical beam gaps and angular discretization
mean its interior is not certified obstacle-free. Radial polygon closures include the
sensor near-field and masked vessel region, neither of which constitutes observed
clearance. No-return directions remain unknown unless supported by the assumed-clear mode below.
Vessel footprint/turning clearance must be checked before navigation use.

## Verification

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="src/lidar_free_space:$PYTHONPATH" python3 -m pytest src/lidar_free_space/test -q
```

Tests cover filtering, nearest returns, unknowns, water reference, seam wrapping,
transforms, validation, and the ROS output adapter using synthetic clouds and TF.
Validate live in VRX with buoys, occlusions, and vessel motion before planner use.

## Open-water assumed-clear mode (100 m)

Enabled by default with `enable_assumed_clear: true` and `assumed_clear_range: 100.0`.
The node additionally subscribes to `/wamv/sensors/lidars/lidar_wamv_sensor/scan`.
A scan must match the cloud timestamp and frame exactly. Only positive-infinite
scan ranges are interpreted as explicit no-hit readings; NaN, negative infinity,
finite out-of-range readings, and absent samples do not qualify. Every scan sample
mapped into a candidate bin must qualify, and the horizontal sensor range must
reach 100 m. Existing cloud obstacle ranges always take precedence.

Unknown cloud bins satisfying these checks extend to 100 m with status 2.
Blue translucent triangles (`assumed_clear`) distinguish these regions from green
observed-boundary regions. Triangles touching an assumed endpoint are also blue.
These are simulation assumptions from the scan plane, not verified clearance of
all obstacle heights. The sensor's live positive-infinity convention still needs
verification in this setup; no live scan was available during implementation.
Set `enable_assumed_clear: false` to restore the strictly cloud-only behavior.
The 100 m setting limits assumed extensions; observed obstacles can still appear
out to the configured 130 m `max_range`.
