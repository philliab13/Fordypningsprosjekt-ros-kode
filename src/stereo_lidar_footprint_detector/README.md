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

## Build environment

When building inside `.venv-footprint`, use the workspace's compatible setuptools
version before invoking colcon with that environment's Python:

```bash
python -m pip install setuptools==78.1.0
python -m colcon build --merge-install --packages-select stereo_lidar_footprint_detector
```

Setuptools 84.0.0 does not accept `setup.py develop --uninstall`, which colcon
uses when switching an existing symlink build to a regular installation. This
causes `error: option --uninstall not recognized`; pinning the build environment
as above fixes it without deleting the workspace's build or install directories.

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
colcon build --packages-select stereo_lidar_footprint_detector --symlink-install --merge-install
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

The node uses the same MA2 LiDAR-to-camera calibration as the detector. With
`fixed_frame` set, it additionally uses time-dependent camera TF to transform
each scan to the image time. Missing required TF rejects the frame after a short
retry window. Without this setting, scans use static calibration and the image
is labelled `uncompensated`. Only ego motion is compensated; moving target boats
can still smear across scans. Use `scan_count:=1` or a tighter time window to
inspect that effect. The combined mask covers all detected boats; points are
shown as one set, without per-instance association or hidden-surface filtering.
