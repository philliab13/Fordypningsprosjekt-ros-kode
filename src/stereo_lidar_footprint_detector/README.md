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
