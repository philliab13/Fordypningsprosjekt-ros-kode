# Bundled footprint detector sources

The `stereo_lidar_footprint_detector/detection` modules were copied from the local
`Fordypningsprosjekt/stereo-lidar-fusion-footprint-detector` checkout at commit
`a67c1d389206c6e067b87bed3bd0a91c69ef5576` (clean working tree).

- `free_space/pipeline.py` → `detection/pipeline.py`
- `free_space/obstacles.py` → `detection/obstacles.py`
- `free_space/yolo.py` → `detection/yolo.py`
- `free_space/outputs.py` → `detection/outputs.py`

Only imports were changed to package-relative imports. Algorithm behavior,
including the existing boat-only YOLO class selection, was preserved.
Offline SVO readers (`zed.py`), dataset loaders and plotting tools are not needed
by this ROS node, which consumes registered image/depth messages.
The source checkout is recorded for provenance and is not read at runtime or
build time. ROS helpers still come from `stixel_free_space`, which is a package
inside the same `vrx_ws` repository. The ROS adapters obtain LiDAR-to-camera
calibration from TF rather than importing the MA2 calibration matrix.

The existing workspace `yolo26n-seg.pt` was copied to `weights/yolo26n-seg.pt` as a
regular local file. Checkpoints in that directory are ignored by Git and are
installed when present at build time. Third-party libraries and model weights
retain their respective licensing terms.
