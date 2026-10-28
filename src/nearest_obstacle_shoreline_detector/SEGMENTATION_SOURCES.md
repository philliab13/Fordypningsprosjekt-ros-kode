# Bundled segmentation sources

RWPS, FastSAMSeg, TemporalFiltering, and calculate_iou were copied from the local
`Stixel_based_free_space_estimation/free-space-estimation` checkout at commit
`4a0404c6296efb90434a8486689ef3c7fe981abb`.

- `free_space/rwps.py` → `nearest_obstacle_shoreline_detector/segmentation/rwps.py`
- `free_space/fastsam.py` → `nearest_obstacle_shoreline_detector/segmentation/fastsam.py`
- `free_space/temporal_filtering.py` → `nearest_obstacle_shoreline_detector/segmentation/temporal_filtering.py`
- Only `calculate_iou` from `free_space/utils.py` → `segmentation/utils.py`
- `configs/rwps.json` → `config/rwps.json`
- `weights/FastSAM-x.pt` → `weights/FastSAM-x.pt`

The algorithm modules retain the source behavior; imports of the shared helper
now use package-relative imports. Unrelated stixel utilities are not included.
The checkpoint is copied as a regular file, not a link to the source checkout.
This record is provenance only; that checkout is not used at runtime or build time.
Third-party libraries and model weights retain their respective licensing terms.
