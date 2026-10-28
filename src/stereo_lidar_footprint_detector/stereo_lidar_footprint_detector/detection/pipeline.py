"""Per-frame detection, sensor-depth fusion and square footprint generation."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .obstacles import create_obstacles


@dataclass
class FrameResult:
    boat_boxes: np.ndarray  # (N, 4) pixel xyxy; retained name for existing callers
    boat_mask: np.ndarray
    lidar_uv: np.ndarray
    lidar_xyz: np.ndarray
    runtime_ms: float
    obstacles: dict
    annotated_image: np.ndarray | None = None


class FreeSpacePipeline:
    """Rectified BGR + registered metric depth + camera-frame LiDAR -> per-frame detection, sensor-depth fusion and square footprint generation"""

    def __init__(
        self,
        K,
        baseline,
        image_size,
        t_body_to_cam=None,
        R_body_to_cam=None,
        yolo_weights=None,
        camera_to_water=None,
        device=None,
        classes=None,
        max_stereo_depth=20,
        detector=None,
    ):
        self.height, self.width = image_size
        self.K = np.asarray(K, dtype=float).copy()
        self.baseline = float(baseline)
        if not np.isfinite(self.baseline) or self.baseline <= 0:
            raise ValueError("Stereo baseline must be positive meters")
        if t_body_to_cam is not None or R_body_to_cam is not None:
            raise ValueError(
                "Use a full camera_to_water transform instead of legacy 2D stixel offsets"
            )
        self.camera_to_water = camera_to_water
        self.max_stereo_depth = max_stereo_depth
        if detector is None:
            from .yolo import YoloSeg

            weights = yolo_weights or "yolo26n-seg.pt"

            detector = YoloSeg(
                model_path=weights,
                device=device,
                classes=classes,
            )

        self.yolo = detector

    def process(
        self,
        timestamp,
        left_img,
        disparity_img,
        depth_img,
        lidar_uv,
        lidar_xyz,
        pose=None,
        camera_to_water=None,
    ):
        """Depth is optical Z in meters, not ray range; pixels use rectified K.

        timestamp, disparity_img and pose retain the sensor-adapter call contract.
        They are not needed for current-frame footprints. A per-frame transform
        overrides the configured transform (e.g. timestamped ROS TF).
        """
        start = time.perf_counter()
        left_img = np.ascontiguousarray(left_img, dtype=np.uint8)
        depth_img = np.asarray(depth_img, dtype=np.float32)
        if left_img.shape != (self.height, self.width, 3) or depth_img.shape != (
            self.height,
            self.width,
        ):
            raise ValueError("Image and depth must match the calibrated image_size")
        boxes, masks, labels, confidences, annotated_image = self.yolo.detect(left_img)
        obstacles = create_obstacles(
            boxes,
            depth_img,
            lidar_uv,
            lidar_xyz,
            self.K,
            instance_masks=masks,
            camera_to_water=self.camera_to_water
            if camera_to_water is None
            else camera_to_water,
            baseline=self.baseline,
            max_stereo_depth=self.max_stereo_depth,
        )
        for feature in obstacles["features"]:
            index = feature["properties"]["detection_index"]
            feature["properties"].update(
                class_id=int(labels[index]), confidence=float(confidences[index])
            )
        union_mask = np.zeros(depth_img.shape, dtype=np.uint8)
        for mask in masks or []:
            union_mask[mask > 0] = 255
        return FrameResult(
            np.asarray(boxes),
            union_mask,
            np.asarray(lidar_uv),
            np.asarray(lidar_xyz),
            (time.perf_counter() - start) * 1000,
            obstacles,
            annotated_image,
        )
