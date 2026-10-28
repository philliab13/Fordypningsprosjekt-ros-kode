"""YOLO detections with separate masks to avoid mixing overlapping objects."""

import cv2
import numpy as np
from ultralytics import YOLO


class YoloSeg:
    def __init__(self, model_path="yolo26n-seg.pt", device=None, classes=None):
        self.model = YOLO(str(model_path))
        self.device = device
        self.classes = [8]  # None = all classes; [8] = COCO boats only

    def detect(self, img):
        result = self.model.predict(
            img,
            device=self.device,
            classes=self.classes,
            show=False,
            retina_masks=True,
            iou=0.5,
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # Ultralytics automatically draws masks, boxes and labels.
        annotated_image = result.plot()

        masks = None
        if result.masks is not None:
            masks = []
            for polygon in result.masks.xy:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                if len(polygon) >= 3:
                    cv2.fillPoly(
                        mask,
                        [np.round(polygon).astype(np.int32)],
                        255,
                    )
                masks.append(mask)

        return boxes, masks, labels, confidences, annotated_image

    def get_boat_mask(self, img, device=None):
        """Legacy union-mask helper; classes are selected at construction."""
        old_device = self.device
        try:
            if device is not None:
                self.device = device

            boxes, masks, _, _, _ = self.detect(img)

        finally:
            self.device = old_device

        union = np.zeros(img.shape[:2], dtype=np.uint8)
        for mask in masks or []:
            union[mask > 0] = 255

        return union, boxes
