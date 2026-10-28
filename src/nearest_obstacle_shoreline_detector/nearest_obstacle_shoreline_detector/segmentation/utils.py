"""Mask overlap helper used by water segmentation."""
import numpy as np


def calculate_iou(mask1, mask2):
    mask1 = (mask1 > 0)
    mask2 = (mask2 > 0)

    intersection = np.count_nonzero(mask1 & mask2)
    union = np.count_nonzero(mask1 | mask2)

    if union == 0:
        return 0.0  # Avoid division by zero

    return intersection / union
