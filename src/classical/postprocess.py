"""Morphological post-processing helpers for binary road masks."""

from __future__ import annotations

import cv2
import numpy as np


def to_uint8_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a boolean, 0/1, probability, or 0/255 mask to uint8 {0, 255}."""
    mask_array = np.asarray(mask)

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    if mask_array.dtype == bool:
        return mask_array.astype(np.uint8) * 255

    mask_array = mask_array.astype(np.float32)
    threshold_value = threshold * 255.0 if mask_array.max(initial=0) > 1.0 else threshold
    return (mask_array > threshold_value).astype(np.uint8) * 255


def apply_morphology(mask: np.ndarray) -> np.ndarray:
    """Apply opening and closing to remove speckles and fill small gaps."""
    binary = to_uint8_mask(mask)

    # Opening removes small isolated white noise. Closing reconnects nearby road pixels.
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)

    return to_uint8_mask(closed)


def remove_small_components(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Remove connected road components smaller than min_area pixels."""
    binary = to_uint8_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected road component."""
    binary = to_uint8_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1

    largest = np.zeros_like(binary)
    largest[labels == largest_label] = 255

    return largest


def postprocess_mask(
    mask: np.ndarray,
    min_area: int = 500,
    keep_largest: bool = False,
) -> np.ndarray:
    """Run the standard classical-baseline post-processing pipeline."""
    processed = apply_morphology(mask)
    processed = remove_small_components(processed, min_area=min_area)

    if keep_largest:
        processed = keep_largest_component(processed)

    return to_uint8_mask(processed)
