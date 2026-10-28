"""Dependency-light color descriptors for experimental cross-camera matching."""
import cv2
import numpy as np
from numpy.typing import NDArray


def hsv_histogram(person_crop_bgr: NDArray[np.uint8], bins: int = 8) -> list[float]:
    """Return an L1-normalized H/S histogram; this is a weak cue, not a ReID embedding."""
    if person_crop_bgr.ndim != 3 or person_crop_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if person_crop_bgr.size == 0:
        return []
    hsv = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    total = float(histogram.sum())
    if total <= 0:
        return []
    return (histogram.reshape(-1) / total).astype(float).tolist()
