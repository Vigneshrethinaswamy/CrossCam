"""Crop validation, temporal sampling, and tracklet-quality utilities.

Raw image crops are returned for in-memory use only. This module never writes
image data to disk.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]
ClippedBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class CropQualityConfig:
    """Provisional crop and tracklet thresholds; tune only on a development split."""

    min_width: int = 32
    min_height: int = 64
    min_detection_score: float = 0.45
    min_visible_fraction: float = 0.75
    min_laplacian_variance: float = 0.0
    min_observations: int = 3
    min_selected_crops: int = 2
    max_crops_per_tracklet: int = 5
    candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if self.min_width < 1 or self.min_height < 1:
            raise ValueError("minimum crop dimensions must be positive")
        if not math.isfinite(self.min_detection_score) or not (
            0.0 <= self.min_detection_score <= 1.0
        ):
            raise ValueError("min_detection_score must be between 0 and 1")
        if not math.isfinite(self.min_visible_fraction) or not (
            0.0 <= self.min_visible_fraction <= 1.0
        ):
            raise ValueError("min_visible_fraction must be between 0 and 1")
        if (
            not math.isfinite(self.min_laplacian_variance)
            or self.min_laplacian_variance < 0
        ):
            raise ValueError("min_laplacian_variance must be finite and non-negative")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        if self.max_crops_per_tracklet < 1:
            raise ValueError("max_crops_per_tracklet must be positive")
        if not 1 <= self.min_selected_crops <= self.max_crops_per_tracklet:
            raise ValueError(
                "min_selected_crops must be between 1 and max_crops_per_tracklet"
            )
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")


@dataclass(frozen=True)
class CropAssessment:
    frame_index: int
    timestamp_seconds: float
    original_box: Box
    clipped_box: ClippedBox | None
    width: int
    height: int
    detection_score: float
    visible_fraction: float
    laplacian_variance: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrackletPurity:
    observation_count: int
    matched_observations: int
    unmatched_observations: int
    dominant_identity: str | None
    identity_purity: float | None
    match_coverage: float


def _validate_frame(frame_bgr: NDArray[np.uint8]) -> tuple[int, int]:
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError("frame_bgr must be a numpy array")
    if frame_bgr.dtype != np.uint8:
        raise TypeError("frame_bgr must have dtype uint8")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3)")
    height, width = frame_bgr.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("frame_bgr must not be empty")
    return height, width


def _normalize_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box coordinates must be finite")
    return values  # type: ignore[return-value]


def clip_box_to_frame(
    box: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> tuple[ClippedBox | None, float]:
    """Clip an xyxy box and return (integer box, visible fraction).

    Coordinates are rounded outward (floor for left/top, ceil for right/bottom)
    so sub-pixel boxes do not accidentally lose visible pixels.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    x1, y1, x2, y2 = _normalize_box(box)
    original_width = x2 - x1
    original_height = y2 - y1
    if original_width <= 0 or original_height <= 0:
        return None, 0.0

    left = max(0, min(frame_width, math.floor(x1)))
    top = max(0, min(frame_height, math.floor(y1)))
    right = max(0, min(frame_width, math.ceil(x2)))
    bottom = max(0, min(frame_height, math.ceil(y2)))
    if right <= left or bottom <= top:
        return None, 0.0

    clipped_area = (right - left) * (bottom - top)
    original_area = original_width * original_height
    visible_fraction = min(1.0, clipped_area / original_area)
    return (left, top, right, bottom), float(visible_fraction)


def assess_person_crop(
    frame_bgr: NDArray[np.uint8],
    box: Sequence[float],
    detection_score: float,
    config: CropQualityConfig,
    *,
    frame_index: int,
    timestamp_seconds: float,
) -> tuple[NDArray[np.uint8] | None, CropAssessment]:
    """Extract one in-memory BGR crop and report measurable quality checks."""
    frame_height, frame_width = _validate_frame(frame_bgr)
    normalized_box = _normalize_box(box)
    score = float(detection_score)
    timestamp = float(timestamp_seconds)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("detection_score must be finite and between 0 and 1")
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp_seconds must be finite and non-negative")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    clipped, visible_fraction = clip_box_to_frame(
        normalized_box, frame_width, frame_height
    )
    if clipped is None:
        assessment = CropAssessment(
            frame_index=int(frame_index),
            timestamp_seconds=timestamp,
            original_box=normalized_box,
            clipped_box=None,
            width=0,
            height=0,
            detection_score=score,
            visible_fraction=0.0,
            laplacian_variance=0.0,
            accepted=False,
            rejection_reasons=("invalid_or_outside_frame",),
        )
        return None, assessment

    x1, y1, x2, y2 = clipped
    crop = np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = (
        float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if width >= 3 and height >= 3
        else 0.0
    )

    reasons: list[str] = []
    if width < config.min_width:
        reasons.append("too_narrow")
    if height < config.min_height:
        reasons.append("too_short")
    if score < config.min_detection_score:
        reasons.append("low_detection_score")
    if visible_fraction < config.min_visible_fraction:
        reasons.append("excessively_clipped")
    if (
        config.min_laplacian_variance > 0
        and sharpness < config.min_laplacian_variance
    ):
        reasons.append("low_sharpness")

    assessment = CropAssessment(
        frame_index=int(frame_index),
        timestamp_seconds=timestamp,
        original_box=normalized_box,
        clipped_box=clipped,
        width=width,
        height=height,
        detection_score=score,
        visible_fraction=visible_fraction,
        laplacian_variance=sharpness,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )
    return crop, assessment


def select_temporal_indices(
    timestamps_seconds: Sequence[float],
    max_items: int,
) -> list[int]:
    """Choose unique observations nearest evenly spaced temporal targets."""
    if max_items < 1:
        raise ValueError("max_items must be positive")
    timestamps = [float(value) for value in timestamps_seconds]
    if not timestamps:
        return []
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError("timestamps must be finite")

    ordered = sorted(range(len(timestamps)), key=lambda index: (timestamps[index], index))
    count = min(max_items, len(ordered))
    if count == len(ordered):
        return ordered

    start = timestamps[ordered[0]]
    end = timestamps[ordered[-1]]
    targets = (
        [(start + end) / 2.0]
        if count == 1
        else [start + (end - start) * i / (count - 1) for i in range(count)]
    )

    selected: list[int] = []
    available = set(ordered)
    for target in targets:
        choice = min(
            available,
            key=lambda index: (abs(timestamps[index] - target), timestamps[index], index),
        )
        selected.append(choice)
        available.remove(choice)

    return sorted(selected, key=lambda index: (timestamps[index], index))


def select_temporal_crops(
    assessments: Sequence[CropAssessment],
    max_crops: int,
) -> list[CropAssessment]:
    """Choose accepted crops spread across the tracklet's time span."""
    accepted = [item for item in assessments if item.accepted]
    if not accepted:
        return []
    selected_indices = select_temporal_indices(
        [item.timestamp_seconds for item in accepted], max_crops
    )
    return [accepted[index] for index in selected_indices]


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union for xyxy boxes."""
    ax1, ay1, ax2, ay2 = _normalize_box(first)
    bx1, by1, bx2, by2 = _normalize_box(second)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def match_frame_detections_to_ground_truth(
    predicted_boxes: Sequence[Sequence[float]],
    ground_truth_boxes: Sequence[Sequence[float]],
    ground_truth_ids: Sequence[str],
    min_iou: float = 0.5,
) -> list[str | None]:
    """Greedy one-to-one IoU matching for one camera/frame.

    The returned list follows predicted_boxes order. A None means no ground-
    truth box passed the IoU threshold. This is for diagnostics, not detector
    training or identity inference.
    """
    if len(ground_truth_boxes) != len(ground_truth_ids):
        raise ValueError("ground_truth_boxes and ground_truth_ids lengths differ")
    if not math.isfinite(min_iou) or not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be between 0 and 1")

    matches: list[str | None] = [None] * len(predicted_boxes)
    pairs = []
    for pred_index, predicted in enumerate(predicted_boxes):
        for gt_index, truth in enumerate(ground_truth_boxes):
            overlap = box_iou(predicted, truth)
            if overlap > 0.0 and overlap >= min_iou:
                pairs.append((overlap, pred_index, gt_index))

    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    for overlap, pred_index, gt_index in sorted(
        pairs, key=lambda item: (-item[0], item[1], item[2])
    ):
        if pred_index in used_predictions or gt_index in used_truth:
            continue
        matches[pred_index] = str(ground_truth_ids[gt_index])
        used_predictions.add(pred_index)
        used_truth.add(gt_index)
    return matches


def summarize_tracklet_purity(
    ground_truth_ids: Sequence[str | None],
) -> TrackletPurity:
    """Summarize labels matched to observations without hiding unmatched rows."""
    observation_count = len(ground_truth_ids)
    if observation_count == 0:
        raise ValueError("ground_truth_ids must contain at least one observation")

    labels = [str(value) for value in ground_truth_ids if value not in (None, "")]
    unmatched = observation_count - len(labels)
    if not labels:
        return TrackletPurity(
            observation_count=observation_count,
            matched_observations=0,
            unmatched_observations=unmatched,
            dominant_identity=None,
            identity_purity=None,
            match_coverage=0.0,
        )

    counts = Counter(labels)
    dominant_identity, dominant_count = sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    )[0]
    return TrackletPurity(
        observation_count=observation_count,
        matched_observations=len(labels),
        unmatched_observations=unmatched,
        dominant_identity=dominant_identity,
        identity_purity=dominant_count / len(labels),
        match_coverage=len(labels) / observation_count,
    )
