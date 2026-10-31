"""Lightweight IoU-based local tracker; IDs are local to one camera."""
from dataclasses import dataclass
import math

from .detection import Detection


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    last_seen_seconds: float


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    box: tuple[float, float, float, float]
    detection_score: float


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """Greedy IoU matching with retention measured in video seconds."""

    def __init__(self, iou_threshold: float = 0.30, max_missed_seconds: float = 2.0) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if not math.isfinite(max_missed_seconds) or max_missed_seconds < 0:
            raise ValueError("max_missed_seconds must be finite and non-negative")
        self.iou_threshold = iou_threshold
        self.max_missed_seconds = float(max_missed_seconds)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_update_seconds: float | None = None

    def update(
        self,
        detections: list[Detection],
        *,
        timestamp_seconds: float,
    ) -> list[TrackedPerson]:
        if not math.isfinite(timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        if (
            self._last_update_seconds is not None
            and timestamp_seconds < self._last_update_seconds
        ):
            raise ValueError("tracker timestamps must be monotonic non-decreasing")
        self._last_update_seconds = timestamp_seconds

        pairs = sorted(
            (
                (intersection_over_union(track.box, detection.box), track_id, detection_index)
                for track_id, track in self._tracks.items()
                for detection_index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assigned: dict[int, int] = {}

        for overlap, track_id, detection_index in pairs:
            if overlap < self.iou_threshold:
                break
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assigned[detection_index] = track_id

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                if timestamp_seconds - track.last_seen_seconds > self.max_missed_seconds:
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                track_id = assigned[detection_index]
                self._tracks[track_id].box = detection.box
                self._tracks[track_id].last_seen_seconds = timestamp_seconds
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(
                    track_id, detection.box, timestamp_seconds
                )
                assigned[detection_index] = track_id

        return [
            TrackedPerson(assigned[index], detection.box, detection.score)
            for index, detection in enumerate(detections)
        ]
