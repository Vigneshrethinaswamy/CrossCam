"""Lightweight IoU-based local tracker; IDs are local to one camera."""
from dataclasses import dataclass

from .detection import Detection


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    missed_frames: int = 0


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
    """Greedy IoU matching with bounded track retention; no appearance ReID yet."""

    def __init__(self, iou_threshold: float = 0.30, max_missed_frames: int = 10) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
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
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                track_id = assigned[detection_index]
                self._tracks[track_id].box = detection.box
                self._tracks[track_id].missed_frames = 0
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, detection.box)
                assigned[detection_index] = track_id

        return [
            TrackedPerson(assigned[index], detection.box, detection.score)
            for index, detection in enumerate(detections)
        ]
