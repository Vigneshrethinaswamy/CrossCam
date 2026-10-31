"""Core typed data structures shared by CrossCam components."""
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


class Decision(str, Enum):
    ACCEPT = "accept"
    DEFER = "defer"
    REJECT = "reject"


@dataclass(frozen=True)
class Transition:
    source_camera: str
    target_camera: str
    min_seconds: float
    max_seconds: float
    prior: float | None = None

    def __post_init__(self) -> None:
        if not self.source_camera or not self.target_camera:
            raise ValueError("transition camera ids must be non-empty")
        if self.source_camera == self.target_camera:
            raise ValueError("a transition must connect two different cameras")
        if not math.isfinite(self.min_seconds) or not math.isfinite(self.max_seconds):
            raise ValueError("transition times must be finite")
        if self.min_seconds < 0 or self.max_seconds < self.min_seconds:
            raise ValueError("transition times require 0 <= min_seconds <= max_seconds")
        if self.prior is not None and (
            not math.isfinite(self.prior) or not 0 <= self.prior <= 1
        ):
            raise ValueError("transition prior must be None or between 0 and 1")


@dataclass(frozen=True)
class CameraRelationship:
    """Undirected knowledge about a pair of camera fields of view."""

    camera_a: str
    camera_b: str
    state: str

    def __post_init__(self) -> None:
        allowed = {"known_overlap", "known_non_overlap", "unknown"}
        if not self.camera_a or not self.camera_b or self.camera_a == self.camera_b:
            raise ValueError("camera relationship must connect two different cameras")
        if self.state not in allowed:
            raise ValueError(f"relationship state must be one of {sorted(allowed)}")


@dataclass
class Tracklet:
    tracklet_id: str
    camera_id: str
    start_time: float
    end_time: float
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_offset_known: bool = True

    def __post_init__(self) -> None:
        if not self.tracklet_id or not self.camera_id:
            raise ValueError("tracklet_id and camera_id must be non-empty")
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("timestamps must be finite")
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")


@dataclass(frozen=True)
class AssociationResult:
    source_tracklet_id: str
    candidate_tracklet_id: str
    score: float
    decision: Decision
    reasons: tuple[str, ...] = ()
    # Score is a heuristic ranking value, not a probability.
