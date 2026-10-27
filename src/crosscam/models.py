"""Core typed data structures shared by CrossCam components."""
from dataclasses import dataclass, field
from enum import Enum
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
    prior: float = 1.0


@dataclass
class Tracklet:
    tracklet_id: str
    camera_id: str
    start_time: float
    end_time: float
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracklet_id or not self.camera_id:
            raise ValueError("tracklet_id and camera_id must be non-empty")
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        if self.start_time < 0:
            raise ValueError("timestamps must be non-negative")


@dataclass(frozen=True)
class AssociationResult:
    source_tracklet_id: str
    candidate_tracklet_id: str
    score: float
    decision: Decision
    reasons: tuple[str, ...] = ()
    # Score is a heuristic ranking value, not a probability.
