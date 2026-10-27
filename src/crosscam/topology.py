"""Camera topology and candidate-generation helpers."""
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .models import Tracklet, Transition


@dataclass(frozen=True)
class Candidate:
    tracklet: Tracklet
    topology_prior: float
    elapsed_seconds: float
    via_topology: bool


class CameraTopology:
    def __init__(self, transitions: Iterable[Transition]):
        self._outgoing: dict[str, list[Transition]] = defaultdict(list)
        for transition in transitions:
            self._outgoing[transition.source_camera].append(transition)

    @classmethod
    def from_config(cls, config: dict) -> "CameraTopology":
        transitions = [
            Transition(
                source_camera=edge["from"],
                target_camera=edge["to"],
                min_seconds=float(edge["min_seconds"]),
                max_seconds=float(edge["max_seconds"]),
                prior=float(edge.get("prior", 1.0)),
            )
            for edge in config.get("transitions", [])
        ]
        return cls(transitions)

    def neighbors(self, camera_id: str) -> set[str]:
        return {edge.target_camera for edge in self._outgoing.get(camera_id, [])}

    def generate_candidates(
        self,
        source: Tracklet,
        candidates: Iterable[Tracklet],
        *,
        widen_if_empty: bool = True,
    ) -> list[Candidate]:
        """Prefer plausible neighboring cameras; optionally widen when none qualify.

        A topology edge prioritizes a route but must not permanently exclude a
        candidate on an unexpected route. If local candidate generation finds
        nothing plausible and widening is enabled, return time-compatible
        candidates from other cameras with a neutral prior.
        """
        all_candidates = list(candidates)
        outgoing = self._outgoing.get(source.camera_id, [])
        by_camera = {edge.target_camera: edge for edge in outgoing}
        results: list[Candidate] = []
        for candidate in all_candidates:
            if candidate.tracklet_id == source.tracklet_id:
                continue
            elapsed = candidate.start_time - source.end_time
            edge = by_camera.get(candidate.camera_id)
            if edge and edge.min_seconds <= elapsed <= edge.max_seconds:
                results.append(Candidate(candidate, edge.prior, elapsed, True))

        if results or not widen_if_empty:
            return sorted(results, key=lambda item: item.elapsed_seconds)

        # Conservative fallback: search other cameras without claiming a route prior.
        widened = []
        for candidate in all_candidates:
            if candidate.tracklet_id == source.tracklet_id or candidate.camera_id == source.camera_id:
                continue
            elapsed = candidate.start_time - source.end_time
            if 0 <= elapsed <= 180:
                widened.append(Candidate(candidate, 1.0, elapsed, False))
        return sorted(widened, key=lambda item: item.elapsed_seconds)
