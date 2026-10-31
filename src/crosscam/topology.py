"""Camera topology and candidate-generation helpers."""
from collections.abc import Iterable
from dataclasses import dataclass

from .models import CameraRelationship, Tracklet, Transition


@dataclass(frozen=True)
class Candidate:
    tracklet: Tracklet
    topology_prior: float | None
    elapsed_seconds: float | None
    via_topology: bool
    relationship_state: str = "unknown"


class CameraTopology:
    def __init__(
        self,
        transitions: Iterable[Transition],
        camera_relationships: Iterable[CameraRelationship] = (),
    ):
        self._transitions: dict[tuple[str, str], Transition] = {}
        for transition in transitions:
            key = (transition.source_camera, transition.target_camera)
            if key in self._transitions:
                raise ValueError(f"duplicate directed transition edge: {key}")
            self._transitions[key] = transition

        self._relationships: dict[tuple[str, str], str] = {}
        for relation in camera_relationships:
            key = tuple(sorted((relation.camera_a, relation.camera_b)))
            if key in self._relationships:
                raise ValueError(f"duplicate camera relationship: {key}")
            self._relationships[key] = relation.state

    @classmethod
    def from_config(cls, config: dict) -> "CameraTopology":
        transitions = [
            Transition(
                source_camera=edge["from"],
                target_camera=edge["to"],
                min_seconds=float(edge["min_seconds"]),
                max_seconds=float(edge["max_seconds"]),
                prior=None if edge.get("prior") is None else float(edge["prior"]),
            )
            for edge in config.get("transitions", [])
        ]
        relationships = [
            CameraRelationship(
                camera_a=entry["cameras"][0],
                camera_b=entry["cameras"][1],
                state=entry["state"],
            )
            for entry in config.get("camera_relationships", [])
        ]
        return cls(transitions, relationships)

    def relationship(self, camera_a: str, camera_b: str) -> str:
        return self._relationships.get(tuple(sorted((camera_a, camera_b))), "unknown")

    def neighbors(self, camera_id: str) -> set[str]:
        return {target for source, target in self._transitions if source == camera_id}

    @staticmethod
    def _intervals_overlap(source: Tracklet, candidate: Tracklet) -> bool:
        return (
            candidate.start_time <= source.end_time
            and source.start_time <= candidate.end_time
        )

    def generate_candidates(
        self,
        source: Tracklet,
        candidates: Iterable[Tracklet],
        *,
        widen_search: bool = False,
        widen_if_empty: bool | None = None,
        max_search_seconds: float = 180.0,
    ) -> list[Candidate]:
        """Return supported candidates and, when requested, a wider candidate union.

        widen_if_empty remains as a compatibility alias. Widening is now a union:
        existing topology candidates are retained and additional eligible candidates
        are appended rather than hidden by an early return.
        """
        if widen_if_empty is not None:
            widen_search = widen_if_empty
        if max_search_seconds < 0:
            raise ValueError("max_search_seconds must be non-negative")

        output: dict[str, Candidate] = {}
        for candidate in candidates:
            if (
                candidate.tracklet_id == source.tracklet_id
                or candidate.camera_id == source.camera_id
            ):
                continue

            timing_known = (
                source.timestamp_offset_known and candidate.timestamp_offset_known
            )
            elapsed = candidate.start_time - source.end_time if timing_known else None
            relation = self.relationship(source.camera_id, candidate.camera_id)
            intervals_overlap = timing_known and self._intervals_overlap(source, candidate)

            # Simultaneous appearances are impossible only for a known non-overlap
            # pair and only when both timestamps share a trusted time base.
            if relation == "known_non_overlap" and intervals_overlap:
                continue

            transition = self._transitions.get((source.camera_id, candidate.camera_id))
            if transition is not None and (
                elapsed is None
                or transition.min_seconds <= elapsed <= transition.max_seconds
            ):
                output[candidate.tracklet_id] = Candidate(
                    candidate, transition.prior, elapsed, True, "known_transition"
                )
                continue

            if relation == "known_overlap":
                if intervals_overlap or elapsed is None or abs(elapsed) <= max_search_seconds:
                    output[candidate.tracklet_id] = Candidate(
                        candidate, None, elapsed, True, "known_overlap"
                    )
                    continue

            if not widen_search:
                continue
            if elapsed is not None and abs(elapsed) > max_search_seconds:
                continue

            state = (
                "known_transition_outside_window"
                if transition is not None
                else relation
            )
            output[candidate.tracklet_id] = Candidate(
                candidate, None, elapsed, False, state
            )

        return sorted(
            output.values(),
            key=lambda item: (
                item.elapsed_seconds is None,
                abs(item.elapsed_seconds) if item.elapsed_seconds is not None else 0.0,
                item.tracklet.tracklet_id,
            ),
        )
