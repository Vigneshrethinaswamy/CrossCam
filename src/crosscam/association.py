"""Simple, intentionally uncalibrated tracklet association baseline."""
import math
from collections.abc import Sequence

from .models import AssociationResult, Decision, Tracklet
from .topology import Candidate


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def score_candidate(source: Tracklet, candidate: Candidate) -> tuple[float, tuple[str, ...]]:
    """Return a ranking heuristic in [0, 1], not a probability."""
    similarity = cosine_similarity(source.embedding, candidate.tracklet.embedding)
    appearance_score = (similarity + 1.0) / 2.0 if source.embedding and candidate.tracklet.embedding else 0.5
    elapsed_score = math.exp(-max(0.0, candidate.elapsed_seconds) / 90.0)
    topology_score = max(0.0, min(1.0, candidate.topology_prior))
    score = 0.70 * appearance_score + 0.20 * elapsed_score + 0.10 * topology_score
    reasons = (
        f"appearance_component={appearance_score:.3f}",
        f"time_component={elapsed_score:.3f}",
        "topology_edge" if candidate.via_topology else "widened_candidate_search",
    )
    return round(max(0.0, min(1.0, score)), 6), reasons


def decide(score: float, *, accept_threshold: float = 0.78, defer_threshold: float = 0.55) -> Decision:
    if not 0 <= defer_threshold <= accept_threshold <= 1:
        raise ValueError("Require 0 <= defer_threshold <= accept_threshold <= 1")
    if score >= accept_threshold:
        return Decision.ACCEPT
    if score >= defer_threshold:
        return Decision.DEFER
    return Decision.REJECT


def associate(source: Tracklet, candidate: Candidate, *, accept_threshold: float = 0.78,
              defer_threshold: float = 0.55) -> AssociationResult:
    score, reasons = score_candidate(source, candidate)
    return AssociationResult(
        source_tracklet_id=source.tracklet_id,
        candidate_tracklet_id=candidate.tracklet.tracklet_id,
        score=score,
        decision=decide(score, accept_threshold=accept_threshold, defer_threshold=defer_threshold),
        reasons=reasons,
    )
