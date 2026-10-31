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


def score_candidate(
    source: Tracklet,
    candidate: Candidate,
    *,
    time_decay_seconds: float = 90.0,
) -> tuple[float, tuple[str, ...]]:
    """Return a ranking heuristic in [0, 1], not a probability."""
    if not math.isfinite(time_decay_seconds) or time_decay_seconds <= 0:
        raise ValueError("time_decay_seconds must be greater than zero")

    similarity = cosine_similarity(source.embedding, candidate.tracklet.embedding)
    appearance_score = (
        max(0.0, similarity)
        if source.embedding and candidate.tracklet.embedding
        else 0.0
    )
    elapsed_score = (
        math.exp(-max(0.0, candidate.elapsed_seconds) / time_decay_seconds)
        if candidate.elapsed_seconds is not None
        else 0.5
    )
    # Missing topology is neutral, not a fabricated maximal positive prior.
    topology_score = (
        0.5
        if candidate.topology_prior is None
        else max(0.0, min(1.0, candidate.topology_prior))
    )
    score = 0.70 * appearance_score + 0.20 * elapsed_score + 0.10 * topology_score
    reasons = (
        f"appearance_component={appearance_score:.3f}",
        (
            f"time_component={elapsed_score:.3f}"
            if candidate.elapsed_seconds is not None
            else "temporal_evidence=unknown"
        ),
        (
            f"topology_prior={candidate.topology_prior:.3f}"
            if candidate.topology_prior is not None
            else "topology_evidence=unknown"
        ),
        f"camera_relationship={candidate.relationship_state}",
    )
    return round(max(0.0, min(1.0, score)), 6), reasons


def decide(
    score: float,
    *,
    accept_threshold: float = 0.78,
    defer_threshold: float = 0.55,
) -> Decision:
    if not 0 <= defer_threshold <= accept_threshold <= 1:
        raise ValueError("Require 0 <= defer_threshold <= accept_threshold <= 1")
    if score >= accept_threshold:
        return Decision.ACCEPT
    if score >= defer_threshold:
        return Decision.DEFER
    return Decision.REJECT


def associate(
    source: Tracklet,
    candidate: Candidate,
    *,
    accept_threshold: float = 0.78,
    defer_threshold: float = 0.55,
    min_margin: float = 0.0,
    runner_up_score: float | None = None,
    time_decay_seconds: float = 90.0,
) -> AssociationResult:
    if not math.isfinite(min_margin) or not 0 <= min_margin <= 1:
        raise ValueError("min_margin must be between 0 and 1")
    score, reasons = score_candidate(
        source, candidate, time_decay_seconds=time_decay_seconds
    )
    decision = decide(
        score,
        accept_threshold=accept_threshold,
        defer_threshold=defer_threshold,
    )
    if (
        decision is Decision.ACCEPT
        and runner_up_score is not None
        and score - runner_up_score < min_margin
    ):
        decision = Decision.DEFER
        reasons = (*reasons, f"decision_margin_below_minimum={min_margin:.3f}")
    return AssociationResult(
        source_tracklet_id=source.tracklet_id,
        candidate_tracklet_id=candidate.tracklet.tracklet_id,
        score=score,
        decision=decision,
        reasons=reasons,
    )
