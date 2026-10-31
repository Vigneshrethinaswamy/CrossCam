"""Market-1501 image-retrieval metrics for a frozen encoder."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

_FILENAME = re.compile(r"^(?P<pid>-?\d+)_c(?P<camera>\d+)", re.IGNORECASE)


def market1501_identity_camera(filename: str) -> tuple[int, int]:
    """Read PID and one-based camera number from a standard Market-1501 filename."""
    match = _FILENAME.match(filename)
    if match is None:
        raise ValueError(f"Unrecognized Market-1501 image filename: {filename}")
    return int(match.group("pid")), int(match.group("camera"))


def evaluate_market1501(
    query_features: NDArray[np.floating],
    query_pids: Sequence[int],
    query_cameras: Sequence[int],
    gallery_features: NDArray[np.floating],
    gallery_pids: Sequence[int],
    gallery_cameras: Sequence[int],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    """Compute CMC, mAP and recall@k, filtering junk and same-camera positives."""
    q = np.asarray(query_features, dtype=np.float32)
    g = np.asarray(gallery_features, dtype=np.float32)
    qp, qc = np.asarray(query_pids), np.asarray(query_cameras)
    gp, gc = np.asarray(gallery_pids), np.asarray(gallery_cameras)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or q.shape[1] == 0:
        raise ValueError("query and gallery features must be compatible 2-D matrices")
    if q.shape[0] != len(qp) or q.shape[0] != len(qc):
        raise ValueError("query feature and label counts do not match")
    if g.shape[0] != len(gp) or g.shape[0] != len(gc):
        raise ValueError("gallery feature and label counts do not match")
    if not q.shape[0] or not g.shape[0]:
        raise ValueError("query and gallery sets must both be non-empty")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("features must be finite")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    q_norms, g_norms = np.linalg.norm(q, axis=1), np.linalg.norm(g, axis=1)
    if np.any(q_norms <= 0) or np.any(g_norms <= 0):
        raise ValueError("features must have non-zero norms")
    similarities = (q / q_norms[:, None]) @ (g / g_norms[:, None]).T
    aps: list[float] = []
    first_hits: list[int] = []
    recalls = {k: 0.0 for k in ks}

    for index, (pid, camera) in enumerate(zip(qp, qc, strict=True)):
        eligible = (gp > 0) & ~((gp == pid) & (gc == camera))
        relevant_count = int(((gp == pid) & eligible).sum())
        if not relevant_count:
            continue
        order = np.argsort(-similarities[index], kind="stable")
        ranked = order[eligible[order]]
        positions = np.flatnonzero(gp[ranked] == pid)
        first_hit = int(positions[0]) + 1
        first_hits.append(first_hit)
        query_matches = gp[ranked] == pid
        precisions = np.arange(1, len(positions) + 1, dtype=np.float64) / (positions + 1)
        aps.append(float(precisions.sum() / relevant_count))
        for k in ks:
            recalls[k] += float(query_matches[:k].sum()) / relevant_count

    if not aps:
        raise ValueError("no valid query has a cross-camera gallery match")
    count = len(aps)
    result: dict[str, float | int] = {
        "valid_queries": count,
        "mean_average_precision": float(np.mean(aps)),
    }
    for k in ks:
        result[f"rank_{k}"] = float(np.mean(np.asarray(first_hits) <= k))
        result[f"recall_at_{k}"] = recalls[k] / count
    return result
