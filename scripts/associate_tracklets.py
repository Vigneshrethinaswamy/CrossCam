"""Rank cross-camera tracklet candidates using HSV appearance and cautious timing/topology."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscam.association import associate, score_candidate
from crosscam.config import load_config
from crosscam.models import Tracklet
from crosscam.topology import CameraTopology


def parse_offsets(values: list[str]) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for value in values:
        try:
            camera, raw_seconds = value.split("=", 1)
            seconds = float(raw_seconds)
        except ValueError as exc:
            raise ValueError(
                f"Invalid camera offset {value!r}; expected CAMERA=SECONDS"
            ) from exc
        if not camera or not math.isfinite(seconds):
            raise ValueError(
                f"Invalid camera offset {value!r}; require a camera and finite seconds"
            )
        if camera in offsets:
            raise ValueError(f"Duplicate camera offset supplied for {camera}")
        offsets[camera] = seconds
    return offsets


def rank(tracklet_rows: list[dict], config: dict, offsets: dict[str, float]) -> list[dict]:
    known_cameras = {camera["id"] for camera in config.get("cameras", [])}
    unknown = {row["camera_id"] for row in tracklet_rows} - known_cameras
    unknown_offsets = set(offsets) - known_cameras
    if unknown:
        raise ValueError(f"Tracklets refer to cameras missing from config: {sorted(unknown)}")
    if unknown_offsets:
        raise ValueError(f"Offsets refer to cameras missing from config: {sorted(unknown_offsets)}")

    tracklets = []
    for row in tracklet_rows:
        camera_id = row["camera_id"]
        offset_known = camera_id in offsets
        offset = offsets.get(camera_id, 0.0)
        tracklets.append(Tracklet(
            tracklet_id=row["tracklet_id"],
            camera_id=camera_id,
            start_time=float(row["start_time"]) + offset,
            end_time=float(row["end_time"]) + offset,
            embedding=row.get("embedding", []),
            metadata={
                **row.get("metadata", {}),
                "timestamp_offset_known": offset_known,
            },
            timestamp_offset_known=offset_known,
        ))

    topology = CameraTopology.from_config(config)
    settings = config.get("association", {})
    widen_enabled = bool(settings.get("widen_search_if_weak", True))
    widen_threshold = float(settings.get("widen_if_best_score_below", 0.78))
    horizon = float(settings.get("widen_search_horizon_seconds", 180.0))
    time_decay = float(settings.get("time_decay_seconds", 90.0))
    accept = float(settings.get("accept_threshold", 0.78))
    defer = float(settings.get("defer_threshold", 0.55))
    min_margin = float(settings.get("min_margin", 0.0))

    output = []
    for source in tracklets:
        candidates = topology.generate_candidates(
            source,
            tracklets,
            widen_search=False,
            max_search_seconds=horizon,
        )
        scored = [
            (candidate, score_candidate(source, candidate, time_decay_seconds=time_decay)[0])
            for candidate in candidates
        ]
        best_score = max((score for _, score in scored), default=0.0)

        widened = False
        if widen_enabled and best_score < widen_threshold:
            widened = True
            candidates = topology.generate_candidates(
                source,
                tracklets,
                widen_search=True,
                max_search_seconds=horizon,
            )
            scored = [
                (candidate, score_candidate(
                    source, candidate, time_decay_seconds=time_decay
                )[0])
                for candidate in candidates
            ]

        ordered = sorted(scored, key=lambda item: item[1], reverse=True)
        for candidate, score in ordered:
            runner_up = max(
                (other_score for other_candidate, other_score in ordered
                 if other_candidate.tracklet.tracklet_id != candidate.tracklet.tracklet_id),
                default=None,
            )
            result = associate(
                source,
                candidate,
                accept_threshold=accept,
                defer_threshold=defer,
                min_margin=min_margin,
                runner_up_score=runner_up,
                time_decay_seconds=time_decay,
            )
            output.append({
                "source_tracklet_id": result.source_tracklet_id,
                "source_camera_id": source.camera_id,
                "candidate_tracklet_id": result.candidate_tracklet_id,
                "candidate_camera_id": candidate.tracklet.camera_id,
                "elapsed_seconds": candidate.elapsed_seconds,
                "timestamp_offsets_known": (
                    source.timestamp_offset_known
                    and candidate.tracklet.timestamp_offset_known
                ),
                "camera_relationship": candidate.relationship_state,
                "via_topology": candidate.via_topology,
                "widened_search_used": widened,
                "score": result.score,
                "decision": result.decision.value,
                "reasons": ";".join(result.reasons),
            })
    return sorted(output, key=lambda row: row["score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", type=Path, default=ROOT / "outputs/tracklets.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument(
        "--camera-offset",
        action="append",
        default=[],
        metavar="CAMERA=SECONDS",
        help="Trusted offset to a shared timeline; repeat for each camera. Missing offsets remain unknown.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/associations.csv")
    args = parser.parse_args()
    if not args.tracklets.is_file():
        print(f"Tracklet file not found: {args.tracklets}", file=sys.stderr)
        return 2
    try:
        offsets = parse_offsets(args.camera_offset)
        config = load_config(args.config)
        rows = json.loads(args.tracklets.read_text(encoding="utf-8"))
        associations = rank(rows, config, offsets)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Could not associate tracklets: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_tracklet_id", "source_camera_id", "candidate_tracklet_id",
        "candidate_camera_id", "elapsed_seconds", "timestamp_offsets_known",
        "camera_relationship", "via_topology", "widened_search_used", "score",
        "decision", "reasons",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(associations)
    print(f"Candidate links: {len(associations)}; saved to {args.output}")
    print("Scores rank heuristic evidence; they are not probabilities or verified identities.")
    print("Camera pairs without explicit offsets carry unknown temporal evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
