"""Rank cross-camera tracklet candidates using HSV appearance, time and topology heuristics."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscam.association import associate
from crosscam.config import load_config
from crosscam.models import Tracklet
from crosscam.topology import CameraTopology


def parse_offsets(values: list[str]) -> dict[str, float]:
    offsets = {}
    for value in values:
        try:
            camera, seconds = value.split("=", 1)
            offsets[camera] = float(seconds)
        except ValueError as exc:
            raise ValueError(f"Invalid camera offset {value!r}; expected CAMERA=SECONDS") from exc
    return offsets


def rank(tracklet_rows: list[dict], config: dict, offsets: dict[str, float]) -> list[dict]:
    tracklets = []
    for row in tracklet_rows:
        offset = offsets.get(row["camera_id"], 0.0)
        tracklets.append(Tracklet(
            tracklet_id=row["tracklet_id"],
            camera_id=row["camera_id"],
            start_time=float(row["start_time"]) + offset,
            end_time=float(row["end_time"]) + offset,
            embedding=row.get("embedding", []),
            metadata=row.get("metadata", {}),
        ))
    topology = CameraTopology.from_config(config)
    settings = config.get("association", {})
    output = []
    for source in tracklets:
        candidates = topology.generate_candidates(
            source,
            tracklets,
            widen_if_empty=settings.get("widen_search_if_weak", True),
        )
        for candidate in candidates:
            result = associate(
                source, candidate,
                accept_threshold=float(settings.get("accept_threshold", 0.78)),
                defer_threshold=float(settings.get("defer_threshold", 0.55)),
            )
            output.append({
                "source_tracklet_id": result.source_tracklet_id,
                "source_camera_id": source.camera_id,
                "candidate_tracklet_id": result.candidate_tracklet_id,
                "candidate_camera_id": candidate.tracklet.camera_id,
                "elapsed_seconds": candidate.elapsed_seconds,
                "via_topology": candidate.via_topology,
                "score": result.score,
                "decision": result.decision.value,
                "reasons": ";".join(result.reasons),
            })
    return sorted(output, key=lambda row: row["score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", type=Path, default=ROOT / "outputs/tracklets.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--camera-offset", action="append", default=[], metavar="CAMERA=SECONDS",
                        help="Offset a video's local timestamps to a shared timeline; repeat as needed")
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
    fields = ["source_tracklet_id", "source_camera_id", "candidate_tracklet_id",
              "candidate_camera_id", "elapsed_seconds", "via_topology", "score",
              "decision", "reasons"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(associations)
    print(f"Candidate links: {len(associations)}; saved to {args.output}")
    print("Scores rank heuristic evidence; they are not probabilities or verified identities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
