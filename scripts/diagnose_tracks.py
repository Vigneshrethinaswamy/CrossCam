"""Summarize camera-local track fragmentation from evaluate_videos.py CSV outputs."""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_track_rows(input_dir: Path) -> list[dict[str, Any]]:
    """Load track rows from *_tracks.csv files in an evaluation output directory."""
    files = sorted(input_dir.glob("*_tracks.csv"))
    if not files:
        raise FileNotFoundError(f"No *_tracks.csv files found in {input_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    rows.append({
                        "video": row["video"],
                        "camera_id": row["camera_id"],
                        "frame_index": int(row["frame_index"]),
                        "timestamp_seconds": float(row["timestamp_seconds"]),
                        "track_id": int(row["track_id"]),
                        "confidence": float(row["confidence"]),
                        "x1": float(row["x1"]),
                        "y1": float(row["y1"]),
                        "x2": float(row["x2"]),
                        "y2": float(row["y2"]),
                    })
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid track row in {path}: {row}") from exc
    return rows


def build_diagnostics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one row per local track and per-video aggregate diagnostics."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["video"], row["camera_id"], row["track_id"])].append(row)

    track_summaries: list[dict[str, Any]] = []
    for (video, camera_id, track_id), observations in sorted(grouped.items()):
        observations.sort(key=lambda item: (item["timestamp_seconds"], item["frame_index"]))
        timestamps = [item["timestamp_seconds"] for item in observations]
        frame_indices = [item["frame_index"] for item in observations]
        gaps = [round(b - a, 4) for a, b in zip(timestamps, timestamps[1:])]
        confidences = [item["confidence"] for item in observations]
        track_summaries.append({
            "video": video,
            "camera_id": camera_id,
            "track_id": track_id,
            "observations": len(observations),
            "start_seconds": round(timestamps[0], 4),
            "end_seconds": round(timestamps[-1], 4),
            "observed_span_seconds": round(timestamps[-1] - timestamps[0], 4),
            "first_frame": frame_indices[0],
            "last_frame": frame_indices[-1],
            "max_observation_gap_seconds": max(gaps, default=0.0),
            "median_observation_gap_seconds": round(statistics.median(gaps), 4) if gaps else 0.0,
            "mean_detection_score": round(statistics.mean(confidences), 4),
            "min_detection_score": round(min(confidences), 4),
            "singleton_track": len(observations) == 1,
        })

    videos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in track_summaries:
        videos[track["video"]].append(track)

    per_video = []
    for video, tracks in sorted(videos.items()):
        lengths = [track["observations"] for track in tracks]
        singletons = sum(track["singleton_track"] for track in tracks)
        per_video.append({
            "video": video,
            "track_count": len(tracks),
            "total_observations": sum(lengths),
            "median_observations_per_track": round(statistics.median(lengths), 2),
            "mean_observations_per_track": round(statistics.mean(lengths), 2),
            "singleton_tracks": singletons,
            "singleton_track_percent": round(100 * singletons / len(tracks), 2) if tracks else 0.0,
            "longest_track_observations": max(lengths, default=0),
            "tracks_with_gaps_over_1s": sum(
                track["max_observation_gap_seconds"] > 1.0 for track in tracks
            ),
        })

    summary = {
        "input_rows": len(rows),
        "distinct_video_count": len(per_video),
        "distinct_local_tracks": len(track_summaries),
        "per_video": per_video,
        "interpretation": (
            "Diagnostics describe observed track continuity only. Without ground-truth identity "
            "labels, they cannot establish identity switches, correct identity matches, or false merges."
        ),
    }
    return track_summaries, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "outputs" / "evaluation",
        help="Directory containing *_tracks.csv files from evaluate_videos.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "diagnostics",
        help="Directory where track-level CSV and JSON summary will be written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2
    try:
        rows = read_track_rows(args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    tracks, summary = build_diagnostics(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    track_csv = args.output_dir / "track_diagnostics.csv"
    summary_json = args.output_dir / "diagnostics_summary.json"
    write_csv(track_csv, tracks)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input observations: {summary['input_rows']}")
    print(f"Videos: {summary['distinct_video_count']}; local tracks: {summary['distinct_local_tracks']}")
    for item in summary["per_video"]:
        print(
            f"{item['video']}: tracks={item['track_count']}, "
            f"median_observations={item['median_observations_per_track']}, "
            f"singletons={item['singleton_tracks']} ({item['singleton_track_percent']:.1f}%), "
            f"longest={item['longest_track_observations']}"
        )
    print(f"Track details: {track_csv}")
    print(f"Summary: {summary_json}")
    print("Reminder: these are continuity diagnostics, not ground-truth identity accuracy metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
