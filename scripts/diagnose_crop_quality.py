"""Diagnose representative person-crop quality without saving raw image crops."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscam.crop_quality import (
    CropAssessment,
    CropQualityConfig,
    assess_person_crop,
    select_temporal_crops,
    select_temporal_indices,
)
from crosscam.video import VideoSource

REQUIRED_COLUMNS = {
    "video",
    "camera_id",
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
}

TrackKey = tuple[str, str, int]


def read_observations(input_dir: Path) -> dict[str, dict[TrackKey, list[dict[str, Any]]]]:
    """Read evaluation CSVs and group observations by video/camera/local track."""
    files = sorted(input_dir.glob("*_tracks.csv"))
    if not files:
        raise FileNotFoundError(f"No *_tracks.csv files found in {input_dir}")

    grouped: dict[str, dict[TrackKey, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[tuple[str, str, int, int]] = set()

    for csv_path in files:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

            for line_number, row in enumerate(reader, start=2):
                try:
                    video = str(row["video"]).strip()
                    camera_id = str(row["camera_id"]).strip()
                    frame_index = int(row["frame_index"])
                    timestamp = float(row["timestamp_seconds"])
                    track_id = int(row["track_id"])
                    score = float(row["confidence"])
                    box = tuple(
                        float(row[name]) for name in ("x1", "y1", "x2", "y2")
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid observation at {csv_path}:{line_number}: {row}"
                    ) from exc

                if not video or Path(video).name != video or not camera_id:
                    raise ValueError(
                        f"Invalid video/camera identifier at {csv_path}:{line_number}"
                    )
                if frame_index < 0 or track_id < 0:
                    raise ValueError(
                        f"Negative frame or track ID at {csv_path}:{line_number}"
                    )
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError(
                        f"Invalid timestamp at {csv_path}:{line_number}"
                    )
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"Invalid detector score at {csv_path}:{line_number}"
                    )
                if not all(math.isfinite(value) for value in box):
                    raise ValueError(
                        f"Non-finite box coordinate at {csv_path}:{line_number}"
                    )

                duplicate_key = (video, camera_id, track_id, frame_index)
                if duplicate_key in seen:
                    raise ValueError(
                        f"Duplicate observation for {camera_id}:{track_id} "
                        f"at frame {frame_index}"
                    )
                seen.add(duplicate_key)

                key = (video, camera_id, track_id)
                grouped[video][key].append({
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "confidence": score,
                    "box": box,
                })

    if not grouped:
        raise ValueError(f"No tracking observations found in {input_dir}")

    for video_tracks in grouped.values():
        for rows in video_tracks.values():
            rows.sort(key=lambda item: (item["timestamp_seconds"], item["frame_index"]))

    return dict(grouped)


def prepare_temporal_candidates(
    observations: dict[str, dict[TrackKey, list[dict[str, Any]]]],
    config: CropQualityConfig,
) -> tuple[
    dict[str, dict[int, list[tuple[TrackKey, dict[str, Any]]]]],
    dict[TrackKey, dict[str, Any]],
]:
    """Select a bounded number of candidate observations across each tracklet."""
    frames: dict[str, dict[int, list[tuple[TrackKey, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    track_metadata: dict[TrackKey, dict[str, Any]] = {}

    candidate_limit = config.max_crops_per_tracklet * config.candidate_multiplier
    for video, tracks in observations.items():
        for key, rows in tracks.items():
            indices = select_temporal_indices(
                [row["timestamp_seconds"] for row in rows],
                max_items=candidate_limit,
            )
            track_metadata[key] = {
                "observations": len(rows),
                "candidate_count": len(indices),
                "camera_id": key[1],
                "track_id": key[2],
            }
            for index in indices:
                row = rows[index]
                frames[video][row["frame_index"]].append((key, row))

    return (
        {video: dict(frame_map) for video, frame_map in frames.items()},
        track_metadata,
    )


def _assessment_summary(assessments: list[CropAssessment]) -> dict[str, Any]:
    rejected = Counter(
        reason
        for item in assessments
        for reason in item.rejection_reasons
    )
    accepted = [item for item in assessments if item.accepted]
    return {
        "decoded_candidate_count": len(assessments),
        "accepted_candidate_count": len(accepted),
        "rejected_candidate_count": len(assessments) - len(accepted),
        "rejection_reason_counts": dict(sorted(rejected.items())),
    }


def build_tracklet_quality_report(
    video_dir: Path,
    observations: dict[str, dict[TrackKey, list[dict[str, Any]]]],
    config: CropQualityConfig,
) -> dict[str, Any]:
    """Decode candidate frames and emit crop/tracklet metadata only.

    Image crops are evaluated in memory and discarded after measurement. No
    crop images, embeddings, or video frames are written to disk.
    """
    frame_candidates, track_metadata = prepare_temporal_candidates(observations, config)
    assessments_by_track: dict[TrackKey, list[CropAssessment]] = defaultdict(list)
    decoded_frames_by_video: dict[str, int] = {}
    unreadable_candidate_frames_by_video: dict[str, int] = {}

    for video, frame_map in frame_candidates.items():
        video_path = video_dir / video
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        camera_ids = {key[1] for keys in frame_map.values() for key, _ in keys}
        if len(camera_ids) != 1:
            raise ValueError(f"Expected one camera ID per video {video}; found {camera_ids}")
        camera_id = next(iter(camera_ids))
        requested_frames = set(frame_map)
        decoded_frames: set[int] = set()

        for packet in VideoSource(video_path, camera_id).frames():
            candidates = frame_map.get(packet.frame_index)
            if not candidates:
                continue

            decoded_frames.add(packet.frame_index)
            for key, row in candidates:
                _, assessment = assess_person_crop(
                    packet.frame_bgr,
                    row["box"],
                    row["confidence"],
                    config,
                    frame_index=packet.frame_index,
                    timestamp_seconds=row["timestamp_seconds"],
                )
                assessments_by_track[key].append(assessment)

        decoded_frames_by_video[video] = len(decoded_frames)
        unreadable_candidate_frames_by_video[video] = len(
            requested_frames - decoded_frames
        )

    track_reports: list[dict[str, Any]] = []
    for key, metadata in sorted(track_metadata.items()):
        video, camera_id, track_id = key
        assessments = assessments_by_track.get(key, [])
        chosen = select_temporal_crops(
            assessments, max_crops=config.max_crops_per_tracklet
        )
        reasons: list[str] = []
        if metadata["observations"] < config.min_observations:
            reasons.append("too_few_track_observations")
        if len(chosen) < config.min_selected_crops:
            reasons.append("insufficient_usable_crops")
        if len(assessments) < metadata["candidate_count"]:
            reasons.append("candidate_frames_not_decoded")

        accepted = [item for item in assessments if item.accepted]
        track_reports.append({
            "tracklet_id": f"{camera_id}:{track_id}",
            "video": video,
            "camera_id": camera_id,
            "track_id": track_id,
            "observation_count": metadata["observations"],
            "temporal_candidate_count": metadata["candidate_count"],
            **_assessment_summary(assessments),
            "selected_crop_count": len(chosen),
            "selected_frame_indices": [item.frame_index for item in chosen],
            "selected_timestamps_seconds": [
                round(item.timestamp_seconds, 4) for item in chosen
            ],
            "median_selected_detection_score": (
                statistics.median(item.detection_score for item in chosen)
                if chosen else None
            ),
            "median_selected_visible_fraction": (
                statistics.median(item.visible_fraction for item in chosen)
                if chosen else None
            ),
            "median_selected_laplacian_variance": (
                statistics.median(item.laplacian_variance for item in chosen)
                if chosen else None
            ),
            "candidate_rejection_reasons": dict(sorted(Counter(
                reason for item in assessments for reason in item.rejection_reasons
            ).items())),
            "crop_quality_eligible_for_next_stage": not reasons,
            "eligibility_reasons": reasons,
        })

    eligible_count = sum(
        item["crop_quality_eligible_for_next_stage"] for item in track_reports
    )
    total_observations = sum(item["observation_count"] for item in track_reports)
    summary = {
        "video_count": len(observations),
        "tracklet_count": len(track_reports),
        "total_track_observations": total_observations,
        "eligible_tracklets": eligible_count,
        "ineligible_tracklets": len(track_reports) - eligible_count,
        "eligible_tracklet_percent": (
            100.0 * eligible_count / len(track_reports) if track_reports else 0.0
        ),
        "decoded_candidate_frames_by_video": decoded_frames_by_video,
        "unreadable_candidate_frames_by_video": unreadable_candidate_frames_by_video,
        "crop_quality_config": asdict(config),
        "limitations": [
            (
                "Eligibility is a crop/observation readiness check, not proof "
                "of tracklet identity purity."
            ),
            (
                "No ground-truth identities are supplied by the detector CSV; "
                "purity is not measured here."
            ),
            "Laplacian variance is a blur proxy and is not a calibrated image-quality score.",
            "Thresholds are provisional and require development-set review.",
            "Raw crops are evaluated in memory and are not saved; embeddings are not generated.",
            (
                "VideoSource timestamps are derived from reported FPS; verify "
                "PTS for variable-frame-rate footage."
            ),
        ],
    }
    return {"summary": summary, "tracklets": track_reports}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "outputs" / "evaluation",
        help="Directory containing *_tracks.csv evaluation files",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=ROOT / "data",
        help="Directory containing the videos referenced by the CSV files",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "crop_quality" / "crop_quality_summary.json",
    )
    parser.add_argument("--min-width", type=int, default=32)
    parser.add_argument("--min-height", type=int, default=64)
    parser.add_argument("--min-detection-score", type=float, default=0.45)
    parser.add_argument("--min-visible-fraction", type=float, default=0.75)
    parser.add_argument(
        "--min-laplacian-variance", type=float, default=0.0,
        help="0 disables blur rejection but still reports sharpness",
    )
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-selected-crops", type=int, default=2)
    parser.add_argument("--max-crops-per-tracklet", type=int, default=5)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.video_dir.is_dir():
        print(f"Video directory not found: {args.video_dir}", file=sys.stderr)
        return 2

    try:
        config = CropQualityConfig(
            min_width=args.min_width,
            min_height=args.min_height,
            min_detection_score=args.min_detection_score,
            min_visible_fraction=args.min_visible_fraction,
            min_laplacian_variance=args.min_laplacian_variance,
            min_observations=args.min_observations,
            min_selected_crops=args.min_selected_crops,
            max_crops_per_tracklet=args.max_crops_per_tracklet,
            candidate_multiplier=args.candidate_multiplier,
        )
        observations = read_observations(args.input_dir)
        report = build_tracklet_quality_report(
            args.video_dir,
            observations,
            config,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Videos: {summary['video_count']}")
    print(f"Tracklets: {summary['tracklet_count']}")
    print(f"Crop-quality eligible for next stage: {summary['eligible_tracklets']}")
    print(f"Ineligible: {summary['ineligible_tracklets']}")
    print(f"Quality report: {args.output}")
    print("No raw crops or embeddings were saved.")
    print("Eligibility is not a ground-truth identity-purity result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
