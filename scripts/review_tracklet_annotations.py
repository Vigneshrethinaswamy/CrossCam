"""Review D4's four mixed-label sample1 tracklets frame by frame.

This tool reads the local annotation manifest and video, then writes decisions to
a separate review-results CSV. It never edits the source manifest or saves frames.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscam.annotation_review import (  # noqa: E402
    COMPLETED_REVIEW_STATUSES,
    REVIEW_FIELDS,
    build_review_rows,
    validate_review_rows,
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def write_csv_atomic(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def load_review_file(
    source_rows: list[dict[str, str]],
    source_fields: list[str],
    output_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    extra_fields = [field for field in REVIEW_FIELDS if field not in source_fields]
    fieldnames = source_fields + extra_fields

    if output_path.exists():
        rows, existing_fields = read_csv(output_path)
        validate_review_rows(source_rows, rows)
        missing_fields = set(REVIEW_FIELDS).difference(existing_fields)
        if missing_fields:
            raise ValueError(
                "Existing review CSV is missing columns: "
                + ", ".join(sorted(missing_fields))
            )
        return rows, existing_fields

    rows = build_review_rows(source_rows)
    write_csv_atomic(output_path, rows, fieldnames)
    return rows, fieldnames


def normalise_replacement_label(camera_id: str, value: str) -> str:
    value = value.strip()
    if value == "UNKNOWN":
        return value
    if re.fullmatch(r"GT-\d+", value):
        value = f"{camera_id}:{value}"
    if not re.fullmatch(re.escape(camera_id) + r":GT-\d+", value):
        raise ValueError(
            f"Enter GT-13 or {camera_id}:GT-13; use U for UNKNOWN"
        )
    return value


def draw_review_frame(frame, row: dict[str, str], position: int, total: int):
    height, width = frame.shape[:2]
    coordinates = (
        round(float(row["x1"])),
        round(float(row["y1"])),
        round(float(row["x2"])),
        round(float(row["y2"])),
    )
    x1 = max(0, min(width - 1, coordinates[0]))
    y1 = max(0, min(height - 1, coordinates[1]))
    x2 = max(0, min(width - 1, coordinates[2]))
    y2 = max(0, min(height - 1, coordinates[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("The manifest contains an invalid bounding box")

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        frame,
        "PERSON TO REVIEW",
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    previous = row.get("previous_ground_truth_identity", "") or "(blank)"
    current = row.get("ground_truth_identity", "") or "(blank)"
    banner_lines = [
        f"Review {position}/{total} | {row['video']} | track {row['track_id']}",
        (
            f"Time {float(row['timestamp_seconds']):.2f}s | "
            f"frame {row['frame_index']}"
        ),
        f"Previous label: {previous} | Current label: {current}",
        "K=keep current  R=reassign (type label in terminal)  U=UNKNOWN",
        "S=skip for now  Q=quit and save",
    ]
    banner_height = 126
    cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 0, 0), -1)
    for index, line in enumerate(banner_lines):
        cv2.putText(
            frame,
            line,
            (12, 22 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_review(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    output_path: Path,
    video_dir: Path,
) -> None:
    cap = None
    active_video = None
    window_name = "CrossCam D4 - Focused annotation review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for index, row in enumerate(rows):
            if row.get("review_status") in COMPLETED_REVIEW_STATUSES:
                continue

            video_path = video_dir / row["video"]
            if active_video != video_path:
                if cap is not None:
                    cap.release()
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video not found: {video_path}")
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError(f"Could not open video: {video_path}")
                active_video = video_path

            frame_index = int(row["frame_index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(
                    f"Could not decode {row['video']} frame {frame_index}; "
                    "leaving this row pending."
                )
                continue

            display = draw_review_frame(frame, row, index + 1, len(rows))
            scale = min(1.0, 1100 / display.shape[1], 760 / display.shape[0])
            if scale < 1.0:
                display = cv2.resize(
                    display,
                    (round(display.shape[1] * scale), round(display.shape[0] * scale)),
                )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key).lower() if key < 128 else ""

            if key_char == "q":
                break
            if key_char == "s":
                print(f"Skipped for now: {row['video']} track {row['track_id']}")
                continue
            if key_char == "k":
                previous = row.get("previous_ground_truth_identity", "").strip()
                if not previous or previous == "UNKNOWN":
                    print("No known previous label to keep; use R or U.")
                    continue
                row["ground_truth_identity"] = previous
                row["review_status"] = "confirmed"
                row["review_notes"] = "Confirmed previous label during focused review."
            elif key_char == "u":
                row["ground_truth_identity"] = "UNKNOWN"
                row["review_status"] = "explicit_unknown"
                row["review_notes"] = "Marked UNKNOWN during focused visual review."
            elif key_char == "r":
                value = input(
                    "Replacement identity (e.g. GT-13 or sample1:GT-13): "
                )
                try:
                    row["ground_truth_identity"] = normalise_replacement_label(
                        row["camera_id"], value
                    )
                except ValueError as error:
                    print(error)
                    continue
                if row["ground_truth_identity"] == "UNKNOWN":
                    row["review_status"] = "explicit_unknown"
                else:
                    row["review_status"] = "reassigned"
                row["review_notes"] = "Identity label edited during focused review."
            else:
                print("Unrecognised key. Use K, R, U, S, or Q.")
                continue

            write_csv_atomic(output_path, rows, fieldnames)
            print(
                f"Saved {row['review_status']}: "
                f"{row['video']} track {row['track_id']} "
                f"at {row['timestamp_seconds']}s"
            )
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        write_csv_atomic(output_path, rows, fieldnames)

    completed = sum(
        row.get("review_status") in COMPLETED_REVIEW_STATUSES for row in rows
    )
    print(f"Focused review saved: {output_path}")
    print(f"Reviewed: {completed}/{len(rows)}; pending/skipped: {len(rows) - completed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review the four mixed-label D4 sample1 tracklets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/crop_quality/annotation_manifest.csv",
    )
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/crop_quality/annotation_review_results.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Annotation manifest not found: {args.manifest}")

    source_rows, source_fields = read_csv(args.manifest)
    rows, fieldnames = load_review_file(source_rows, source_fields, args.output)
    print(f"Loaded {len(rows)} selected observations.")
    print("The source annotation manifest will not be modified.")
    run_review(rows, fieldnames, args.output, args.video_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
