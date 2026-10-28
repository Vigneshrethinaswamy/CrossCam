"""Build camera-local tracklet JSON with mean HSV color histograms from evaluation CSVs."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from crosscam.appearance import hsv_histogram


def extract(input_dir: Path, video_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("*_tracks.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_tracks.csv files found in {input_dir}")
    result = []
    for csv_path in paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        videos = defaultdict(list)
        for row in rows:
            videos[row["video"]].append(row)
        for video_name, video_rows in videos.items():
            video_path = video_dir / video_name
            if not video_path.is_file():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            by_frame = defaultdict(list)
            for row in video_rows:
                by_frame[int(row["frame_index"])].append(row)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                raise ValueError(f"Could not open video file: {video_path}")
            features = defaultdict(list)
            bounds = {}
            try:
                for frame_index in sorted(by_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    height, width = frame.shape[:2]
                    for row in by_frame[frame_index]:
                        track_id = int(row["track_id"])
                        x1 = max(0, min(width, int(float(row["x1"]))))
                        y1 = max(0, min(height, int(float(row["y1"]))))
                        x2 = max(0, min(width, int(float(row["x2"]))))
                        y2 = max(0, min(height, int(float(row["y2"]))))
                        feature = hsv_histogram(frame[y1:y2, x1:x2])
                        if feature:
                            features[track_id].append(feature)
                        timestamp = float(row["timestamp_seconds"])
                        if track_id not in bounds:
                            bounds[track_id] = [timestamp, timestamp]
                        else:
                            bounds[track_id][0] = min(bounds[track_id][0], timestamp)
                            bounds[track_id][1] = max(bounds[track_id][1], timestamp)
            finally:
                cap.release()
            for track_id, histograms in sorted(features.items()):
                mean = np.asarray(histograms, dtype=np.float64).mean(axis=0)
                mean /= mean.sum()
                start, end = bounds[track_id]
                result.append({
                    "tracklet_id": f"{video_path.stem}:{track_id}",
                    "camera_id": video_path.stem,
                    "start_time": start,
                    "end_time": end,
                    "embedding": mean.tolist(),
                    "metadata": {"video": video_name, "observations": len(histograms),
                                 "descriptor": "mean_hsv_hue_saturation_histogram"},
                })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tracklets.json")
    args = parser.parse_args()
    try:
        tracklets = extract(args.input_dir, args.video_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tracklets, indent=2), encoding="utf-8")
    print(f"Tracklets: {len(tracklets)}; saved to {args.output}")
    print("Appearance descriptors are simple color histograms, not person ReID embeddings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
