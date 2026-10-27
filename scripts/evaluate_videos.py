"""Batch-evaluate local person detection and camera-local tracking on videos in data/."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from crosscam.detection import PersonDetector
from crosscam.tracker import IoUTracker
from crosscam.video import FrameSampler, VideoSource

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CSV_FIELDS = [
    "video",
    "camera_id",
    "frame_index",
    "timestamp_seconds",
    "detection_index",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--max-missed-frames", type=int, default=10)
    parser.add_argument("--max-videos", type=int, default=0, help="Optional limit; 0 means all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_fps <= 0:
        print("--sample-fps must be greater than zero", file=sys.stderr)
        return 2
    if not 0 <= args.iou_threshold <= 1:
        print("--iou-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.max_missed_frames < 0:
        print("--max-missed-frames must be non-negative", file=sys.stderr)
        return 2
    if args.max_videos < 0:
        print("--max-videos must be non-negative", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    videos = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        print(f"No supported videos found in {args.input_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = PersonDetector(args.device, args.confidence)
    summaries = []
    print(f"Found {len(videos)} video(s); inference device: {detector.device}")
    print(f"Sample rate: {args.sample_fps:g} FPS; output: {args.output_dir}")

    for video_path in videos:
        camera_id = video_path.stem
        csv_path = args.output_dir / f"{video_path.stem}_tracks.csv"
        tracker = IoUTracker(
            iou_threshold=args.iou_threshold,
            max_missed_frames=args.max_missed_frames,
        )
        sampler = FrameSampler(args.sample_fps)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = source_frames / source_fps if source_fps > 0 else 0.0

        sampled_frames = 0
        total_detections = 0
        peak_detections = 0
        observed_track_ids = set()
        elapsed_start = time.perf_counter()

        try:
            import torch

            if str(detector.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            torch = None

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for packet in VideoSource(video_path, camera_id).frames():
                if not sampler.should_sample(packet):
                    continue
                detections = detector.predict(packet.frame_bgr)
                tracked = tracker.update(detections)
                sampled_frames += 1
                total_detections += len(detections)
                peak_detections = max(peak_detections, len(detections))
                observed_track_ids.update(person.track_id for person in tracked)

                for index, person in enumerate(tracked):
                    x1, y1, x2, y2 = person.box
                    writer.writerow({
                        "video": video_path.name,
                        "camera_id": camera_id,
                        "frame_index": packet.frame_index,
                        "timestamp_seconds": f"{packet.timestamp_seconds:.4f}",
                        "detection_index": index,
                        "track_id": person.track_id,
                        "confidence": f"{person.detection_score:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                    })

        elapsed = time.perf_counter() - elapsed_start
        processing_fps = sampled_frames / elapsed if elapsed > 0 else 0.0
        peak_gpu_mb = None
        if torch is not None and str(detector.device).startswith("cuda") and torch.cuda.is_available():
            peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

        summary = {
            "video": video_path.name,
            "camera_id": camera_id,
            "width": width,
            "height": height,
            "source_fps": round(source_fps, 3),
            "source_frames": source_frames,
            "duration_seconds": round(duration, 3),
            "sample_target_fps": args.sample_fps,
            "sampled_frames": sampled_frames,
            "total_person_detections": total_detections,
            "peak_detections_in_sampled_frame": peak_detections,
            "distinct_local_track_ids_observed": len(observed_track_ids),
            "elapsed_seconds": round(elapsed, 3),
            "processing_sampled_fps": round(processing_fps, 3),
            "peak_torch_cuda_allocated_mb": peak_gpu_mb,
            "tracking_csv": str(csv_path),
            "note": "Track IDs are camera-local heuristic IDs, not verified unique identities.",
        }
        summaries.append(summary)
        print(
            f"{video_path.name}: sampled={sampled_frames}, detections={total_detections}, "
            f"peak_people_in_frame={peak_detections}, "
            f"observed_local_track_ids={len(observed_track_ids)}, "
            f"processing_fps={processing_fps:.2f}, elapsed={elapsed:.2f}s"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "device": str(detector.device),
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou_threshold,
        "max_missed_frames": args.max_missed_frames,
        "sample_target_fps": args.sample_fps,
        "videos_processed": len(summaries),
        "results": summaries,
    }, indent=2), encoding="utf-8")
    print(f"\nEvaluation summary: {summary_path}")
    print("Reminder: local track IDs are not cross-camera identities or ground-truth metrics.")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
