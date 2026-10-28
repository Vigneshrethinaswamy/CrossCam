"""Build a concise Markdown report from CrossCam evaluation artifacts."""
import json
from pathlib import Path
from typing import Any


def build_report(summary: dict[str, Any], diagnostics: dict[str, Any] | None = None) -> str:
    """Render measured runtime and continuity diagnostics without implying ID accuracy."""
    lines = [
        "# CrossCam run report", "",
        f"- Device: {summary.get('device', 'unknown')}",
        f"- Videos processed: {summary.get('videos_processed', 0)}",
        f"- Sample target: {summary.get('sample_target_fps', 'unknown')} FPS",
        f"- Detector threshold: {summary.get('confidence_threshold', 'unknown')}",
        f"- IoU threshold: {summary.get('iou_threshold', 'unknown')}", "",
        "## Per-video observations", "",
        "| Video | Sampled frames | Person detections | Local tracks | Processing FPS | Peak CUDA MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in summary.get("results", []):
        memory = result.get("peak_torch_cuda_allocated_mb")
        lines.append(
            "| {video} | {sampled} | {detections} | {tracks} | {fps} | {memory} |".format(
                video=result.get("video", ""),
                sampled=result.get("sampled_frames", 0),
                detections=result.get("total_person_detections", 0),
                tracks=result.get("distinct_local_track_ids_observed", 0),
                fps=result.get("processing_sampled_fps", 0),
                memory=memory if memory is not None else "n/a",
            )
        )
    if diagnostics is not None:
        lines.extend(["", "## Track continuity", ""])
        lines.append(f"- Distinct local tracks: {diagnostics.get('distinct_local_tracks', 0)}")
        lines.append(f"- Detection observations: {diagnostics.get('input_rows', 0)}")
        for item in diagnostics.get("per_video", []):
            lines.append(
                f"- {item.get('video')}: {item.get('singleton_tracks', 0)} singleton tracks "
                f"({item.get('singleton_track_percent', 0)}%); longest track "
                f"{item.get('longest_track_observations', 0)} observations"
            )
    lines.extend([
                
        "", "## Interpretation", "",
        (
            "These measurements describe detector output, runtime, and camera-local track continuity. "
            "They do not establish identity switches, cross-camera match correctness, or unique-person "
            "accuracy. Cross-camera identity evaluation requires synchronized camera timestamps and "
            "ground-truth identity labels. The IoU track IDs are local heuristic labels."
        ), "",
    ])
    return "\n".join(lines)


def write_report(summary_path: str | Path, output_path: str | Path,
                 diagnostics_path: str | Path | None = None) -> Path:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    diagnostics = None
    if diagnostics_path is not None and Path(diagnostics_path).is_file():
        diagnostics = json.loads(Path(diagnostics_path).read_text(encoding="utf-8"))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(summary, diagnostics), encoding="utf-8")
    return destination
