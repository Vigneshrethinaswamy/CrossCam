# Track diagnostics

This step adds **track-continuity diagnostics** to the recorded-video evaluation workflow. It does not change the detector or IoU tracker, so previous experiments remain comparable.

## 1. Run the baseline evaluation

From PowerShell in `C:\Works\CrossCam`:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_videos.py --input-dir data --output-dir outputs\evaluation --sample-fps 3 --device auto --confidence 0.45 --iou-threshold 0.30 --max-missed-frames 10
```

## 2. Generate track diagnostics

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_tracks.py --input-dir outputs\evaluation --output-dir outputs\diagnostics
```

Outputs:
- `outputs\diagnostics\track_diagnostics.csv`: one row per observed camera-local track, including observation count, start/end time, span, largest time gap, and mean/min detector score.
- `outputs\diagnostics\diagnostics_summary.json`: per-video track counts, median observations per track, singleton-track percentage, longest track, and tracks with gaps over one second.

The input directory can be changed to compare another experiment, such as `outputs\experiment_iou020` or `outputs\experiment_retention20`. Keep each experiment's output directory separate.

## How to interpret this

- **Singleton track:** a local ID appeared in only one sampled frame. This can be a short genuine appearance, a false positive, or a fragmentation symptom; the metric alone cannot tell which.
- **Observations per track:** counts sampled detections assigned to that local ID, not original video frames or unique people.
- **Maximum observation gap:** the largest time interval between consecutive observations for one ID. Large gaps may indicate intermittent detection or missed samples.
- **Detector score:** the detector's confidence-like score, not a calibrated probability that the box is correct.

Lower ID counts, fewer singletons, or longer tracks do not automatically mean better tracking: a tracker can also incorrectly merge different people. These summaries are diagnostic proxies, not IDF1/HOTA or ground-truth identity accuracy. For trustworthy accuracy metrics, annotate a small representative evaluation set with person boxes and identity labels.

## Why do tracks fragment?

The current tracker greedily matches detections by bounding-box intersection-over-union (IoU). It has no motion prediction and no appearance features. When a box shifts, a person is occluded, or detections disappear, overlap may fall below the threshold and a new local ID is created.

The next stage is to inspect diagnostic output and add a compact appearance encoder behind an interface, then compare it to this preserved IoU baseline. Appearance similarity is a cue, not proof of identity; cross-camera association will also need time and camera-topology constraints.
