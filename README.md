# CrossCam
**Uncertainty-aware multi-camera person tracking and cross-camera identity association.**

CrossCam is a research-oriented prototype for tracking people across four camera views. It starts with reproducible recorded-video experiments and is designed to grow toward authorized live RTSP feeds.

> **Current status:** recorded-video replay, pretrained person detection, camera-local IoU tracking, batch evaluation, continuity diagnostics, a report generator, and a dashboard for evaluation artifacts are implemented. Cross-camera ReID remains experimental; no identity accuracy is claimed.

## Goals
- Maintain local tracks independently in each camera.
- Build tracklets and associate them across cameras using appearance, time, and camera topology.
- Represent uncertain matches explicitly (accept / defer / reject).
- Expand beyond nearby cameras when local candidates are weak, to avoid brittle route assumptions.
- Evaluate identity quality and uncertainty instead of optimizing accuracy alone.

## Architecture
VideoSource → FrameSampler → Detector → LocalTracker → Tracklets → ReIDEncoder → CandidateGenerator → AssociationEngine → TrackStore → Dashboard

See [architecture](docs/ARCHITECTURE.md), [evaluation](docs/EVALUATION.md), [dataset notes](docs/DATASETS.md), [milestones](docs/MILESTONES.md), and [environment setup](docs/ENVIRONMENT.md).

## Hardware-aware plan
Target hardware is a Windows laptop with an RTX 3050 Laptop GPU (4 GB VRAM) and 16 GB RAM. The default detector is Torchvision's pretrained SSDLite MobileNetV3-Large. Frames are sampled, association stays on CPU, and training from scratch is avoided. Actual throughput and VRAM use must be measured locally.

## Quick start (Windows PowerShell)
~~~powershell
cd C:\Works\CrossCam
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
~~~

## Recorded-video evaluation
Place only authorized videos under data/ (video files are ignored by Git). The first detector run may download pretrained weights.

~~~powershell
.\.venv\Scripts\python.exe scripts\evaluate_videos.py --input-dir data --output-dir outputs\evaluation --sample-fps 3 --device auto
.\.venv\Scripts\python.exe scripts\diagnose_tracks.py --input-dir outputs\evaluation --output-dir outputs\diagnostics
.\.venv\Scripts\python.exe scripts\make_report.py
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

The report is written to outputs/crosscam_report.md; the dashboard reads the evaluation and diagnostics artifacts. Use scripts/replay_video.py for an interactive single-video replay.

The local tracker uses camera-local IDs and bounding-box overlap only. IDs can switch when people cross, disappear, or change appearance. This is a baseline, not a robust MOT/ReID solution.

## Important limitations and privacy
- Cross-camera ReID/association has not been validated end to end. The available association score is a heuristic, not a calibrated probability.
- Without synchronized camera timestamps and ground-truth identity labels, do not interpret proposed matches as verified identities or report identity accuracy.
- No video files, model weights, person embeddings, or biometric data belong in Git.
- Only use footage and datasets you are authorized to process. Avoid covert tracking and follow applicable privacy rules.

## Development sequence
1. Validate environment and topology configuration.
2. Run recorded-video detection and local tracking on authorized samples.
3. Add validated tracklet appearance features and cross-camera candidate association.
4. Establish ground-truth baselines and evaluate identity errors and uncertainty.
5. Expand the dashboard and consider authorized live streams only after replay is stable.
