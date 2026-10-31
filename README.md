# CrossCam
**Uncertainty-aware multi-camera person tracking and cross-camera identity association.**

CrossCam is a research-oriented prototype for tracking people across four camera views. It starts with reproducible recorded-video experiments and is designed to grow toward authorized live RTSP feeds.

> **Current status:** recorded-video replay, pretrained person detection, camera-local IoU tracking, batch evaluation, continuity diagnostics, a report generator, and a dashboard for evaluation artifacts are implemented. An experimental HSV-histogram candidate workflow is available; cross-camera identity accuracy is not validated.

## Goals
- Maintain local tracks independently in each camera.
- Rank possible cross-camera tracklet links using appearance, time, and camera topology.
- Represent uncertain matches explicitly (accept / defer / reject).
- Evaluate identity quality and uncertainty instead of optimizing accuracy alone.

## Architecture
VideoSource → FrameSampler → Detector → LocalTracker → Tracklets → Appearance descriptor → CandidateGenerator → AssociationEngine → Dashboard

See [architecture](docs/ARCHITECTURE.md), [evaluation](docs/EVALUATION.md), [cross-camera experiment](docs/CROSS_CAMERA_EXPERIMENT.md), [dataset notes](docs/DATASETS.md), [milestones](docs/MILESTONES.md), and [environment setup](docs/ENVIRONMENT.md).

## Hardware-aware plan
Target hardware is a Windows laptop with an RTX 3050 Laptop GPU (4 GB VRAM) and 16 GB RAM. The detector uses Torchvision's pretrained SSDLite MobileNetV3-Large. Actual throughput and VRAM use must be measured locally.

## Quick start (Windows PowerShell)
~~~powershell
cd C:\Works\CrossCam
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
~~~

## Recorded-video evaluation and report
Place only authorized videos under data/, named with configured camera IDs (for example, cam_01.mp4). The first detector run may download pretrained weights.

~~~powershell
.\.venv\Scripts\python.exe scripts\evaluate_videos.py --input-dir data --output-dir outputs\evaluation --sample-fps 3 --device auto --max-missed-seconds 2.0
.\.venv\Scripts\python.exe scripts\diagnose_tracks.py --input-dir outputs\evaluation --output-dir outputs\diagnostics
.\.venv\Scripts\python.exe scripts\make_report.py
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

The report is written to outputs/crosscam_report.md. The dashboard reads evaluation and diagnostic outputs. Use scripts/replay_video.py for an interactive single-video replay.

## Experimental cross-camera candidate ranking
After evaluating videos from configured cameras, derive HSV crop histograms and rank tracklet candidates on a shared timeline:

~~~powershell
.\.venv\Scripts\python.exe scripts\extract_tracklets.py --input-dir outputs\evaluation --video-dir data --output outputs\tracklets.json
.\.venv\Scripts\python.exe scripts\associate_tracklets.py --tracklets outputs\tracklets.json --camera-offset cam_01=0 --camera-offset cam_02=30 --output outputs\associations.csv
~~~

Set offsets from trusted camera recording start times; do not infer them from tracking outputs. Supply an offset for every camera whose timestamps you want compared on a shared timeline. Cameras without an explicit offset are marked temporally unknown; their local time origins are never silently assumed to be synchronized. Weak topology candidates trigger a widened search that preserves existing candidates and adds alternatives within the configured horizon. Decision margins can defer ambiguous top-ranked links. This uses simple color histograms, not a learned ReID model. Candidate links and accept/defer/reject labels remain heuristic and uncalibrated.

The local tracker uses camera-local IDs and bounding-box overlap only. IDs can switch when people cross, disappear, or change appearance.

## Important limitations and privacy
- Cross-camera candidate ranking has not been validated on synchronized, identity-labeled footage and does not establish that a match is correct.
- Without ground-truth identity labels, do not report identity accuracy, ID switches, or false merge rates as measured outcomes.
- No video files, model weights, person embeddings, or biometric data belong in Git.
- Only use footage and datasets you are authorized to process. Avoid covert tracking and follow applicable privacy rules.
