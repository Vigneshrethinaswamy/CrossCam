# CrossCam
**Uncertainty-aware multi-camera person tracking and cross-camera identity association.**

CrossCam is a research-oriented prototype for tracking people across four camera views. It starts with reproducible recorded-video experiments and is designed to grow toward authorized live RTSP feeds.

> **Current status:** recorded-video replay, frame sampling, pretrained person detection, and camera-local IoU tracking are implemented. Cross-camera ReID and global identity association are still future milestones.

## Goals
- Maintain local tracks independently in each camera.
- Build tracklets and associate them across cameras using appearance, time, and camera topology.
- Represent uncertain matches explicitly (accept / defer / reject).
- Expand beyond nearby cameras when local candidates are weak, to avoid brittle route assumptions.
- Evaluate identity quality and uncertainty instead of optimizing accuracy alone.

## Architecture
`VideoSource → FrameSampler → Detector → LocalTracker → Tracklets → ReIDEncoder → CandidateGenerator → AssociationEngine → TrackStore → Dashboard`

See [architecture](docs/ARCHITECTURE.md), [evaluation](docs/EVALUATION.md), [dataset notes](docs/DATASETS.md), [milestones](docs/MILESTONES.md), and [environment setup](docs/ENVIRONMENT.md).

## Hardware-aware plan
Target hardware is a Windows laptop with an RTX 3050 Laptop GPU (4 GB VRAM) and 16 GB RAM. The default detector is Torchvision's pretrained SSDLite MobileNetV3-Large, designed to be relatively compact. Frames are sampled, association stays on CPU, and training from scratch is avoided. Actual throughput and VRAM use must be measured locally.

## Quick start (Windows PowerShell)
~~~powershell
cd C:\Works\CrossCam
git pull origin main
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_gpu.py
~~~

## Run recorded-video replay
Place a video you are authorized to process somewhere outside Git-tracked source folders, for example `data\sample.mp4`. Video files are ignored by Git.

~~~powershell
.\.venv\Scripts\python.exe scripts\replay_video.py --video data\sample.mp4 --camera-id cam_01 --sample-fps 3 --device auto --display
~~~

Omit `--display` for headless processing. The first detector run may download pretrained weights. The model automatically selects CUDA when available, otherwise CPU. If GPU memory is tight, lower `--sample-fps`, close other GPU-heavy applications, or use `--device cpu`.

The local IoU tracker uses camera-local IDs and bounding-box overlap only. IDs can switch when people cross, disappear, or change appearance; this is a baseline, not a robust MOT/ReID solution.

## Repository layout
- `configs/`: camera topology and tunable settings
- `docs/`: architecture, datasets, evaluation, and milestones
- `src/crosscam/video.py`: recorded video reader and timestamp sampler
- `src/crosscam/detection.py`: pretrained Torchvision person detector
- `src/crosscam/tracker.py`: camera-local IoU tracker
- `src/crosscam/`: core data models and starter association logic
- `scripts/preflight.py`: environment and configuration checks
- `scripts/check_gpu.py`: PyTorch CUDA execution check
- `scripts/replay_video.py`: end-to-end recorded replay runner
- `tests/`: deterministic unit tests
- `app.py`: synthetic dashboard scaffold

## Important limitations and privacy
- The dashboard still displays synthetic example data; it is not yet wired to replay output.
- The included local tracker does not perform cross-camera identity matching.
- The included association score is a heuristic, **not a calibrated probability**.
- No video files, model weights, person embeddings, or biometric data belong in Git.
- Only use footage and datasets you are authorized to process. Avoid covert tracking and follow applicable privacy rules.

## Development sequence
1. Validate environment and topology configuration.
2. Add recorded-video input and local tracking.
3. Add tracklet embeddings and cross-camera candidate association.
4. Establish baselines and evaluate identity errors and selective uncertainty.
5. Add dashboard diagnostics, then authorized live streams.
