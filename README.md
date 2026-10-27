# CrossCam
**Uncertainty-aware multi-camera person tracking and cross-camera identity association.**

CrossCam is a research-oriented prototype for tracking people across four camera views. It starts with reproducible recorded-video experiments and is designed to grow toward authorized live RTSP feeds.

> **Current status:** architecture and evaluation scaffold. The initial association logic is heuristic; real detector/ReID inference and validated tracking metrics are not yet implemented.

## Goals
- Maintain local tracks independently in each camera.
- Build tracklets and associate them across cameras using appearance, time, and camera topology.
- Represent uncertain matches explicitly (accept / defer / reject).
- Expand beyond nearby cameras when local candidates are weak, to avoid brittle route assumptions.
- Evaluate identity quality and uncertainty instead of optimizing accuracy alone.

## Architecture
`VideoSource → FrameSampler → Detector → LocalTracker → TrackletBuilder → ReIDEncoder → CandidateGenerator → AssociationEngine → TrackStore → Dashboard`

See [architecture](docs/ARCHITECTURE.md), [evaluation plan](docs/EVALUATION.md), [dataset notes](docs/DATASETS.md), [milestones](docs/MILESTONES.md), and [environment setup](docs/ENVIRONMENT.md).

## Hardware-aware plan
Target hardware is a Windows laptop with an RTX 3050 Laptop GPU (4 GB VRAM) and 16 GB RAM. Use pretrained, compact models; sample frames; resize inputs; keep association on CPU; bound queues; and avoid training from scratch. Throughput targets are hypotheses to measure, not guarantees.

## Quick start (Windows PowerShell)
```powershell
cd C:\Works
git clone https://github.com/Vigneshrethinaswamy/CrossCam.git
cd CrossCam
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
# Install CUDA-enabled PyTorch; see docs/ENVIRONMENT.md
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
.\.venv\Scripts\python.exe scripts\check_gpu.py
python scripts/preflight.py
pytest
streamlit run app.py
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly or review your local execution policy. Keep each project in its own virtual environment.

## Repository layout
- `configs/`: camera topology and tunable settings
- `docs/`: architecture, datasets, evaluation, and milestones
- `src/crosscam/`: typed data models and starter association logic
- `scripts/preflight.py`: environment and configuration checks
- `scripts/check_gpu.py`: PyTorch CUDA execution check
- `tests/`: deterministic unit tests
- `app.py`: synthetic dashboard scaffold
- `data/`, `models/`: local-only runtime artifacts (not committed)

## Important limitations
- The included dashboard uses synthetic example data, not real camera footage.
- The included association score is a heuristic, **not a calibrated probability**.
- No video files, model weights, person embeddings, or biometric data belong in Git.
- Only use footage and datasets you are authorized to process. Avoid covert tracking and follow applicable privacy rules.

## Development sequence
1. Validate environment and topology configuration.
2. Add recorded-video input and local tracking.
3. Add tracklet embeddings and cross-camera candidate association.
4. Establish baselines and evaluate identity errors and selective uncertainty.
5. Add dashboard diagnostics, then authorized live streams.
