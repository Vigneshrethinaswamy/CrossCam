# Milestones

## M0 — Scaffold and architecture
- [x] Define the pipeline and camera topology configuration.
- [x] Add typed tracklet/association models and heuristic candidate scoring.
- [x] Add deterministic tests and dashboard scaffold.
- [x] Review scaffold locally and run preflight.

## M1 — Hardware and data preflight
- [x] Confirm Python, OpenCV, PyTorch/CUDA and GPU memory.
- [ ] Select a dataset sequence and verify its license/annotations.
- [ ] Establish a reproducible baseline run on an authorized video.

## M2 — Recorded replay and local tracking
- [x] Add timestamped video reader and frame sampler.
- [x] Integrate pretrained person detector and a lightweight local IoU tracker.
- [x] Add batch evaluation and camera-local continuity diagnostics.
- [x] Add dashboard views and Markdown report generation for evaluation artifacts.
- [ ] Run the end-to-end pipeline on an authorized sample video.
- [ ] Measure sampled FPS, inference latency, and peak GPU memory on the target laptop.

## M3 — Experimental cross-camera association
- [x] Add HSV appearance descriptors for extracted camera-local tracklets.
- [x] Add time/topology candidate ranking with explanations and accept/defer/reject bands.
- [ ] Validate camera offsets and candidate behavior on synchronized camera videos.
- [ ] Validate a pretrained person ReID encoder and its weights before replacing the HSV descriptor.
- [x] Implement the standalone OSNet x0.25 adapter with hash-verified safe loading and vendored MIT source.
- [x] Record the target-laptop synthetic CUDA smoke test; OSNet produced finite 512-D outputs at batch sizes 1 and 4.
- [x] Verify the adapter on Windows: 65 tests passed and Ruff reported all checks passed at `bb5db36`.
- [ ] Measure CPU fallback and CPU/GPU embedding parity on the target laptop.
- [x] Add a local Market-1501 retrieval runner and protocol tests; the dataset is acquired for local research-only use.
- [ ] Run Market-1501 retrieval only after the MSMT17 checkpoint's downstream usage rights are explicitly cleared.
- [ ] Persist reviewed global identity hypotheses with human confirmation.

## M4 — Reliability evaluation
- [ ] Annotate a representative, licensed evaluation set with identity labels.
- [ ] Add cross-camera metrics and error review tools.
- [ ] Measure selective risk/coverage and calibrate confidence on held-out data.
- [ ] Compare appearance-only, topology-gated, and uncertainty-aware variants.

## M5 — Dashboard and authorized live feeds
- [x] Show measured evaluation diagnostics and experimental candidate links.
- [ ] Add real track timelines and evidence inspection.
- [ ] Add bounded asynchronous RTSP ingestion only after replay is stable.
- [ ] Add health, latency, dropped-frame, and memory indicators.
