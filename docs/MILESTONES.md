# Milestones

## M0 — Scaffold and architecture
- [x] Define the pipeline and camera topology configuration.
- [x] Add typed tracklet/association models and heuristic candidate scoring.
- [x] Add deterministic tests and synthetic dashboard.
- [ ] Review scaffold locally and run preflight.

## M1 — Hardware and data preflight
- [ ] Confirm Python, OpenCV, PyTorch/CUDA and GPU memory.
- [ ] Select a dataset sequence and verify its license/annotations.
- [ ] Establish a reproducible baseline run.

## M2 — Recorded replay and local tracking
- [ ] Add timestamped video reader and frame sampler.
- [ ] Integrate pretrained person detector and local tracker.
- [ ] Save local tracklets and replay diagnostics.

## M3 — Re-identification and global association
- [ ] Integrate compact pretrained person ReID encoder.
- [ ] Implement candidate widening fallback and calibrated thresholds.
- [ ] Persist global identity hypotheses and link explanations.

## M4 — Reliability evaluation
- [ ] Add cross-camera metrics and error review tools.
- [ ] Measure selective risk/coverage and calibrate confidence.
- [ ] Compare appearance-only, topology-gated, and uncertainty-aware variants.

## M5 — Dashboard and authorized live feeds
- [ ] Add real track timelines and evidence inspection.
- [ ] Add bounded asynchronous RTSP ingestion only after replay is stable.
- [ ] Add health, latency, dropped-frame, and memory indicators.
