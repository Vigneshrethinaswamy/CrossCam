# Evaluation Plan

## Baselines
1. Appearance-only association.
2. Appearance plus elapsed-time gating.
3. Topology/time-gated candidate generation.
4. Topology plus uncertainty-aware accept/defer/reject.
5. Full candidate widening fallback.

## Metrics
- HOTA and IDF1 for multi-object tracking where annotations and tooling permit.
- ID switches, fragmentation, and per-camera tracking quality.
- Cross-camera match precision, recall, and F1.
- False identity merge rate (different people assigned one global identity).
- Selective risk versus coverage for deferred/accepted matches.
- ECE and Brier score only after probability calibration on held-out data.
- Candidate count, inference FPS, end-to-end latency, VRAM and RAM use.

## Experimental protocol
1. Split by video sequence/session, not random adjacent frames.
2. Keep threshold selection/calibration data separate from final test data.
3. Fix model versions, configuration, and random seeds where applicable.
4. Report per-camera and per-transition results as well as aggregate metrics.
5. Compare topology gating with an ungated baseline; record missed true candidates caused by gating.
6. Report performance at multiple sampling rates and image sizes.
7. Record every deferred match and count it in coverage analysis.

Do not report synthetic dashboard data as model performance. All initial scores are placeholders until real annotated sequences are evaluated.
