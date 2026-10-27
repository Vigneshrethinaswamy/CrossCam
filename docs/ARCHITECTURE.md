# CrossCam Architecture

## Design principles
1. **Tracklets, not frame-by-frame global matching:** local trackers produce short tracklets; cross-camera reasoning operates over tracklets.
2. **Topology guides, never dictates:** camera adjacency and travel-time windows prioritize candidates. If neighboring-camera candidates are absent or weak, widen the search.
3. **Uncertainty is an output:** association decisions can be accepted, deferred for more evidence, or rejected.
4. **Reproducibility first:** recorded-video replay is the first implementation target; live ingest comes later.
5. **Resource-aware execution:** pretrained compact models, sampled frames, bounded queues, resized images, and CPU-side association.

## Pipeline
`VideoSource → FrameSampler → Detector → LocalTracker → TrackletBuilder → ReIDEncoder → CandidateGenerator → AssociationEngine → TrackStore → Dashboard`

- **VideoSource:** file replay initially; authorized RTSP streams later.
- **FrameSampler:** controls inference rate and handles timestamps.
- **Detector:** pretrained person detector; inference adapter to be selected after preflight.
- **LocalTracker:** maintains per-camera track IDs.
- **TrackletBuilder:** aggregates observations into tracklets.
- **ReIDEncoder:** computes appearance embeddings with a pretrained person ReID model.
- **CandidateGenerator:** applies topology and elapsed-time gates, then broadens candidate search when needed.
- **AssociationEngine:** ranks candidate links and returns accept/defer/reject.
- **TrackStore:** SQLite-backed local metadata store; no raw frames by default.
- **Dashboard:** shows tracks, tentative links, timelines, and uncertainty.

## Camera graph
Cameras are nodes. Directed edges represent plausible transitions and expected travel-time intervals. The graph is a candidate-prioritization mechanism, not a hard constraint: people can take unexpected routes or a camera may miss a track.

## Association and uncertainty
The starter implementation combines a simple appearance similarity, time compatibility, and topology prior. Its score is a ranking heuristic. It must not be displayed as a probability until calibrated on held-out data. Thresholds must be selected using validation data and reported alongside coverage.

- **Accept:** score and margin satisfy validation-selected criteria.
- **Defer:** evidence is ambiguous; wait for more observations or request wider candidate search.
- **Reject:** candidate evidence is too weak or contradictory.

Measure both selective risk and coverage so that deferring every match cannot appear successful.

## Runtime layout
Keep inference queues bounded. A slow camera should not grow an unbounded backlog. Record timestamps and dropped-frame counts. Keep heavy model inference separate from SQLite writes and UI refreshes.
