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
Cameras are nodes. Directed edges represent plausible transitions and expected travel-time intervals. Explicit camera-pair relationships may be known-overlap, known-non-overlap, or unknown. An absent transition edge is not proof of non-overlap. Same-time tracklets may be excluded only when offsets are trusted and the pair is known non-overlapping. Duplicate directed edges and duplicate undirected relationships are rejected rather than overwritten.

## Association and uncertainty
The starter implementation combines simple appearance similarity, elapsed time when both camera offsets are explicit, and a topology prior only when one is explicitly configured. Missing temporal or topology evidence is represented as unknown and receives a neutral contribution; it is not silently converted to zero offset or maximal prior. Its score is a ranking heuristic, not a probability. It must not be displayed as a probability until calibrated on held-out data. Thresholds and decision margins must be selected using validation data and reported alongside coverage.

When the best topology-supported candidate is below the configured weak-score threshold, candidate generation widens and returns the union of topology-supported and additional eligible candidates. Widened candidates do not inherit a positive topology prior. The widening horizon and time-decay scale are explicit configuration values.

- **Accept:** score and margin satisfy validation-selected criteria.
- **Defer:** evidence is ambiguous; wait for more observations or request wider candidate search.
- **Reject:** candidate evidence is too weak or contradictory.

Measure both selective risk and coverage so that deferring every match cannot appear successful.

## Runtime layout
The camera-local tracker expires unmatched tracks using elapsed video seconds rather than a count of sampled frames. Callers must pass monotonic frame timestamps. Keep inference queues bounded. A slow camera should not grow an unbounded backlog. Record timestamps and dropped-frame counts. Keep heavy model inference separate from SQLite writes and UI refreshes.
