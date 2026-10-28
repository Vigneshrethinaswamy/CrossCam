"""CrossCam dashboard for recorded-video evaluation and association artifacts."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
EVALUATION_DIR = ROOT / "outputs" / "evaluation"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"
ASSOCIATION_PATH = ROOT / "outputs" / "associations.csv"

st.set_page_config(page_title="CrossCam", page_icon="📹", layout="wide")
st.title("CrossCam")
st.caption("Recorded-video person detection and camera-local tracking")

summary_path = EVALUATION_DIR / "summary.json"
diagnostics_path = DIAGNOSTICS_DIR / "diagnostics_summary.json"
track_path = DIAGNOSTICS_DIR / "track_diagnostics.csv"

if not summary_path.is_file():
    st.info(
        "No evaluation run is available yet. Add authorized video files under data/ and run "
        "scripts/evaluate_videos.py, then run scripts/diagnose_tracks.py to populate this view."
    )
    st.code(
        "python scripts/evaluate_videos.py --input-dir data --output-dir outputs/evaluation\n"
        "python scripts/diagnose_tracks.py --input-dir outputs/evaluation "
        "--output-dir outputs/diagnostics",
        language="powershell",
    )
    st.warning(
        "Cross-camera identity association is experimental and is not evaluated by this dashboard. "
        "Local track IDs are not verified identities."
    )
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
results = summary.get("results", [])
videos = summary.get("videos_processed", 0)
tracks = sum(item.get("distinct_local_track_ids_observed", 0) for item in results)
frames = sum(item.get("sampled_frames", 0) for item in results)
detections = sum(item.get("total_person_detections", 0) for item in results)
a, b, c, d = st.columns(4)
a.metric("Videos", videos)
b.metric("Sampled frames", frames)
c.metric("Person detections", detections)
d.metric("Observed local track IDs", tracks)

st.subheader("Run settings")
st.json({
    "device": summary.get("device"),
    "sample_target_fps": summary.get("sample_target_fps"),
    "confidence_threshold": summary.get("confidence_threshold"),
    "iou_threshold": summary.get("iou_threshold"),
    "max_missed_frames": summary.get("max_missed_frames"),
})

if results:
    st.subheader("Per-video results")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

if diagnostics_path.is_file():
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    st.subheader("Track continuity diagnostics")
    st.caption(diagnostics.get("interpretation", "Continuity summaries are not identity accuracy."))
    per_video = diagnostics.get("per_video", [])
    if per_video:
        st.dataframe(pd.DataFrame(per_video), use_container_width=True, hide_index=True)
    if track_path.is_file():
        with st.expander("Track-level details"):
            st.dataframe(pd.read_csv(track_path), use_container_width=True, hide_index=True)
else:
    st.info("Run scripts/diagnose_tracks.py to add track continuity summaries.")

if ASSOCIATION_PATH.is_file():
    st.subheader("Experimental cross-camera candidates")
    st.caption(
        "Ranked heuristic suggestions from mean HSV color histograms, elapsed time, and camera "
        "topology. Accept/defer/reject are uncalibrated decisions and are not identity verification."
    )
    associations = pd.read_csv(ASSOCIATION_PATH)
    st.dataframe(associations, use_container_width=True, hide_index=True)
else:
    st.info(
        "To inspect experimental candidates, run scripts/extract_tracklets.py and "
        "scripts/associate_tracklets.py after evaluating videos from configured cameras."
    )

st.warning(
    "These outputs describe detections and heuristic candidate rankings. Without synchronized "
    "timestamps and ground-truth identity labels, they do not measure cross-camera match accuracy "
    "or prove that two track IDs belong to the same person."
)
