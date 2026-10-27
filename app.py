"""Synthetic CrossCam dashboard scaffold. No real tracking data is shown yet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from crosscam.config import load_config

st.set_page_config(page_title="CrossCam", page_icon="📹", layout="wide")
config = load_config(ROOT / "configs" / "default.yaml")

st.title("CrossCam")
st.caption("Uncertainty-aware multi-camera tracking — architecture scaffold")
st.warning("This dashboard currently displays synthetic placeholder data, not real camera detections or identity matches.")

left, right = st.columns(2)
with left:
    st.subheader("Configured cameras")
    st.dataframe([{"Camera ID": c["id"], "Label": c.get("label", c["id"])} for c in config["cameras"]],
                 use_container_width=True, hide_index=True)
with right:
    st.subheader("Runtime settings")
    st.metric("Target sampled FPS / feed", config["runtime"]["target_sample_fps"])
    st.metric("Max queue size", config["runtime"]["max_queue_size"])
    st.metric("Topology transitions", len(config.get("transitions", [])))

st.subheader("Architecture")
st.code("Video source → sampler → detector → local tracker → tracklets → ReID → candidate generation → association → store → dashboard", language="text")
st.subheader("Next implementation milestone")
st.markdown("Add a recorded-video reader, then integrate a pretrained person detector and local tracker. Keep this dashboard separate from model inference.")
