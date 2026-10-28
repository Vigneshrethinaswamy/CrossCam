# Experimental cross-camera candidates

CrossCam can now create an offline candidate list from per-video evaluation outputs. The appearance descriptor is a mean HSV hue/saturation histogram computed from each detected person crop. It uses the existing OpenCV dependency and adds no pretrained model weights.

## Run

Name video files with the configured camera IDs (for example, cam_01.mp4 and cam_02.mp4) and place authorized footage under data/. Run evaluation separately as one batch:

~~~powershell
.\.venv\Scripts\python.exe scripts\evaluate_videos.py --input-dir data --output-dir outputs\evaluation --sample-fps 3 --device auto
.\.venv\Scripts\python.exe scripts\extract_tracklets.py --input-dir outputs\evaluation --video-dir data --output outputs\tracklets.json
.\.venv\Scripts\python.exe scripts\associate_tracklets.py --tracklets outputs\tracklets.json --camera-offset cam_01=0 --camera-offset cam_02=30 --output outputs\associations.csv
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

Offsets put each camera's video-local timestamps on a shared timeline. Use measured recording start times where possible. For example, 30 means that camera 2's local time zero occurred 30 seconds after the camera 1 reference. Offsets may be negative; CrossCam normalizes them while preserving relative timing. Do not infer offsets from candidate results.

The association command ranks time-compatible candidates using the existing heuristic appearance/time/topology score and accept/defer/reject thresholds. It emits candidate links, not one-to-one global identities. The HSV descriptor can be dominated by background, lighting, and clothing; it is a weak appearance cue and is not a learned person ReID embedding.

## Interpretation and validation

Treat every score and decision as an experimental ranking only. Thresholds are not calibrated probabilities. Do not claim that a match is correct or report identity accuracy without synchronized videos and ground-truth identity labels. The evaluation report and dashboard repeat this limitation.

The CSV and JSON inputs/outputs may contain derived appearance descriptors. Keep them local and do not commit videos, embeddings, or other biometric data. Use only footage you are authorized to process.
