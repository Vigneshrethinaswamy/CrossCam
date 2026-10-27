"""Check the local Python environment and CrossCam configuration."""
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import psutil

from crosscam.config import load_config


def main() -> int:
    print("CrossCam preflight")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"OpenCV: {cv2.__version__}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GiB")
    config = load_config(ROOT / "configs" / "default.yaml")
    print(f"Cameras: {len(config['cameras'])}")
    print(f"Transitions: {len(config.get('transitions', []))}")
    print("CUDA: run scripts/check_gpu.py to validate PyTorch GPU execution.")
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
