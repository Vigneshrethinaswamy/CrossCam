from crosscam.appearance import hsv_histogram
from crosscam.association import associate
from crosscam.config import load_config
from crosscam.models import Tracklet
from crosscam.topology import CameraTopology
import numpy as np


def test_hsv_histogram_is_normalized_and_deterministic():
    image = np.zeros((20, 10, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    feature = hsv_histogram(image)
    assert len(feature) == 64
    assert sum(feature) == pytest.approx(1.0)
