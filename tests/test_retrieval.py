import numpy as np
import pytest

from crosscam.retrieval import evaluate_market1501, market1501_identity_camera


def test_market_filename_parser():
    assert market1501_identity_camera("0007_c3s2_000123_01.jpg") == (7, 3)
    assert market1501_identity_camera("-1_c6s4_000100_00.jpg") == (-1, 6)
    with pytest.raises(ValueError):
        market1501_identity_camera("person.jpg")


def test_market_metrics_filter_junk_and_same_camera_matches():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    result = evaluate_market1501(q, [1], [1], g, [-1, 1, 1], [2, 1, 2], ks=(1, 2))
    assert result["valid_queries"] == 1
    assert result["rank_1"] == 1.0
    assert result["mean_average_precision"] == 1.0


def test_market_metrics_compute_cmc_and_average_precision():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    result = evaluate_market1501(q, [1], [1], g, [2, 1, 1], [2, 2, 3], ks=(1, 2, 5))
    assert result["rank_1"] == 0.0
    assert result["rank_2"] == 1.0
    assert result["mean_average_precision"] == pytest.approx(0.5)
    assert result["recall_at_2"] == 0.5


def test_market_metrics_reject_bad_shapes_and_nonfinite_features():
    with pytest.raises(ValueError):
        evaluate_market1501(
            np.array([[1.0]]), [1], [1], np.array([[1.0, 0.0]]), [1], [2]
        )
    with pytest.raises(ValueError):
        evaluate_market1501(
            np.array([[np.nan]]), [1], [1], np.array([[1.0]]), [1], [2]
        )
