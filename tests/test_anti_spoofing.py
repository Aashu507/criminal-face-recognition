"""
Unit tests for AntiSpoofingDetector module.
"""

import numpy as np
import pytest
from core.anti_spoofing import AntiSpoofingDetector


@pytest.fixture
def detector():
    return AntiSpoofingDetector(liveness_threshold=0.50)


def test_evaluate_liveness_normal_face(detector):
    # Simulated skin texture with natural gradient and color variance
    face = np.random.randint(90, 180, (128, 128, 3), dtype=np.uint8)
    # Add skin-like chrominance
    face[:, :, 2] += 20  # Red channel emphasis
    is_live, score, metrics = detector.evaluate_liveness(face)

    assert 0.0 <= score <= 1.0
    assert "fourier_score" in metrics
    assert "chroma_score" in metrics
    assert "attack_type_flag" in metrics


def test_evaluate_liveness_flat_screen_pattern(detector):
    # Flat single color simulating monochromatic or blank screen replay
    flat_screen = np.ones((128, 128, 3), dtype=np.uint8) * 128
    is_live, score, metrics = detector.evaluate_liveness(flat_screen)

    # Liveness score on flat image should be low
    assert score < 0.60


def test_empty_crop(detector):
    empty = np.array([])
    is_live, score, metrics = detector.evaluate_liveness(empty)
    assert not is_live
    assert score == 0.0
