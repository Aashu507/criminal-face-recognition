"""
Unit tests for AdaFaceRecognizer module (quality scoring, adaptive similarity).
"""

import numpy as np
import pytest
from core.adaface_recognizer import AdaFaceRecognizer
from core.pose_aligner import PoseAligner


@pytest.fixture
def adaface():
    return AdaFaceRecognizer(base_recognizer=None, pose_aligner=PoseAligner())


def test_quality_score_sharp_vs_blurry(adaface):
    # Sharp gradient image
    sharp_img = np.zeros((112, 112, 3), dtype=np.uint8)
    sharp_img[::4, ::4] = 255
    q_sharp = adaface.compute_quality_score(sharp_img)

    # Completely flat blurry image
    blurry_img = np.ones((112, 112, 3), dtype=np.uint8) * 128
    q_blurry = adaface.compute_quality_score(blurry_img)

    assert q_sharp > q_blurry
    assert 0.0 <= q_sharp <= 1.0
    assert 0.0 <= q_blurry <= 1.0


def test_extract_adaptive_embedding(adaface):
    dummy_img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    landmarks = np.array([
        [40.0, 50.0],
        [75.0, 50.0],
        [57.0, 70.0],
        [42.0, 90.0],
        [72.0, 90.0]
    ], dtype=np.float32)

    emb, q_score, telemetry = adaface.extract_adaptive_embedding(dummy_img, landmarks=landmarks)
    assert emb.shape == (512,)
    # Verify unit norm: ||emb|| ≈ 1.0
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
    assert "quality_score" in telemetry
    assert "pose" in telemetry


def test_adaptive_similarity_identical_embeddings(adaface):
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    sim_high_q = adaface.adaptive_similarity(emb, emb, quality1=0.9, quality2=0.9)
    assert np.isclose(sim_high_q, 1.0, atol=1e-3)

    sim_low_q = adaface.adaptive_similarity(emb, emb, quality1=0.3, quality2=0.3)
    assert sim_low_q > 0.8  # Adaptive soft margin allows match despite low quality
