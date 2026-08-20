"""
Unit tests for CCTVVideoScanner module.
"""

import tempfile
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock
from core.video_scanner import CCTVVideoScanner, VideoDetectionEvent
from core.detector import DetectedFace


@pytest.fixture
def mock_matcher():
    matcher = MagicMock()
    # Mock search_image returning one face with match
    face = DetectedFace(
        bbox=np.array([50, 50, 150, 150]),
        confidence=0.98,
        landmarks=np.zeros((5, 2)),
        embedding=np.random.rand(512).astype(np.float32)
    )
    matcher.search_image.return_value = [{
        "face": face,
        "matches": [{
            "id": "CRIM-TEST",
            "name": "Test Suspect",
            "similarity": 0.75,
            "metadata": {"crime": "Theft"}
        }]
    }]
    return matcher


def test_timestamp_formatting():
    assert CCTVVideoScanner.format_timestamp(65.5) == "01:05.500"
    assert CCTVVideoScanner.format_timestamp(0.0) == "00:00.000"


def test_video_scanner_mock(mock_matcher):
    # Create a synthetic 10-frame video
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tf:
        temp_video = tf.name

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_video, fourcc, 10.0, (320, 240))
    for _ in range(10):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    scanner = CCTVVideoScanner(mock_matcher, frame_step=2, min_confidence=0.5)
    events_collected = []

    for frame_idx, timestamp_sec, annotated, events in scanner.scan_video_stream(temp_video):
        events_collected.extend(events)
        assert annotated.shape == (240, 320, 3)

    assert len(events_collected) == 5  # 10 frames / step 2 = 5 scanned frames
    assert events_collected[0].criminal_id == "CRIM-TEST"
