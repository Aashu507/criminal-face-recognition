"""
Unit tests for FaceTracker module (IoU, trajectory tracking, query throttling).
"""

import time
import numpy as np
import pytest
from core.tracker import FaceTracker, compute_iou, TrackedFace
from core.detector import DetectedFace


def test_compute_iou():
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    assert compute_iou(box1, box2) == 1.0

    box3 = [50, 0, 150, 100]
    # Intersection = 50x100 = 5000, Union = 15000 -> IoU = 1/3
    assert np.isclose(compute_iou(box1, box3), 0.3333, atol=1e-3)

    box_disjoint = [200, 200, 300, 300]
    assert compute_iou(box1, box_disjoint) == 0.0


def test_tracker_creation_and_persistence():
    tracker = FaceTracker(iou_threshold=0.3)

    det1 = DetectedFace(
        bbox=np.array([10, 10, 80, 80]),
        confidence=0.95,
        landmarks=np.zeros((5, 2)),
        embedding=np.random.rand(512).astype(np.float32)
    )

    t0 = 1000.0
    tracks_frame1 = tracker.update([det1], current_time=t0)
    assert len(tracks_frame1) == 1
    track_id = tracks_frame1[0].track_id
    assert track_id == 101

    # Frame 2: slightly shifted bounding box (same person)
    det2 = DetectedFace(
        bbox=np.array([12, 11, 82, 81]),
        confidence=0.96,
        landmarks=np.zeros((5, 2)),
        embedding=det1.embedding
    )
    tracks_frame2 = tracker.update([det2], current_time=t0 + 0.5)
    assert len(tracks_frame2) == 1
    assert tracks_frame2[0].track_id == track_id  # Persistent ID!
    assert tracks_frame2[0].hits == 2
    assert tracks_frame2[0].dwell_time_seconds == 0.5


def test_tracker_query_throttling():
    tracker = FaceTracker(query_cooldown_seconds=2.0)
    track = TrackedFace(
        track_id=101,
        bbox=[0, 0, 50, 50],
        confidence=0.9,
        first_seen=100.0,
        last_seen=100.0,
        last_query_time=100.0
    )

    assert not tracker.should_query_database(track, current_time=101.0)
    assert tracker.should_query_database(track, current_time=102.5)
