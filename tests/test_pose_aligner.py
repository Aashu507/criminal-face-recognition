"""
Unit tests for PoseAligner module (5-point affine alignment and head pose estimation).
"""

import numpy as np
import pytest
from core.pose_aligner import PoseAligner, CANONICAL_5PTS


@pytest.fixture
def aligner():
    return PoseAligner()


@pytest.fixture
def frontal_landmarks():
    return np.array([
        [38.0, 51.0],  # left eye
        [74.0, 51.0],  # right eye
        [56.0, 71.0],  # nose
        [41.0, 92.0],  # left mouth
        [71.0, 92.0]   # right mouth
    ], dtype=np.float32)


@pytest.fixture
def rotated_landmarks():
    # Rotated landmarks simulating tilted CCTV camera
    theta = np.radians(25)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    center = np.array([56.0, 71.0])
    pts = np.dot(CANONICAL_5PTS - center, R.T) + center
    return pts.astype(np.float32)


def test_align_face_5pts_output_dimensions(aligner, frontal_landmarks):
    dummy_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    aligned = aligner.align_face_5pts(dummy_img, frontal_landmarks)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_align_rotated_face(aligner, rotated_landmarks):
    dummy_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    aligned = aligner.align_face_5pts(dummy_img, rotated_landmarks)
    assert aligned.shape == (112, 112, 3)


def test_estimate_pose_angles_frontal(aligner, frontal_landmarks):
    angles = aligner.estimate_pose_angles(frontal_landmarks)
    assert "yaw" in angles and "pitch" in angles and "roll" in angles
    assert abs(angles["yaw"]) < 15.0
    assert abs(angles["pitch"]) < 15.0
    assert abs(angles["roll"]) < 15.0
    assert not angles["is_extreme"]


def test_estimate_pose_angles_extreme_yaw(aligner):
    # Skewed nose representing turned head
    turned_pts = np.array([
        [30.0, 50.0],
        [80.0, 50.0],
        [75.0, 70.0],  # Nose shifted far right
        [35.0, 90.0],
        [75.0, 90.0]
    ], dtype=np.float32)
    angles = aligner.estimate_pose_angles(turned_pts)
    assert abs(angles["yaw"]) > 30.0
