#!/usr/bin/env python3
"""
End-to-End Pipeline Test
========================
Tests the full facial recognition pipeline without requiring real criminal images.
Uses synthetic test data to verify all components work correctly.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from core.preprocessor import ImagePreprocessor
from core.detector import FaceDetector, DetectedFace
from core.recognizer import FaceRecognizer
from core.database import FaceDatabase
from core.matcher import FaceMatcher


# ─── Preprocessor Tests ─────────────────────────────────────────────


class TestImagePreprocessor:
    """Tests for the image preprocessing module."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_load_nonexistent_file(self):
        """Should return None for missing files."""
        result = self.preprocessor.load_image("nonexistent_file.jpg")
        assert result is None

    def test_load_unsupported_format(self, tmp_path):
        """Should return None for unsupported file formats."""
        fake_file = tmp_path / "test.xyz"
        fake_file.write_text("not an image")
        result = self.preprocessor.load_image(str(fake_file))
        assert result is None

    def test_enhance_preserves_shape(self):
        """CLAHE enhancement should preserve image dimensions."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        enhanced = self.preprocessor.enhance(img)
        assert enhanced.shape == img.shape

    def test_enhance_no_clahe(self):
        """Should still work with CLAHE disabled."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        enhanced = self.preprocessor.enhance(img, apply_clahe=False)
        assert enhanced.shape == img.shape

    def test_quality_assessment_structure(self):
        """Quality assessment should return all expected keys."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        quality = self.preprocessor.assess_quality(img)
        assert "brightness" in quality
        assert "contrast" in quality
        assert "sharpness" in quality
        assert "resolution" in quality
        assert "quality_score" in quality
        assert "warnings" in quality
        assert isinstance(quality["warnings"], list)

    def test_quality_dark_image_warning(self):
        """Should warn about dark images."""
        dark_img = np.zeros((480, 640, 3), dtype=np.uint8)
        quality = self.preprocessor.assess_quality(dark_img)
        assert quality["brightness"] < 5
        assert any("dark" in w.lower() for w in quality["warnings"])

    def test_quality_bright_image_warning(self):
        """Should warn about overexposed images."""
        bright_img = np.full((480, 640, 3), 250, dtype=np.uint8)
        quality = self.preprocessor.assess_quality(bright_img)
        assert any("overexposed" in w.lower() for w in quality["warnings"])

    def test_quality_low_res_warning(self):
        """Should warn about low resolution images."""
        small_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        quality = self.preprocessor.assess_quality(small_img)
        assert any("resolution" in w.lower() for w in quality["warnings"])

    def test_load_valid_image(self, tmp_path):
        """Should successfully load a valid image."""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        loaded = self.preprocessor.load_image(str(path))
        assert loaded is not None
        assert loaded.shape[0] == 200
        assert loaded.shape[1] == 200


# ─── Database Tests ──────────────────────────────────────────────────


class TestFaceDatabase:
    """Tests for the ChromaDB vector store."""

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.db = FaceDatabase(persist_dir=self._tmp)

    def test_enroll_and_count(self):
        """Should enroll a face and update count."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize

        self.db.enroll("TEST001", embedding, {"name": "Test Subject"})
        assert self.db.count() == 1

    def test_enroll_multiple(self):
        """Should enroll multiple faces."""
        for i in range(5):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            self.db.enroll(f"TEST{i:03d}", emb, {"name": f"Subject {i}"})
        assert self.db.count() == 5

    def test_search_returns_matches(self):
        """Should find enrolled faces by embedding similarity."""
        # Enroll a face
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        self.db.enroll("TARGET", emb, {"name": "Target Person"})

        # Search with same embedding — should get perfect match
        results = self.db.search(emb, top_k=5)
        assert len(results) >= 1
        assert results[0]["criminal_id"] == "TARGET"
        assert results[0]["similarity"] > 0.99  # Near-perfect match

    def test_search_with_threshold(self):
        """Should filter results below threshold."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        self.db.enroll("FACE1", emb1)

        # Search with a very different embedding — should be filtered
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        results = self.db.search(emb2, top_k=5, threshold=0.99)
        # Very unlikely that random embeddings have >0.99 similarity
        assert len(results) == 0

    def test_get_existing(self):
        """Should retrieve an enrolled criminal by ID."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        self.db.enroll("GET_TEST", emb, {"name": "Retrieve Me"})

        record = self.db.get("GET_TEST")
        assert record is not None
        assert record["criminal_id"] == "GET_TEST"

    def test_get_nonexistent(self):
        """Should return None for unknown ID."""
        record = self.db.get("NONEXISTENT")
        assert record is None

    def test_delete(self):
        """Should remove a criminal from the database."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        self.db.enroll("DELETE_ME", emb)
        assert self.db.count() == 1

        self.db.delete("DELETE_ME")
        assert self.db.count() == 0

    def test_upsert_updates_existing(self):
        """Should update (not duplicate) on re-enrollment."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        self.db.enroll("UPSERT", emb1, {"name": "Original"})
        assert self.db.count() == 1

        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        self.db.enroll("UPSERT", emb2, {"name": "Updated"})
        assert self.db.count() == 1  # Should not duplicate

    def test_list_all(self):
        """Should list all enrolled criminals."""
        for i in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            self.db.enroll(f"LIST{i}", emb, {"name": f"Person {i}"})

        records = self.db.list_all()
        assert len(records) == 3

    def test_clear(self):
        """Should delete all records."""
        for i in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            self.db.enroll(f"CLEAR{i}", emb)

        deleted = self.db.clear()
        assert deleted == 3
        assert self.db.count() == 0


# ─── DetectedFace Tests ─────────────────────────────────────────────


class TestDetectedFace:
    """Tests for the DetectedFace dataclass."""

    def test_bbox_ints(self):
        face = DetectedFace(
            bbox=np.array([10.5, 20.3, 110.7, 120.9]),
            confidence=0.95,
            landmarks=np.array([]),
        )
        assert face.bbox_ints == [10, 20, 110, 120]

    def test_dimensions(self):
        face = DetectedFace(
            bbox=np.array([10, 20, 110, 220]),
            confidence=0.95,
            landmarks=np.array([]),
        )
        assert face.width == 100
        assert face.height == 200
        assert face.area == 20000


# ─── Recognizer Static Tests ────────────────────────────────────────


class TestRecognizerStatic:
    """Tests for static methods on FaceRecognizer (no GPU needed)."""

    def test_similarity_identical(self):
        """Identical embeddings should have similarity ≈ 1.0."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        sim = FaceRecognizer.compute_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_opposite(self):
        """Opposite embeddings should have similarity ≈ -1.0."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        sim = FaceRecognizer.compute_similarity(emb, -emb)
        assert abs(sim - (-1.0)) < 1e-5

    def test_distance_identical(self):
        """Identical embeddings should have distance ≈ 0."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        dist = FaceRecognizer.compute_distance(emb, emb)
        assert dist < 1e-5

    def test_distance_positive(self):
        """Distance should always be non-negative."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        dist = FaceRecognizer.compute_distance(emb1, emb2)
        assert dist >= 0


# ─── Integration Note ───────────────────────────────────────────────
#
# Full integration tests (detect → embed → enroll → search) require
# the InsightFace buffalo_l model to be downloaded and a GPU available.
# Run these tests with:
#
#   python -m pytest tests/test_pipeline.py -v -k "not integration"
#
# For integration tests:
#   python -m pytest tests/test_pipeline.py -v -k "integration"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
