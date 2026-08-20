"""
Face Recognizer
===============
Extracts 512-dimensional ArcFace embeddings from detected faces.

The ArcFace model (w600k_r50, bundled with buffalo_l) produces highly
discriminative face embeddings that can be compared via cosine similarity.
It was trained on WebFace600K — a dataset of 600K+ identities spanning
global demographics, which provides strong baseline performance on
Indian face recognition without fine-tuning.

Embeddings are L2-normalized, so cosine similarity = dot product.
"""

import numpy as np
from typing import List, Optional
from rich.console import Console

from core.detector import FaceDetector, DetectedFace

console = Console()


class FaceRecognizer:
    """
    Extract face embeddings using ArcFace (via InsightFace).

    The recognizer builds on top of FaceDetector — it uses the same
    InsightFace app instance, which already computes embeddings during
    detection. This class adds convenience methods for:

    - Extracting embeddings from images directly
    - Batch processing multiple images
    - Computing similarity between two faces
    """

    def __init__(self, detector: Optional[FaceDetector] = None, **detector_kwargs):
        """
        Initialize the recognizer.

        Args:
            detector: Existing FaceDetector instance to reuse.
                      If None, creates a new one with default settings.
            **detector_kwargs: Passed to FaceDetector() if creating a new one.
        """
        if detector is not None:
            self.detector = detector
        else:
            self.detector = FaceDetector(**detector_kwargs)

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the face embedding from an image containing a single face.

        Args:
            image: BGR numpy array.

        Returns:
            512-dim normalized embedding, or None if no face detected.
        """
        face = self.detector.detect_single(image)
        if face is None or face.embedding is None:
            return None
        return face.embedding

    def get_all_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Get embeddings for all faces detected in an image.

        Args:
            image: BGR numpy array.

        Returns:
            List of 512-dim normalized embeddings.
        """
        faces = self.detector.detect(image)
        return [f.embedding for f in faces if f.embedding is not None]

    def get_face_with_embedding(
        self, image: np.ndarray
    ) -> Optional[DetectedFace]:
        """
        Detect the primary face and return the full DetectedFace object
        (with embedding, bbox, landmarks, age, gender).

        Args:
            image: BGR numpy array.

        Returns:
            DetectedFace with embedding populated, or None.
        """
        return self.detector.detect_single(image)

    @staticmethod
    def compute_similarity(
        embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two face embeddings.

        Since InsightFace embeddings are already L2-normalized,
        cosine similarity = dot product.

        Args:
            embedding1: 512-dim normalized embedding.
            embedding2: 512-dim normalized embedding.

        Returns:
            Similarity score in range [-1, 1]. Higher = more similar.
            Typical thresholds:
              > 0.45: Likely same person
              > 0.55: High confidence same person
              < 0.30: Definitely different people
        """
        return float(np.dot(embedding1, embedding2))

    @staticmethod
    def compute_distance(
        embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute Euclidean distance between two face embeddings.

        Args:
            embedding1: 512-dim normalized embedding.
            embedding2: 512-dim normalized embedding.

        Returns:
            Distance score >= 0. Lower = more similar.
        """
        diff = embedding1 - embedding2
        return float(np.sqrt(np.dot(diff, diff)))

    def verify(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.45,
    ) -> dict:
        """
        1:1 Face verification — are these two images the same person?

        Args:
            image1: BGR numpy array of first face.
            image2: BGR numpy array of second face.
            threshold: Similarity threshold for "same person" decision.

        Returns:
            Dict with:
                - verified: bool (same person or not)
                - similarity: float (cosine similarity)
                - distance: float (Euclidean distance)
                - threshold: float (threshold used)
        """
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)

        if emb1 is None:
            return {"verified": False, "error": "No face detected in image 1"}
        if emb2 is None:
            return {"verified": False, "error": "No face detected in image 2"}

        similarity = self.compute_similarity(emb1, emb2)
        distance = self.compute_distance(emb1, emb2)

        return {
            "verified": similarity >= threshold,
            "similarity": round(similarity, 4),
            "distance": round(distance, 4),
            "threshold": threshold,
        }
