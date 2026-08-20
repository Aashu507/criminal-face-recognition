"""
Facial Recognition System — Core Engine
========================================
A locally-running facial recognition pipeline for criminal identification,
optimized for Indian face demographics and GPU-accelerated inference.

Components:
    - preprocessor: Image preprocessing (CLAHE, quality assessment)
    - detector: Face detection via InsightFace SCRFD
    - recognizer: ArcFace embedding extraction
    - database: ChromaDB vector store for criminal embeddings
    - matcher: Similarity matching engine
"""

from core.preprocessor import ImagePreprocessor
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.database import FaceDatabase
from core.matcher import FaceMatcher
from core.cctv_enhancer import CCTVEnhancer
from core.video_scanner import CCTVVideoScanner
from core.pose_aligner import PoseAligner
from core.adaface_recognizer import AdaFaceRecognizer

__all__ = [
    "ImagePreprocessor",
    "FaceDetector",
    "FaceRecognizer",
    "FaceDatabase",
    "FaceMatcher",
    "CCTVEnhancer",
    "CCTVVideoScanner",
    "PoseAligner",
    "AdaFaceRecognizer",
]
