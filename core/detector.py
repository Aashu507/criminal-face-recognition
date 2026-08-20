"""
Face Detector
=============
GPU-accelerated face detection using InsightFace's SCRFD model
(bundled in the buffalo_l model pack).

SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
is a state-of-the-art anchor-free detector that handles:
- Multiple faces per image
- Extreme poses and partial occlusion
- Diverse skin tones and facial structures
- Low-resolution and CCTV-quality inputs
"""

import os
import sys
import glob
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console

console = Console()


def _register_nvidia_dll_dirs():
    """
    Register NVIDIA CUDA DLL directories from pip-installed packages.

    When CUDA libraries (cuBLAS, cuDNN, etc.) are installed via pip
    (e.g., nvidia-cublas-cu12), the DLLs live inside site-packages/nvidia/.
    Windows won't find them unless we explicitly register those directories
    via os.add_dll_directory().

    This must be called BEFORE importing onnxruntime.
    """
    if sys.platform != "win32":
        return

    try:
        # Find the venv's site-packages
        import site
        site_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []

        for site_dir in site_dirs:
            nvidia_base = os.path.join(site_dir, "nvidia")
            if not os.path.isdir(nvidia_base):
                continue

            # Register all bin/ and lib/ directories under nvidia packages
            for pattern in ["*/bin", "*/lib"]:
                for dll_dir in glob.glob(os.path.join(nvidia_base, pattern)):
                    if os.path.isdir(dll_dir):
                        try:
                            os.add_dll_directory(dll_dir)
                        except OSError:
                            pass  # Directory already registered or invalid
    except Exception:
        pass  # Non-critical — falls back to CPU gracefully


# Register NVIDIA DLLs before any ONNX Runtime import
_register_nvidia_dll_dirs()

# Lazy-loaded to avoid import-time model download
_insightface = None


def _get_insightface():
    """Lazy import insightface to defer model download."""
    global _insightface
    if _insightface is None:
        import insightface
        _insightface = insightface
    return _insightface


@dataclass
class DetectedFace:
    """A single detected face with all associated data."""

    bbox: np.ndarray          # [x1, y1, x2, y2] bounding box
    confidence: float         # Detection confidence (0-1)
    landmarks: np.ndarray     # 5-point facial landmarks (eyes, nose, mouth corners)
    embedding: Optional[np.ndarray] = None  # 512-dim ArcFace embedding (set by recognizer)
    age: Optional[int] = None
    gender: Optional[str] = None  # 'M' or 'F'

    # Metadata for display
    face_image: Optional[np.ndarray] = None  # Cropped face region

    @property
    def bbox_ints(self) -> List[int]:
        """Bounding box as integer list [x1, y1, x2, y2]."""
        return [int(c) for c in self.bbox]

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector:
    """
    Detect faces in images using InsightFace SCRFD.

    Uses the buffalo_l model pack by default, which includes:
    - det_10g.onnx (SCRFD face detector, 10GFlops variant)
    - 2d106det.onnx (106-point landmark detector)
    - genderage.onnx (gender & age estimator)

    Models auto-download on first use to ~/.insightface/models/
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        gpu_id: int = 0,
        det_size: int = 640,
        det_thresh: float = 0.5,
    ):
        """
        Initialize the face detector.

        Args:
            model_name: InsightFace model pack name.
                        'buffalo_l' = best accuracy (recommended).
                        'buffalo_s' = faster, slightly less accurate.
            gpu_id: GPU device ID. 0 = first GPU, -1 = CPU only.
            det_size: Detection input size. Higher = more accurate but slower.
                      Options: 320, 480, 640 (default), 800.
            det_thresh: Minimum detection confidence threshold (0-1).
                        Lower = more detections (more false positives).
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.det_size = (det_size, det_size)
        self.det_thresh = det_thresh
        self._app = None  # Lazy initialization

    def _ensure_initialized(self):
        """Initialize InsightFace app on first use (triggers model download)."""
        if self._app is not None:
            return

        insightface = _get_insightface()

        console.print(
            f"[cyan]⟳ Initializing InsightFace[/cyan] "
            f"(model={self.model_name}, gpu={self.gpu_id}, det_size={self.det_size[0]})"
        )

        # Set up execution providers based on GPU availability
        if self.gpu_id >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self._app = insightface.app.FaceAnalysis(
                name=self.model_name,
                providers=providers,
            )
            self._app.prepare(ctx_id=self.gpu_id, det_size=self.det_size)

            # Report which provider is active
            active_provider = "GPU (CUDA)" if self.gpu_id >= 0 else "CPU"
            console.print(
                f"[green]✓ InsightFace ready[/green] — "
                f"Provider: {active_provider}, "
                f"Models: {len(self._app.models)} loaded"
            )

        except Exception as e:
            console.print(f"[red]✗ InsightFace initialization failed:[/red] {e}")
            if "CUDA" in str(e):
                console.print(
                    "[yellow]  Hint: Ensure CUDA toolkit and onnxruntime-gpu "
                    "are installed correctly.[/yellow]"
                )
            raise

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = 0,
        extract_crops: bool = True,
    ) -> List[DetectedFace]:
        """
        Detect all faces in an image.

        Args:
            image: BGR numpy array (OpenCV format).
            max_faces: Maximum faces to return (0 = unlimited).
            extract_crops: Whether to extract cropped face images.

        Returns:
            List of DetectedFace objects sorted by confidence (highest first).
        """
        self._ensure_initialized()

        # Run InsightFace detection + analysis
        raw_faces = self._app.get(image, max_num=max_faces)

        if not raw_faces:
            return []

        detected = []
        for face in raw_faces:
            # Filter by confidence threshold
            det_score = float(face.det_score) if hasattr(face, "det_score") else 0.0
            if det_score < self.det_thresh:
                continue

            # Extract cropped face region
            face_crop = None
            if extract_crops:
                bbox = face.bbox.astype(int)
                h, w = image.shape[:2]
                x1 = max(0, bbox[0])
                y1 = max(0, bbox[1])
                x2 = min(w, bbox[2])
                y2 = min(h, bbox[3])
                if x2 > x1 and y2 > y1:
                    face_crop = image[y1:y2, x1:x2].copy()

            # Determine gender string
            gender_str = None
            if hasattr(face, "gender") and face.gender is not None:
                gender_str = "M" if face.gender == 1 else "F"

            detected.append(
                DetectedFace(
                    bbox=face.bbox,
                    confidence=det_score,
                    landmarks=face.kps if hasattr(face, "kps") else np.array([]),
                    embedding=face.normed_embedding if hasattr(face, "normed_embedding") else None,
                    age=int(face.age) if hasattr(face, "age") and face.age is not None else None,
                    gender=gender_str,
                    face_image=face_crop,
                )
            )

        # Sort by confidence (highest first)
        detected.sort(key=lambda f: f.confidence, reverse=True)

        return detected

    def detect_single(self, image: np.ndarray) -> Optional[DetectedFace]:
        """
        Detect the most prominent (highest confidence) face in an image.

        Args:
            image: BGR numpy array.

        Returns:
            Single DetectedFace, or None if no face detected.
        """
        faces = self.detect(image, max_faces=1)
        return faces[0] if faces else None
