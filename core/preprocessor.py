"""
Image Preprocessor
==================
Handles image loading, enhancement, and quality assessment to maximize
face detection and recognition accuracy, especially for challenging
conditions common in Indian law enforcement datasets (low light, CCTV
footage, uneven lighting, etc.).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

console = Console()


class ImagePreprocessor:
    """Preprocess images for optimal face detection and recognition."""

    # Supported image formats
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            clahe_clip_limit: Contrast limiting for CLAHE (higher = more contrast).
            clahe_grid_size: Grid size for CLAHE adaptive regions.
            target_size: Optional (width, height) to resize images to.
                         None = keep original size.
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.target_size = target_size

        # Create CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size,
        )

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            BGR numpy array, or None if loading failed.
        """
        path = Path(image_path)

        if not path.exists():
            console.print(f"[red][-] File not found:[/red] {image_path}")
            return None

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            console.print(
                f"[red][-] Unsupported format:[/red] {path.suffix} "
                f"(supported: {', '.join(self.SUPPORTED_FORMATS)})"
            )
            return None

        img = cv2.imread(str(path))
        if img is None:
            console.print(f"[red][-] Failed to decode image:[/red] {image_path}")
            return None

        return img

    def enhance(self, image: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
        """
        Enhance an image for better face detection.

        Applies:
        1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
           luminance channel to normalize lighting without color distortion.
        2. Optional resizing.

        Args:
            image: BGR numpy array.
            apply_clahe: Whether to apply CLAHE enhancement.

        Returns:
            Enhanced BGR numpy array.
        """
        result = image.copy()

        # Apply CLAHE on the L channel of LAB color space
        # This normalizes lighting without distorting colors — critical for
        # skin tone preservation across diverse Indian demographics.
        if apply_clahe:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self._clahe.apply(l_channel)
            lab = cv2.merge([l_channel, a_channel, b_channel])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Resize if target size is specified
        if self.target_size is not None:
            result = cv2.resize(
                result,
                self.target_size,
                interpolation=cv2.INTER_AREA,  # Best for downscaling
            )

        return result

    def assess_quality(self, image: np.ndarray) -> dict:
        """
        Assess image quality metrics useful for face recognition.

        Returns a dict with:
            - brightness: Mean brightness (0-255). Ideal: 80-180.
            - contrast: Std deviation of brightness. Ideal: >40.
            - sharpness: Laplacian variance. Ideal: >100.
            - resolution: (height, width) in pixels.
            - quality_score: Overall quality (0-100).
            - warnings: List of quality issues.

        Args:
            image: BGR numpy array.

        Returns:
            Quality metrics dictionary.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        warnings = []
        quality_score = 100.0

        # Check brightness
        if brightness < 50:
            warnings.append("Image is too dark")
            quality_score -= 25
        elif brightness > 220:
            warnings.append("Image is overexposed")
            quality_score -= 20

        # Check contrast
        if contrast < 25:
            warnings.append("Very low contrast")
            quality_score -= 25
        elif contrast < 40:
            warnings.append("Low contrast")
            quality_score -= 10

        # Check sharpness (blur detection)
        if sharpness < 50:
            warnings.append("Image is very blurry")
            quality_score -= 30
        elif sharpness < 100:
            warnings.append("Image is slightly blurry")
            quality_score -= 15

        # Check resolution
        if h < 100 or w < 100:
            warnings.append(f"Very low resolution ({w}x{h})")
            quality_score -= 20
        elif h < 200 or w < 200:
            warnings.append(f"Low resolution ({w}x{h})")
            quality_score -= 10

        quality_score = max(0, min(100, quality_score))

        return {
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1),
            "resolution": (h, w),
            "quality_score": round(quality_score, 1),
            "warnings": warnings,
        }

    def load_and_enhance(
        self, image_path: str, apply_clahe: bool = True
    ) -> Optional[np.ndarray]:
        """
        Convenience method: load an image and enhance it in one step.

        Args:
            image_path: Path to the image file.
            apply_clahe: Whether to apply CLAHE enhancement.

        Returns:
            Enhanced BGR numpy array, or None if loading failed.
        """
        img = self.load_image(image_path)
        if img is None:
            return None
        return self.enhance(img, apply_clahe=apply_clahe)
