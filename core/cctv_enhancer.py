"""
CCTV & Low-Resolution Image Enhancer
====================================
Advanced image processing pipeline specifically designed for low-resolution,
grainy, night-vision, and poorly-lit CCTV surveillance footage of Indian demographics.

Features:
- Gamma correction & Shadow lift for dark/harsh lighting
- CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB color space (preserves skin tones)
- Fast Edge-Preserving Bilateral Filtering (removes sensor noise / compression artifacts)
- Unsharp Masking & Laplacian detail boost for face recognition super-resolution
- Auto-exposure adjustment
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CCTVEnhancer:
    """
    Enhances low-quality CCTV frames to boost face detection and recognition accuracy.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        denoise_strength: int = 5,
        sharpen_strength: float = 1.2,
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size
        )
        self.denoise_strength = denoise_strength
        self.sharpen_strength = sharpen_strength

    def auto_gamma_correction(self, image: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
        """
        Dynamically adjusts gamma based on overall image brightness.
        Brightens underexposed night/CCTV footage while preventing washout.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_mean = float(np.mean(gray))

        if current_mean < 10.0:
            current_mean = 10.0
        elif current_mean > 245.0:
            current_mean = 245.0

        # Calculate gamma factor
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = float(np.clip(gamma, 0.3, 2.5))

        # LUT for fast nonlinear transformation: (i / 255)^gamma * 255
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_contrast_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE on the L (Luminance) channel in LAB color space.
        Preserves natural skin undertones crucial for Indian face recognition.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def denoise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Applies bilateral filtering to smooth out JPEG compression blocks & sensor noise
        while preserving sharp facial edges (eyes, nose, jawline).
        """
        return cv2.bilateralFilter(
            image,
            d=self.denoise_strength,
            sigmaColor=50,
            sigmaSpace=50
        )

    def unsharp_mask(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Enhances high-frequency edge details and facial features.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(
            image, 1.0 + self.sharpen_strength, blurred, -self.sharpen_strength, 0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def super_resolve_crop(self, face_crop: np.ndarray, target_size: int = 160) -> np.ndarray:
        """
        Upscales and sharpens small low-res face crops extracted from CCTV.
        """
        h, w = face_crop.shape[:2]
        if h < target_size or w < target_size:
            # Bicubic interpolation
            resized = cv2.resize(
                face_crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC
            )
            # Gentle unsharp mask for resized crop
            return self.unsharp_mask(resized, sigma=0.8)
        return face_crop

    def enhance(
        self,
        image: np.ndarray,
        apply_gamma: bool = True,
        apply_clahe: bool = True,
        apply_denoise: bool = True,
        apply_sharpen: bool = True,
    ) -> np.ndarray:
        """
        Full CCTV enhancement pipeline.
        """
        result = image.copy()

        if apply_gamma:
            result = self.auto_gamma_correction(result)

        if apply_clahe:
            result = self.enhance_contrast_lab(result)

        if apply_denoise:
            result = self.denoise_bilateral(result)

        if apply_sharpen:
            result = self.unsharp_mask(result)

        return result
