"""
Unit tests for CCTVEnhancer module.
"""

import numpy as np
import pytest
from core.cctv_enhancer import CCTVEnhancer


@pytest.fixture
def enhancer():
    return CCTVEnhancer()


@pytest.fixture
def dummy_dark_image():
    # Very dark image simulating night CCTV
    return np.random.randint(5, 40, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def dummy_bright_image():
    # Bright image
    return np.random.randint(180, 250, (240, 320, 3), dtype=np.uint8)


def test_auto_gamma_correction_dark(enhancer, dummy_dark_image):
    enhanced = enhancer.auto_gamma_correction(dummy_dark_image)
    assert enhanced.shape == dummy_dark_image.shape
    assert enhanced.dtype == np.uint8
    assert np.mean(enhanced) > np.mean(dummy_dark_image)


def test_auto_gamma_correction_bright(enhancer, dummy_bright_image):
    enhanced = enhancer.auto_gamma_correction(dummy_bright_image)
    assert enhanced.shape == dummy_bright_image.shape
    assert enhanced.dtype == np.uint8


def test_enhance_contrast_lab(enhancer, dummy_dark_image):
    enhanced = enhancer.enhance_contrast_lab(dummy_dark_image)
    assert enhanced.shape == dummy_dark_image.shape
    assert enhanced.dtype == np.uint8


def test_denoise_bilateral(enhancer, dummy_dark_image):
    denoised = enhancer.denoise_bilateral(dummy_dark_image)
    assert denoised.shape == dummy_dark_image.shape
    assert denoised.dtype == np.uint8


def test_unsharp_mask(enhancer, dummy_dark_image):
    sharpened = enhancer.unsharp_mask(dummy_dark_image)
    assert sharpened.shape == dummy_dark_image.shape
    assert sharpened.dtype == np.uint8


def test_super_resolve_crop(enhancer):
    small_crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    upscaled = enhancer.super_resolve_crop(small_crop, target_size=160)
    assert upscaled.shape == (160, 160, 3)
    assert upscaled.dtype == np.uint8


def test_full_enhance_pipeline(enhancer, dummy_dark_image):
    result = enhancer.enhance(
        dummy_dark_image,
        apply_gamma=True,
        apply_clahe=True,
        apply_denoise=True,
        apply_sharpen=True,
    )
    assert result.shape == dummy_dark_image.shape
    assert result.dtype == np.uint8
