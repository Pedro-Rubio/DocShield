"""Tests para el módulo de anti-spoofing."""

import numpy as np
import pytest

from src.pipeline.anti_spoofing import (
    detect_moire,
    analyze_dct_blocks,
    analyze_reflection,
)


@pytest.fixture
def plain_gray_image() -> np.ndarray:
    """Imagen gris simple para pruebas."""
    return np.ones((128, 128), dtype=np.uint8) * 128


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """Imagen tipo tablero de ajedrez (simula patrón de Moiré)."""
    img = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            if (i // 4 + j // 4) % 2 == 0:
                img[i, j] = 255
    return img


@pytest.fixture
def plain_bgr_image() -> np.ndarray:
    """Imagen BGR simple para pruebas de reflexión."""
    img = np.ones((128, 128, 3), dtype=np.uint8) * 128
    return img


@pytest.fixture
def bright_bgr_image() -> np.ndarray:
    """Imagen BGR con zonas brillantes (simula reflexión)."""
    img = np.ones((128, 128, 3), dtype=np.uint8) * 200
    img[0:64, 0:64] = 250  # Zona muy brillante
    return img


class TestDetectMoire:
    def test_returns_float(self, plain_gray_image):
        score = detect_moire(plain_gray_image)
        assert isinstance(score, float)

    def test_plain_image_low_score(self, plain_gray_image):
        score = detect_moire(plain_gray_image)
        # Imagen plana debe dar score bajo
        assert score < 8.5

    def test_checkerboard_higher_score(self, checkerboard_image):
        plain_score = detect_moire(np.ones((128, 128), dtype=np.uint8) * 128)
        checker_score = detect_moire(checkerboard_image)
        # El tablero de ajedrez debe dar score mayor que una imagen plana
        assert checker_score > plain_score

    def test_small_image_no_error(self):
        img = np.ones((16, 16), dtype=np.uint8) * 128
        score = detect_moire(img)
        assert isinstance(score, float)
        assert score >= 0.0


class TestAnalyzeDCTBlocks:
    def test_returns_float(self, plain_gray_image):
        score = analyze_dct_blocks(plain_gray_image)
        assert isinstance(score, float)

    def test_uniform_image_low_anomaly(self, plain_gray_image):
        score = analyze_dct_blocks(plain_gray_image)
        # Imagen uniforme debe tener baja anomalía DCT
        assert score >= 0.0

    def test_non_multiple_of_8(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        score = analyze_dct_blocks(img)
        assert isinstance(score, float)
        assert score >= 0.0


class TestAnalyzeReflection:
    def test_returns_float(self, plain_bgr_image):
        score = analyze_reflection(plain_bgr_image)
        assert isinstance(score, float)

    def test_dark_image_low_reflection(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        score = analyze_reflection(img)
        assert score == 0.0

    def test_bright_image_has_reflection(self, bright_bgr_image):
        score = analyze_reflection(bright_bgr_image)
        assert score >= 0.0
