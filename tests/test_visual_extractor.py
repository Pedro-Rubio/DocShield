"""Tests para el módulo de extracción de features visuales."""

import os
import cv2
import numpy as np
import pytest

from src.pipeline.visual_extractor import (
    extract_visual_features,
    compute_ela,
    _laplacian_variance,
    _edge_density,
    _noise_ratio,
    _symmetry_score,
    _color_variance,
)


@pytest.fixture
def sample_image(tmp_path) -> str:
    """Crea una imagen de prueba y retorna su ruta."""
    img_path = str(tmp_path / "test_doc.png")
    # Crear una imagen tipo documento (rectángulo claro con texto simulado)
    img = np.ones((300, 500, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (50, 50), (450, 250), (200, 200, 200), -1)
    cv2.putText(img, "DNI TEST", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(img_path, img)
    return img_path


@pytest.fixture
def blurry_image(tmp_path) -> str:
    """Crea una imagen borrosa para pruebas."""
    img_path = str(tmp_path / "test_blurry.png")
    img = np.ones((300, 500, 3), dtype=np.uint8) * 200
    img = cv2.GaussianBlur(img, (21, 21), 0)
    cv2.imwrite(img_path, img)
    return img_path


class TestExtractVisualFeatures:
    def test_returns_all_keys(self, sample_image):
        features = extract_visual_features(sample_image)
        expected_keys = {
            "blur_score",
            "edge_density",
            "brightness",
            "contrast",
            "noise_ratio",
            "symmetry_score",
            "color_variance",
        }
        assert set(features.keys()) == expected_keys

    def test_blur_score_positive(self, sample_image):
        features = extract_visual_features(sample_image)
        assert features["blur_score"] > 0

    def test_blur_score_blurry_vs_sharp(self, blurry_image, sample_image):
        features_blurry = extract_visual_features(blurry_image)
        features_sharp = extract_visual_features(sample_image)
        # La imagen borrosa debe tener menor blur_score
        assert features_blurry["blur_score"] < features_sharp["blur_score"]

    def test_edge_density_range(self, sample_image):
        features = extract_visual_features(sample_image)
        assert 0.0 <= features["edge_density"] <= 1.0

    def test_brightness_range(self, sample_image):
        features = extract_visual_features(sample_image)
        assert 0.0 <= features["brightness"] <= 255.0

    def test_contrast_positive(self, sample_image):
        features = extract_visual_features(sample_image)
        assert features["contrast"] >= 0.0

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            extract_visual_features("/nonexistent/path/image.jpg")


class TestHelperFunctions:
    def test_laplacian_variance_sharp(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
        variance = _laplacian_variance(img)
        assert variance > 0

    def test_edge_density_returns_float(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        density = _edge_density(img)
        assert isinstance(density, float)
        assert 0.0 <= density <= 1.0

    def test_noise_ratio_returns_float(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        ratio = _noise_ratio(img)
        assert isinstance(ratio, float)
        assert ratio >= 0.0

    def test_symmetry_score_range(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        score = _symmetry_score(img)
        assert isinstance(score, float)

    def test_color_variance_returns_float(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        variance = _color_variance(img)
        assert isinstance(variance, float)


class TestComputeELA:
    def test_ela_score_positive(self, sample_image):
        score = compute_ela(sample_image)
        assert score >= 0.0

    def test_ela_with_different_quality(self, sample_image):
        score_90 = compute_ela(sample_image, quality=90)
        score_50 = compute_ela(sample_image, quality=50)
        # Ambos deben ser no negativos
        assert score_90 >= 0.0
        assert score_50 >= 0.0

    def test_ela_cleans_up_temp_file(self, sample_image):
        compute_ela(sample_image)
        temp_path = sample_image + ".ela_tmp.jpg"
        assert not os.path.exists(temp_path)
