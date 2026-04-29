import numpy as np
import cv2
import pytest
from src.pipeline.visual_extractor import extract_visual_features, compute_ela
import os
import tempfile

def create_test_image(path: str, size=(200, 300)):
    """Crea una imagen de prueba simple."""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (size[1]-20, size[0]-20), (0, 0, 0), 2)
    cv2.imwrite(path, img)
    return path

def create_test_jpeg(path: str, size=(200, 300), quality=95):
    """Crea una imagen JPEG para probar ELA."""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path

class TestExtractVisualFeatures:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test_doc.png")
        create_test_image(self.image_path)
    
    def teardown_method(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        os.rmdir(self.temp_dir)
    
    def test_returns_dict_with_expected_keys(self):
        features = extract_visual_features(self.image_path)
        expected_keys = ["blur_score", "edge_density", "brightness", 
                        "contrast", "noise_ratio", "symmetry_score", "color_variance"]
        assert all(key in features for key in expected_keys)
    
    def test_blur_score_positive(self):
        features = extract_visual_features(self.image_path)
        assert features["blur_score"] >= 0
    
    def test_brightness_range(self):
        features = extract_visual_features(self.image_path)
        assert 0 <= features["brightness"] <= 255
    
    def test_edge_density_range(self):
        features = extract_visual_features(self.image_path)
        assert 0 <= features["edge_density"] <= 1

class TestComputeELA:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test_ela.jpg")
        create_test_jpeg(self.image_path, quality=95)
    
    def teardown_method(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        os.rmdir(self.temp_dir)
    
    def test_returns_float(self):
        ela_score = compute_ela(self.image_path)
        assert isinstance(ela_score, float)
    
    def test_ela_positive(self):
        ela_score = compute_ela(self.image_path)
        assert ela_score >= 0
