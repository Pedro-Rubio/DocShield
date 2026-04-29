from src.pipeline.ocr_extractor import extract_ocr_features
import pytest
import os
import tempfile
import numpy as np
import cv2

def create_test_image(path: str):
    """Crea una imagen con texto simple."""
    img = np.ones((100, 400), dtype=np.uint8) * 255
    cv2.putText(img, "DOCUMENTO OFICIAL", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(path, img)

class TestExtractOCRExtractor:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test_ocr.png")
        create_test_image(self.image_path)
    
    def teardown_method(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        os.rmdir(self.temp_dir)
    
    def test_returns_dict_with_ocr_confidence(self):
        features = extract_ocr_features(self.image_path)
        assert "ocr_confidence" in features
    
    def test_ocr_confidence_range(self):
        features = extract_ocr_features(self.image_path)
        assert 0.0 <= features["ocr_confidence"] <= 1.0
