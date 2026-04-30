import pytest
from src.pipeline.risk_engine import calculate_risk_score
from src.pipeline.risk_engine import get_risk_level_description

class TestCalculateRiskScore:
    def test_returns_dict_with_expected_keys(self):
        features = {
            "blur_score": 200, "edge_density": 0.1, "brightness": 150,
            "contrast": 50, "noise_ratio": 2.0, "symmetry_score": 0.8,
            "color_variance": 500, "ela_score": 5.0, "moire_score": 1.0,
            "dct_score": 0.5, "reflection_score": 0.3, "ocr_confidence": 0.9
        }
        result = calculate_risk_score(features)
        
        expected_keys = ["is_fraud", "confidence", "risk_score", "risk_level", "signals", "threshold_used"]
        assert all(key in result for key in expected_keys)
    
    def test_low_risk(self):
        features = {
            "blur_score": 200, "edge_density": 0.1, "brightness": 150,
            "contrast": 50, "noise_ratio": 2.0, "symmetry_score": 0.8,
            "color_variance": 500, "ela_score": 5.0, "moire_score": 1.0,
            "dct_score": 0.5, "reflection_score": 0.3, "ocr_confidence": 0.9
        }
        result = calculate_risk_score(features)
        assert result["risk_level"] == "LOW"
        assert result["is_fraud"] == False
    
    def test_high_risk(self):
        features = {
            "blur_score": 50, "edge_density": 0.3, "brightness": 100,
            "contrast": 70, "noise_ratio": 10.0, "symmetry_score": 0.6,
            "color_variance": 800, "ela_score": 40.0, "moire_score": 12.0,
            "dct_score": 1.8, "reflection_score": 4.5, "ocr_confidence": 0.4
        }
        result = calculate_risk_score(features)
        assert result["risk_level"] == "HIGH"
        assert result["is_fraud"] == True
    
    def test_signals_populated(self):
        features = {
            "blur_score": 50, "edge_density": 0.3, "brightness": 100,
            "contrast": 70, "noise_ratio": 10.0, "symmetry_score": 0.6,
            "color_variance": 800, "ela_score": 40.0, "moire_score": 12.0,
            "dct_score": 1.8, "reflection_score": 4.5, "ocr_confidence": 0.4
        }
        result = calculate_risk_score(features)
        assert len(result["signals"]) > 0
        assert isinstance(result["signals"], dict)
    
    def test_with_capture_meta(self):
        features = {
            "blur_score": 50, "edge_density": 0.3, "brightness": 100,
            "contrast": 70, "noise_ratio": 10.0, "symmetry_score": 0.6,
            "color_variance": 800, "ela_score": 40.0, "moire_score": 12.0,
            "dct_score": 1.8, "reflection_score": 4.5, "ocr_confidence": 0.4
        }
        capture_meta = {
            "liveness_passed": False,
            "ip_risk_score": 0.8,
            "emulator_detected": 1,
            "tor_detected": 1,
            "vpn_detected": 0,
            "repeated_attempts": 5
        }
        result = calculate_risk_score(features, capture_meta)
        assert result["risk_level"] == "HIGH"  # Por penalización
        assert "liveness" in result["signals"]
    
    def test_configurable_threshold(self):
        features = {
            "blur_score": 150, "edge_density": 0.15, "brightness": 140,
            "contrast": 55, "noise_ratio": 3.0, "symmetry_score": 0.75,
            "color_variance": 550, "ela_score": 20.0, "moire_score": 5.0,
            "dct_score": 0.8, "reflection_score": 1.0, "ocr_confidence": 0.7
        }
        result_low = calculate_risk_score(features, threshold=30.0)
        result_high = calculate_risk_score(features, threshold=50.0)
        assert result_low["is_fraud"] != result_high["is_fraud"]


class TestGetRiskLevelDescription:
    def test_low_description(self):
        desc = get_risk_level_description("LOW")
        assert "pocas" in desc.lower() or "bajo" in desc.lower()
    
    def test_medium_description(self):
        desc = get_risk_level_description("MEDIUM")
        assert "moderada" in desc.lower() or "revisión" in desc.lower()
    
    def test_high_description(self):
        desc = get_risk_level_description("HIGH")
        assert "alto" in desc.lower() or "riesgo" in desc.lower()
