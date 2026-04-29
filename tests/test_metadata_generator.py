"""Tests para el módulo de generación de metadatos de sesión."""

import pytest
from src.pipeline.metadata_generator import generate_session_metadata

class TestGenerateSessionMetadata:
    def test_returns_dict(self):
        result = generate_session_metadata()
        assert isinstance(result, dict)
    
    def test_has_expected_keys(self):
        result = generate_session_metadata()
        expected_keys = {
            "ip_risk_score",
            "emulator_detected",
            "tor_detected",
            "vpn_detected",
            "repeated_attempts",
            "liveness_passed",
        }
        assert set(result.keys()) == expected_keys
    
    def test_ip_risk_score_range(self):
        result = generate_session_metadata()
        assert 0 <= result["ip_risk_score"] <= 1
    
    def test_emulator_detected_is_bool_like(self):
        result = generate_session_metadata()
        assert result["emulator_detected"] in [0, 1, True, False]
    
    def test_liveness_passed_is_bool_like(self):
        result = generate_session_metadata()
        assert result["liveness_passed"] in [0, 1, True, False]
