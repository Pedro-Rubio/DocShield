"""Tests para el módulo de generación de metadatos de sesión."""

import pytest

from src.pipeline.metadata_generator import (
    generate_session_metadata,
    _compute_ip_risk,
    _detect_emulator,
    _detect_tor,
    _detect_vpn,
    _count_repeated_attempts,
    _evaluate_liveness,
    _compute_device_fingerprint,
)


class TestGenerateSessionMetadata:
    def test_returns_all_keys(self):
        capture_meta = {
            "user_agent": "Mozilla/5.0",
            "screen_width": 1080,
            "screen_height": 1920,
            "platform": "android",
            "ip_address": "192.168.1.1",
            "liveness_passed": True,
            "accelerometer_data": [0.1, 0.2, 0.3],
        }
        result = generate_session_metadata(capture_meta)

        expected_keys = {
            "ip_risk_score",
            "emulator_detected",
            "tor_detected",
            "vpn_detected",
            "repeated_attempts",
            "liveness_passed",
            "device_fingerprint_score",
        }
        assert set(result.keys()) == expected_keys

    def test_liveness_passed_true(self):
        capture_meta = {
            "user_agent": "Mozilla/5.0",
            "screen_width": 1080,
            "screen_height": 1920,
            "platform": "android",
            "ip_address": "192.168.1.1",
            "liveness_passed": True,
            "accelerometer_data": [0.1, 0.2, 0.3],
        }
        result = generate_session_metadata(capture_meta)
        assert result["liveness_passed"] == 1

    def test_liveness_passed_false(self):
        capture_meta = {
            "user_agent": "Mozilla/5.0",
            "screen_width": 1080,
            "screen_height": 1920,
            "platform": "android",
            "ip_address": "192.168.1.1",
            "liveness_passed": False,
            "accelerometer_data": [0.1, 0.2, 0.3],
        }
        result = generate_session_metadata(capture_meta)
        assert result["liveness_passed"] == 0


class TestComputeIPRisk:
    def test_empty_ip(self):
        assert _compute_ip_risk("") == 0.5

    def test_private_ip(self):
        assert _compute_ip_risk("192.168.1.1") == 0.3
        assert _compute_ip_risk("10.0.0.1") == 0.3
        assert _compute_ip_risk("127.0.0.1") == 0.3

    def test_public_ip(self):
        assert _compute_ip_risk("8.8.8.8") == 0.1


class TestDetectEmulator:
    def test_no_emulator(self):
        meta = {
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0)",
            "accelerometer_data": [0.1, 0.2, 0.3, 0.15, 0.25],
        }
        assert _detect_emulator(meta) == 0

    def test_emulator_in_ua(self):
        meta = {
            "user_agent": "Mozilla/5.0 (Android Emulator)",
            "accelerometer_data": [],
        }
        assert _detect_emulator(meta) == 1

    def test_no_accelerometer(self):
        meta = {
            "user_agent": "Mozilla/5.0",
            "accelerometer_data": [],
        }
        assert _detect_emulator(meta) == 1

    def test_static_accelerometer(self):
        meta = {
            "user_agent": "Mozilla/5.0",
            "accelerometer_data": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
        assert _detect_emulator(meta) == 1


class TestDetectTor:
    def test_no_tor(self):
        assert _detect_tor("Mozilla/5.0") == 0

    def test_tor_browser(self):
        assert _detect_tor("Mozilla/5.0 (Tor Browser)") == 1


class TestDetectVPN:
    def test_no_vpn(self):
        assert _detect_vpn("Mozilla/5.0") == 0

    def test_vpn_in_ua(self):
        assert _detect_vpn("Mozilla/5.0 (NordVPN)") == 1


class TestCountRepeatedAttempts:
    def test_no_history(self):
        assert _count_repeated_attempts(None) == 0

    def test_empty_history(self):
        assert _count_repeated_attempts([]) == 0

    def test_with_history(self):
        history = [{"timestamp": "2024-01-01"}, {"timestamp": "2024-01-02"}]
        assert _count_repeated_attempts(history) == 2


class TestEvaluateLiveness:
    def test_liveness_true(self):
        meta = {"liveness_passed": True}
        assert _evaluate_liveness(meta) == 1

    def test_liveness_false(self):
        meta = {"liveness_passed": False}
        assert _evaluate_liveness(meta) == 0

    def test_liveness_missing(self):
        meta = {}
        assert _evaluate_liveness(meta) == 0


class TestComputeDeviceFingerprint:
    def test_mobile_platform(self):
        meta = {
            "screen_width": 1080,
            "screen_height": 1920,
            "platform": "android",
        }
        score = _compute_device_fingerprint(meta)
        assert score > 0.5

    def test_no_platform(self):
        meta = {
            "screen_width": 0,
            "screen_height": 0,
            "platform": "",
        }
        score = _compute_device_fingerprint(meta)
        assert score == 0.5

    def test_invalid_resolution(self):
        meta = {
            "screen_width": 10000,
            "screen_height": 10000,
            "platform": "unknown",
        }
        score = _compute_device_fingerprint(meta)
        assert score <= 0.5
