import pytest
from fastapi.testclient import TestClient
import base64
import numpy as np
import cv2
from src.api.main import app

client = TestClient(app)

def create_test_image_base64(width=400, height=250):
    """Crea una imagen de prueba y la codifica en base64."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (350, 200), (0, 0, 0), 2)
    cv2.putText(img, "DOCUMENTO OFICIAL", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

class TestVerifyDocumentEndpoint:
    def test_verify_document_returns_valid_response(self):
        img_base64 = create_test_image_base64()
        
        payload = {
            "image": img_base64,
            "capture_meta": {
                "liveness_passed": True,
                "ip_risk_score": 0.2,
                "emulator_detected": 0,
                "tor_detected": 0,
                "vpn_detected": 0,
                "repeated_attempts": 0
            }
        }
        
        response = client.post("/api/v1/verify-document", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "risk_score" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "signals" in data
        assert "confidence" in data
        assert "processing_ms" in data
        assert isinstance(data["risk_score"], (int, float))
        assert isinstance(data["is_fraud"], bool)
    
    def test_verify_document_without_metadata(self):
        img_base64 = create_test_image_base64()
        
        payload = {"image": img_base64}
        response = client.post("/api/v1/verify-document", json=payload)
        assert response.status_code == 200
    
    def test_verify_document_invalid_base64(self):
        payload = {"image": "invalid_base64!!!"}
        response = client.post("/api/v1/verify-document", json=payload)
        assert response.status_code == 400
