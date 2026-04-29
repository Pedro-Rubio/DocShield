"""Tests para la API FastAPI."""

import base64

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_check(client):
    """Test del endpoint /health."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


async def test_verify_document_invalid_base64(client):
    """Test de verificación con base64 inválido."""
    payload = {
        "image": "not_valid_base64!!",
        "capture_meta": {
            "user_agent": "test",
            "screen_width": 1080,
            "screen_height": 1920,
            "platform": "android",
            "ip_address": "192.168.1.1",
            "liveness_passed": True,
            "accelerometer_data": [],
        },
    }
    response = await client.post("/api/v1/verify-document", json=payload)
    # Debería fallar al decodificar base64
    assert response.status_code in (400, 422, 500)


async def test_verify_document_missing_fields(client):
    """Test de verificación con campos faltantes."""
    payload = {}
    response = await client.post("/api/v1/verify-document", json=payload)
    assert response.status_code == 422


async def test_verify_document_valid_image(client):
    """Test de verificación con imagen base64 válida."""
    # Crear imagen de prueba en memoria
    img = np.ones((200, 300, 3), dtype=np.uint8) * 200
    _, buffer = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    payload = {
        "image": img_base64,
        "capture_meta": {
            "user_agent": "Mozilla/5.0",
            "screen_width": 1080,
            "screen_height": 1920,
            "platform": "android",
            "ip_address": "192.168.1.1",
            "liveness_passed": True,
            "accelerometer_data": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
        },
    }
    response = await client.post("/api/v1/verify-document", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "is_fraud" in data
    assert "signals" in data
    assert "confidence" in data
    assert "processing_ms" in data
    assert isinstance(data["fraud_score"], float)
    assert isinstance(data["is_fraud"], bool)
    assert isinstance(data["signals"], list)
