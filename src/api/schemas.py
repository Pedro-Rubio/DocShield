"""
Modelos Pydantic para la API de DocShield.

Define los schemas de request y response para los endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class CaptureMeta(BaseModel):
    """Metadatos de captura del documento."""

    user_agent: str = Field(..., description="User-Agent del dispositivo")
    screen_width: int = Field(..., ge=100, le=4096, description="Ancho de pantalla")
    screen_height: int = Field(..., ge=100, le=4096, description="Alto de pantalla")
    platform: str = Field(..., description="Plataforma (ios, android, web)")
    ip_address: str = Field(..., description="Dirección IP del usuario")
    liveness_passed: bool = Field(..., description="Si pasó la prueba de liveness")
    accelerometer_data: list[float] = Field(
        default_factory=list,
        description="Datos del acelerómetro",
    )


class VerifyDocumentRequest(BaseModel):
    """Request para el endpoint de verificación."""

    image: str = Field(..., description="Imagen en base64")
    capture_meta: CaptureMeta = Field(..., description="Metadatos de captura")


class VerifyDocumentResponse(BaseModel):
    """Response del endpoint de verificación."""

    fraud_score: float = Field(..., description="Score de fraude (0-100)")
    is_fraud: bool = Field(..., description="Si se detectó fraude")
    signals: list[str] = Field(..., description="Señales detectadas (en español)")
    confidence: float = Field(..., description="Confianza del modelo (0-1)")
    processing_ms: int = Field(..., description="Tiempo de procesamiento en ms")


class HealthResponse(BaseModel):
    """Response del health check."""

    model_config = {"protected_namespaces": ()}

    status: str = Field(..., description="Estado del servicio")
    model_version: str = Field(..., description="Versión del modelo")
