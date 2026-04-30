from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone

class CaptureMetadata(BaseModel):
    """Metadatos de la sesión de captura desde el dispositivo móvil."""
    timestamp: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    accelerometer_data: Optional[List[float]] = None
    liveness_passed: bool = False
    liveness_rotation_deg: Optional[float] = None
    ip_address: Optional[str] = None
    ip_risk_score: float = 0.0
    emulator_detected: int = 0
    tor_detected: int = 0
    vpn_detected: int = 0
    repeated_attempts: int = 0
    session_id: Optional[str] = None

class VerifyDocumentRequest(BaseModel):
    """Request para verificación de documento."""
    image: str = Field(..., description="Imagen del documento en base64")
    capture_meta: Optional[CaptureMetadata] = None

class Signal(BaseModel):
    """Señal de fraude detectada."""
    name: str
    description: str
    weight: float
    value: float

class VerifyDocumentResponse(BaseModel):
    """Response de verificación de documento."""
    is_fraud: bool = Field(..., description="Si el documento es fraudulento")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    risk_score: float = Field(..., description="Score de riesgo (0-100)")
    risk_level: str = Field(..., description="Nivel de riesgo: LOW, MEDIUM, HIGH")
    signals: List[str] = Field(default_factory=list, description="Señales detectadas")
    signal_details: Optional[Dict[str, float]] = None
    processing_ms: int = Field(..., description="Tiempo de procesamiento en ms")
    document_type: Optional[str] = None
    fraud_type: Optional[str] = None

class HealthResponse(BaseModel):
    """Response del health check."""
    status: str = "ok"
    model_version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ErrorResponse(BaseModel):
    """Response de error."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
