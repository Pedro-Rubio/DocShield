"""
Aplicación FastAPI principal de DocShield.

Endpoints:
- POST /api/v1/verify-document: Verifica un documento
- GET /health: Health check
"""

import base64
import io
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api import anti_spoofing_api
from src.api.schemas import (
    HealthResponse,
    VerifyDocumentRequest,
    VerifyDocumentResponse,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiting básico (en memoria)
_rate_limit_store = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # segundos
RATE_LIMIT_MAX_REQUESTS = 30  # requests por ventana


def check_rate_limit(client_ip: str) -> bool:
    """
    Verifica si un cliente excedió el rate limit.

    Args:
        client_ip: IP del cliente.

    Returns:
        True si el cliente está dentro del límite.
    """
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)

    # Limpiar requests antiguos
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]

    # Verificar límite
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    _rate_limit_store[client_ip].append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle de la aplicación."""
    logger.info("Iniciando DocShield API...")
    # Cargar modelo al inicio
    anti_spoofing_api.get_model()
    logger.info("Modelo cargado")
    yield
    logger.info("Cerrando DocShield API...")


app = FastAPI(
    title="DocShield API",
    description="Sistema de detección de fraude documental para LATAM",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio."""
    model = anti_spoofing_api.get_model()
    model_version = "0.1.0" if model is not None else "heuristic-only"

    return HealthResponse(
        status="ok",
        model_version=model_version,
    )


@app.post("/api/v1/verify-document", response_model=VerifyDocumentResponse)
async def verify_document(request: VerifyDocumentRequest):
    """
    Verifica un documento de identidad.

    La imagen se procesa en memoria y NUNCA se guarda en disco.
    """
    start_time = time.time()

    # Rate limiting
    client_ip = request.capture_meta.ip_address
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Demasiadas solicitudes. Esperá unos segundos.",
        )

    try:
        # Decodificar base64
        bgr = anti_spoofing_api.decode_base64_image(request.image)
        logger.info(f"Imagen decodificada: {bgr.shape}")

        # Extraer features
        features = anti_spoofing_api.extract_all_features(
            bgr,
            request.capture_meta.model_dump(),
        )

        # Liberar memoria de la imagen EXPLÍCITAMENTE antes del scoring
        del bgr

        # Predecir con el modelo
        fraud_score, confidence, signals = anti_spoofing_api.predict_with_model(features)

        # Determinar si es fraude
        threshold = float(
            getattr(app, "fraud_threshold", anti_spoofing_api.FRAUD_THRESHOLD)
        )
        is_fraud = fraud_score > threshold

    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

    processing_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Verificación completada: score={fraud_score:.1f}, "
        f"is_fraud={is_fraud}, signals={len(signals)}, "
        f"processing_ms={processing_ms}"
    )

    return VerifyDocumentResponse(
        fraud_score=round(fraud_score, 2),
        is_fraud=is_fraud,
        signals=signals,
        confidence=round(confidence, 4),
        processing_ms=processing_ms,
    )
