import os
import time
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from src.api.schemas import (
    VerifyDocumentRequest, VerifyDocumentResponse, 
    HealthResponse, ErrorResponse, CaptureMetadata
)
from src.api.anti_spoofing_api import verify_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="DocShield API",
    description="Motor de detección de spoofing en onboarding digital",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración desde .env
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "35.0"))

# Cargar pesos desde .env (opcional)
def _load_weights_from_env():
    from src.pipeline.risk_engine import DEFAULT_WEIGHTS
    weights = DEFAULT_WEIGHTS.copy()
    env_mapping = {
        "WEIGHT_ELA": "ela",
        "WEIGHT_MOIRE": "moire",
        "WEIGHT_DCT": "dct",
        "WEIGHT_BLUR": "blur",
        "WEIGHT_OCR": "ocr",
        "WEIGHT_REFLECTION": "reflection"
    }
    for env_key, weight_key in env_mapping.items():
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                weights[weight_key] = float(env_val)
            except ValueError:
                logger.warning(f"Valor inválido para {env_key}: {env_val}")
    return weights

RISK_WEIGHTS = _load_weights_from_env()
logger.info(f"Pesos de riesgo cargados: {RISK_WEIGHTS}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check."""
    return HealthResponse()


@app.post("/api/v1/verify-document", response_model=VerifyDocumentResponse)
@limiter.limit("10/minute")
async def verify_document_endpoint(request: Request, body: VerifyDocumentRequest):
    """
    Endpoint para verificación de documento.
    
    Recibe imagen en base64 y metadatos de captura, devuelve score de riesgo.
    """
    start_time = time.time()
    
    try:
        capture_meta = None
        if body.capture_meta:
            capture_meta = body.capture_meta.dict()
        
        result = verify_document(body.image, capture_meta)
        
        processing_ms = int((time.time() - start_time) * 1000)
        
        return VerifyDocumentResponse(
            is_fraud=result["is_fraud"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            signals=result["signals"],
            signal_details=result.get("signal_details"),
            processing_ms=processing_ms
        )
        
    except ValueError as e:
        logger.warning(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error interno: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
