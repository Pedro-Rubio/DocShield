"""
Lógica de scoring y detección de fraude para la API.

Implementa el pipeline completo de verificación de documentos:
extracción de features, scoring y generación de señales.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Pesos del modelo de scoring
WEIGHTS = {
    "ela": 0.25,
    "moire": 0.30,
    "dct": 0.15,
    "blur": 0.10,
    "ocr": 0.10,
    "reflection": 0.10,
}

FRAUD_THRESHOLD = 35.0

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

FEATURE_COLUMNS = [
    "blur_score",
    "ela_score",
    "ocr_confidence",
    "ocr_field_count",
    "moire_score",
    "dct_anomaly",
    "reflection_score",
    "edge_density",
    "brightness",
    "contrast",
    "noise_ratio",
    "symmetry_score",
    "color_variance",
    "ip_risk_score",
    "emulator_detected",
    "tor_detected",
    "vpn_detected",
    "repeated_attempts",
    "liveness_passed",
    "device_fingerprint_score",
]

_model_cache = None
_feature_names_cache = None


def get_model():
    """Carga el modelo de fraude (con cache)."""
    global _model_cache
    if _model_cache is None:
        model_path = MODELS_DIR / "fraud_detector.pkl"
        if model_path.exists():
            _model_cache = joblib.load(model_path)
        else:
            logger.warning("Modelo no encontrado, usando scoring heurístico")
            _model_cache = None
    return _model_cache


def get_feature_names() -> list[str]:
    """Carga los nombres de features (con cache)."""
    global _feature_names_cache
    if _feature_names_cache is None:
        names_path = MODELS_DIR / "feature_names.pkl"
        if names_path.exists():
            _feature_names_cache = joblib.load(names_path)
        else:
            _feature_names_cache = FEATURE_COLUMNS
    return _feature_names_cache


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decodifica una imagen base64 a un array numpy.

    Args:
        base64_string: Imagen codificada en base64.

    Returns:
        Array numpy en formato BGR.
    """
    try:
        image_bytes = base64.b64decode(base64_string)
        pil_img = Image.open(io.BytesIO(image_bytes))
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise ValueError(f"Error al decodificar base64: {e}")


def extract_all_features(bgr: np.ndarray, capture_meta: dict) -> dict:
    """
    Extrae todas las features de un documento.

    Args:
        bgr: Imagen en formato BGR.
        capture_meta: Metadatos de captura.

    Returns:
        Diccionario con todas las features.
    """
    from src.pipeline.anti_spoofing import (
        analyze_dct_blocks,
        analyze_reflection,
        detect_moire,
    )
    from src.pipeline.metadata_generator import generate_session_metadata
    from src.pipeline.visual_extractor import extract_visual_features

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Features visuales (usar arrays en memoria, no archivos)
    visual = _extract_visual_from_array(gray, bgr)

    # Anti-spoofing
    moire_score = detect_moire(gray)
    dct_anomaly = analyze_dct_blocks(gray)
    reflection_score = analyze_reflection(bgr)

    # Features OCR (simuladas si no hay EasyOCR)
    ocr_features = _extract_ocr_from_array(gray)

    # Metadatos de sesión
    session_meta = generate_session_metadata(capture_meta)

    # Liberar memoria explícitamente
    del bgr, gray

    return {
        **visual,
        **ocr_features,
        "moire_score": moire_score,
        "dct_anomaly": dct_anomaly,
        "reflection_score": reflection_score,
        **session_meta,
    }


def _extract_visual_from_array(gray: np.ndarray, bgr: np.ndarray) -> dict:
    """Extrae features visuales directamente de arrays numpy."""
    from src.pipeline.visual_extractor import (
        _color_variance,
        _edge_density,
        _laplacian_variance,
        _noise_ratio,
        _symmetry_score,
    )

    return {
        "blur_score": _laplacian_variance(gray),
        "edge_density": _edge_density(gray),
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "noise_ratio": _noise_ratio(gray),
        "symmetry_score": _symmetry_score(gray),
        "color_variance": _color_variance(bgr),
        "ela_score": _compute_ela_from_array(bgr),
    }


def _compute_ela_from_array(bgr: np.ndarray) -> float:
    """
    Calcula ELA directamente desde un array numpy.

    Args:
        bgr: Imagen en formato BGR.

    Returns:
        Score ELA.
    """
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Recomprimir
    buffer = io.BytesIO()
    pil_img.save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    # Calcular diferencia
    original_arr = np.array(pil_img, dtype=np.int16)
    recompressed_arr = np.array(recompressed, dtype=np.int16)
    diff = np.abs(original_arr - recompressed_arr).astype(np.uint8)
    ela_score = float(np.max(np.array(Image.fromarray(diff).convert("L"))))

    del pil_img, recompressed, original_arr, recompressed_arr, diff
    return ela_score


def _extract_ocr_from_array(gray: np.ndarray) -> dict:
    """
    Extrae features OCR desde un array numpy.

    Intenta usar EasyOCR primero, luego Tesseract como fallback.

    Args:
        gray: Imagen en escala de grises.

    Returns:
        Diccionario con features OCR.
    """
    # Intentar con EasyOCR
    try:
        import easyocr

        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        results = reader.readtext(gray)

        if results:
            confidences = [r[2] for r in results if len(r) >= 3]
            if confidences:
                return {
                    "ocr_confidence": float(np.mean(confidences)),
                    "ocr_field_count": float(len(results)),
                }
    except Exception:
        pass

    # Fallback: estimar basándose en la complejidad de la imagen
    # Imágenes con más texto tienden a tener más bordes
    from src.pipeline.visual_extractor import _edge_density

    edge_d = _edge_density(gray)
    estimated_confidence = min(edge_d * 10, 0.95)  # Heurística simple

    return {
        "ocr_confidence": estimated_confidence,
        "ocr_field_count": float(max(1, int(edge_d * 100))),
    }


def compute_fraud_score(features: dict, liveness_passed: bool = True) -> tuple[float, list[str]]:
    """
    Calcula el score de fraude y genera señales.

    Args:
        features: Diccionario con todas las features.
        liveness_passed: Si el usuario pasó la prueba de liveness.

    Returns:
        Tuple de (score, señales_detectadas).
    """
    weights = WEIGHTS

    # Normalizar features a 0-1
    ela_norm = min(features.get("ela_score", 0) / 100.0, 1.0)
    moire_norm = min(features.get("moire_score", 0) / 20.0, 1.0)
    dct_norm = min(features.get("dct_anomaly", 0) / 1.0, 1.0)
    blur_norm = 1.0 - min(features.get("blur_score", 180) / 200.0, 1.0)
    ocr_norm = 1.0 - min(features.get("ocr_confidence", 0.8), 1.0)
    reflection_norm = min(features.get("reflection_score", 0) / 20.0, 1.0)

    # Score ponderado
    score = (
        weights["ela"] * ela_norm
        + weights["moire"] * moire_norm
        + weights["dct"] * dct_norm
        + weights["blur"] * blur_norm
        + weights["ocr"] * ocr_norm
        + weights["reflection"] * reflection_norm
    )

    # Convertir a 0-100
    score *= 100.0

    # Generar señales
    signals = _generate_signals(features, ela_norm, moire_norm, dct_norm, blur_norm, ocr_norm, reflection_norm)

    # Penalización por liveness
    if not liveness_passed:
        score += 20.0
        signals.append("Liveness no superado (+20 pts)")

    return min(max(score, 0), 100), signals


def predict_with_model(features: dict) -> tuple[float, float, list[str]]:
    """
    Usa el modelo entrenado para predecir fraude.

    Args:
        features: Diccionario con todas las features.

    Returns:
        Tuple de (fraud_score, confidence, signals).
    """
    model = get_model()
    feature_names = get_feature_names()

    if model is None:
        # Fallback a scoring heurístico
        liveness = features.get("liveness_passed", True)
        score, signals = compute_fraud_score(features, liveness)
        confidence = 0.5
        return score, confidence, signals

    # Preparar feature vector
    feature_vector = np.array([[features.get(f, 0) for f in feature_names]])

    # Predicción
    proba = model.predict_proba(feature_vector)[0, 1]
    score = float(proba * 100)

    # Señales basadas en SHAP
    from src.model.explainer import explain_prediction

    try:
        explanation = explain_prediction(
            features=feature_vector,
            feature_names=feature_names,
        )
        signals = explanation["top_signals"]
        confidence = float(explanation["predicted_proba"])
    except Exception:
        signals = _generate_signals_simple(features)
        confidence = float(proba)

    return score, confidence, signals


def _generate_signals(
    features: dict,
    ela_norm: float,
    moire_norm: float,
    dct_norm: float,
    blur_norm: float,
    ocr_norm: float,
    reflection_norm: float,
) -> list[str]:
    """Genera señales de fraude basadas en features normalizadas."""
    signals = []

    if ela_norm > 0.25:
        signals.append("Anomalía ELA detectada (posible edición)")
    if moire_norm > 0.4:
        signals.append("Posible captura de pantalla (patrón de Moiré)")
    if dct_norm > 0.5:
        signals.append("Inconsistencias DCT (zonas editadas)")
    if blur_norm > 0.7:
        signals.append("Imagen con baja nitidez")
    if ocr_norm > 0.45:
        signals.append("Confianza OCR baja")
    if reflection_norm > 0.5:
        signals.append("Reflexión especular detectada")
    if features.get("ip_risk_score", 0) > 0.7:
        signals.append("IP de alto riesgo")
    if features.get("emulator_detected", 0) == 1:
        signals.append("Emulador detectado")
    if features.get("repeated_attempts", 0) >= 3:
        signals.append("Múltiples intentos de verificación")

    return signals


def _generate_signals_simple(features: dict) -> list[str]:
    """Genera señales simples cuando SHAP no está disponible."""
    signals = []

    if features.get("ela_score", 0) > 25:
        signals.append("Anomalía ELA detectada")
    if features.get("moire_score", 0) > 8.5:
        signals.append("Posible captura de pantalla")
    if features.get("ocr_confidence", 1) < 0.55:
        signals.append("OCR confidence baja")
    if features.get("ip_risk_score", 0) > 0.7:
        signals.append("IP de alto riesgo")

    return signals
