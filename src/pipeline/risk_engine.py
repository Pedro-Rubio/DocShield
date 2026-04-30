import numpy as np
from typing import Dict, List, Tuple, Any

# Configuración por defecto (se puede sobreescribir con .env)
DEFAULT_WEIGHTS = {
    "ela": 0.25,
    "moire": 0.30,
    "dct": 0.15,
    "blur": 0.10,
    "ocr": 0.10,
    "reflection": 0.10,
}

DEFAULT_THRESHOLD = 35.0


def calculate_risk_score(features: Dict[str, float], 
                        capture_meta: Dict = None,
                        weights: Dict[str, float] = None,
                        threshold: float = None) -> Dict[str, Any]:
    """
    Motor de riesgo unificado para DocShield.
    
    Args:
        features: Diccionario con features extraídas (blur_score, ela_score, etc.)
        capture_meta: Metadatos de captura (liveness, IP, etc.)
        weights: Pesos personalizados (opcional).
        threshold: Umbral personalizado (opcional).
    
    Returns:
        Diccionario estructurado con resultado de riesgo.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    
    signals = {}
    score = 0.0
    
    # ELA
    ela_norm = min(features.get("ela_score", 0) / 50.0, 1.0)
    score += weights["ela"] * 100 * ela_norm
    if ela_norm > 0.5:
        signals["ela"] = float(ela_norm)
    
    # Moiré (mayor peso - anti-spoofing directo)
    moire_norm = min(features.get("moire_score", 0) / 15.0, 1.0)
    score += weights["moire"] * 100 * moire_norm
    if moire_norm > 0.5:
        signals["moire"] = float(moire_norm)
    
    # DCT
    dct_norm = min(features.get("dct_score", 0) / 2.0, 1.0)
    score += weights["dct"] * 100 * dct_norm
    if dct_norm > 0.5:
        signals["dct"] = float(dct_norm)
    
    # Blur
    blur_norm = 1.0 - min(features.get("blur_score", 0) / 300.0, 1.0)
    score += weights["blur"] * 100 * blur_norm
    if blur_norm > 0.5:
        signals["blur"] = float(blur_norm)
    
    # OCR
    ocr_norm = 1.0 - features.get("ocr_confidence", 0)
    score += weights["ocr"] * 100 * ocr_norm
    if ocr_norm > 0.45:
        signals["ocr"] = float(ocr_norm)
    
    # Reflection
    reflection_norm = min(features.get("reflection_score", 0) / 5.0, 1.0)
    score += weights["reflection"] * 100 * reflection_norm
    if reflection_norm > 0.5:
        signals["reflection"] = float(reflection_norm)
    
    # Metadatos de sesión
    if capture_meta:
        if not capture_meta.get("liveness_passed", False):
            score += 20
            signals["liveness"] = 1.0
        
        if capture_meta.get("ip_risk_score", 0) > 0.7:
            score += 10
            signals["ip_risk"] = float(capture_meta.get("ip_risk_score", 0))
        
        if capture_meta.get("emulator_detected", 0) == 1:
            score += 5
            signals["emulator"] = 1.0
        
        if (capture_meta.get("tor_detected", 0) + capture_meta.get("vpn_detected", 0)) > 0:
            score += 8
            signals["tor_vpn"] = 1.0
        
        if capture_meta.get("repeated_attempts", 0) >= 3:
            score += 7
            signals["repeated_attempts"] = float(capture_meta.get("repeated_attempts", 0))
    
    score = min(score, 100.0)
    
    # Determinar nivel de riesgo
    if score >= 70:
        risk_level = "HIGH"
    elif score >= threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    is_fraud = score > threshold
    confidence = 1.0 - (score / 100.0)
    
    return {
        "is_fraud": bool(is_fraud),
        "confidence": float(confidence),
        "risk_score": float(score),
        "risk_level": risk_level,
        "signals": signals,
        "threshold_used": float(threshold)
    }


def get_risk_level_description(risk_level: str) -> str:
    """Retorna descripción legible del nivel de riesgo."""
    descriptions = {
        "LOW": "Documento con pocas señales de fraude. Verificación pasada.",
        "MEDIUM": "Documento con señales moderadas de fraude. Requiere revisión manual.",
        "HIGH": "Documento con múltiples señales de fraude. Alto riesgo detectado."
    }
    return descriptions.get(risk_level, "Nivel de riesgo desconocido")
