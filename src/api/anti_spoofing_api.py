import base64
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Tuple
from PIL import Image
import io

from src.pipeline.visual_extractor import extract_visual_features, compute_ela
from src.pipeline.anti_spoofing import detect_moire, analyze_dct_blocks, analyze_reflection
from src.pipeline.ocr_extractor import extract_ocr_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS = {
    "ela": 0.25,
    "moire": 0.30,
    "dct": 0.15,
    "blur": 0.10,
    "ocr": 0.10,
    "reflection": 0.10,
}

FRAUD_THRESHOLD = 35.0

def decode_base64_image(base64_string: str) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    Decodifica una imagen en base64 a arrays en memoria.

    Args:
        base64_string: String de imagen codificada en base64.

    Returns:
        Tupla de (bgr array, gray array, PIL Image).
    """
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        pil_img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        bgr = np.array(pil_img)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        return bgr, gray, pil_img
    except Exception as e:
        logger.error(f"Error decodificando imagen base64: {e}")
        raise ValueError(f"Imagen base64 inválida: {e}")

def extract_all_features(bgr: np.ndarray, gray: np.ndarray, pil_img: Image.Image) -> Dict[str, float]:
    """
    Extrae todas las features del pipeline para una imagen.

    Args:
        bgr: Imagen en formato BGR.
        gray: Imagen en escala de grises.
        pil_img: Imagen en formato PIL.

    Returns:
        Diccionario con todas las features extraídas.
    """
    features = {}
    
    temp_path = "temp_docshield_verify.jpg"
    try:
        cv2.imwrite(temp_path, bgr)
        
        visual_feats = extract_visual_features(temp_path)
        features.update(visual_feats)
        
        ela_score = compute_ela(temp_path)
        features["ela_score"] = ela_score
        
        ocr_feats = extract_ocr_features(temp_path)
        features.update(ocr_feats)
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    features["moire_score"] = detect_moire(gray)
    features["dct_score"] = analyze_dct_blocks(gray)
    features["reflection_score"] = analyze_reflection(bgr)
    
    return features

def calculate_fraud_score(features: Dict[str, float], capture_meta: Dict = None) -> Tuple[float, List[str]]:
    """
    Calcula el score de fraude ponderado con las features extraídas.

    Args:
        features: Diccionario de features extraídas.
        capture_meta: Metadatos de captura (opcional).

    Returns:
        Tupla de (fraud_score 0-100, lista de señales detectadas).
    """
    signals = []
    score = 0.0
    
    ela_norm = min(features.get("ela_score", 0) / 50.0, 1.0)
    score += WEIGHTS["ela"] * 100 * ela_norm
    if ela_norm > 0.5:
        signals.append(f"ELA anomalía alta ({features.get('ela_score', 0):.1f})")
    
    moire_norm = min(features.get("moire_score", 0) / 15.0, 1.0)
    score += WEIGHTS["moire"] * 100 * moire_norm
    if moire_norm > 0.5:
        signals.append(f"Posible screen capture (Moiré: {features.get('moire_score', 0):.1f})")
    
    dct_norm = min(features.get("dct_score", 0) / 2.0, 1.0)
    score += WEIGHTS["dct"] * 100 * dct_norm
    
    blur_norm = 1.0 - min(features.get("blur_score", 0) / 300.0, 1.0)
    score += WEIGHTS["blur"] * 100 * blur_norm
    if blur_norm > 0.5:
        signals.append(f"Blur bajo ({features.get('blur_score', 0):.1f})")
    
    ocr_norm = 1.0 - features.get("ocr_confidence", 0)
    score += WEIGHTS["ocr"] * 100 * ocr_norm
    if ocr_norm > 0.45:
        signals.append(f"OCR confidence baja ({features.get('ocr_confidence', 0):.2f})")
    
    reflection_norm = min(features.get("reflection_score", 0) / 5.0, 1.0)
    score += WEIGHTS["reflection"] * 100 * reflection_norm
    
    if capture_meta:
        if not capture_meta.get("liveness_passed", False):
            score += 20
            signals.append("Liveness check fallido")
        
        if capture_meta.get("ip_risk_score", 0) > 0.7:
            score += 10
            signals.append(f"IP de alto riesgo ({capture_meta.get('ip_risk_score', 0):.2f})")
        
        if capture_meta.get("emulator_detected", 0) == 1:
            score += 5
            signals.append("Emulador detectado")
        
        if (capture_meta.get("tor_detected", 0) + capture_meta.get("vpn_detected", 0)) > 0:
            score += 8
            signals.append("Tor/VPN detectado")
        
        if capture_meta.get("repeated_attempts", 0) >= 3:
            score += 7
            signals.append(f"Múltiples intentos ({capture_meta.get('repeated_attempts', 0)})")
    
    score = min(score, 100.0)
    
    return score, signals

def verify_document(base64_image: str, capture_meta: Dict = None) -> Dict:
    """
    Pipeline completo de verificación de documento.

    Args:
        base64_image: Imagen en base64.
        capture_meta: Metadatos de captura.

    Returns:
        Diccionario con resultados de verificación.
    """
    start_time = time.time()
    
    bgr, gray, pil_img = decode_base64_image(base64_image)
    
    try:
        features = extract_all_features(bgr, gray, pil_img)
        
        del bgr, gray, pil_img
        
        fraud_score, signals = calculate_fraud_score(features, capture_meta)
        
        is_fraud = fraud_score > FRAUD_THRESHOLD
        confidence = 1.0 - (fraud_score / 100.0)
        
        processing_ms = int((time.time() - start_time) * 1000)
        
        return {
            "fraud_score": float(fraud_score),
            "is_fraud": bool(is_fraud),
            "signals": signals,
            "confidence": float(confidence),
            "processing_ms": processing_ms,
            "features": features
        }
    except Exception as e:
        logger.error(f"Error en verificación: {e}")
        raise
    finally:
        if 'bgr' in locals():
            del bgr
        if 'gray' in locals():
            del gray
        if 'pil_img' in locals():
            del pil_img
