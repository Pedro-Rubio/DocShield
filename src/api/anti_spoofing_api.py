import base64
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Tuple
from PIL import Image
import io

from src.pipeline.anti_spoofing import detect_moire, analyze_dct_blocks, analyze_reflection
from src.pipeline.risk_engine import calculate_risk_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
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
        raise ValueError(f"Imagen base64 invalida: {e}")


def extract_all_features(bgr: np.ndarray, gray: np.ndarray, pil_img: Image.Image) -> Dict[str, float]:
    features = {}
    visual_feats = _extract_visual_features_from_array(bgr)
    features.update(visual_feats)
    ela_score = _compute_ela_from_pil(pil_img)
    features["ela_score"] = ela_score
    ocr_feats = _extract_ocr_from_pil(pil_img)
    features.update(ocr_feats)
    features["moire_score"] = detect_moire(gray)
    features["dct_score"] = analyze_dct_blocks(gray)
    features["reflection_score"] = analyze_reflection(bgr)
    return features


def _extract_visual_features_from_array(bgr: np.ndarray) -> Dict[str, float]:
    gray_local = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray_local.shape
    laplacian = cv2.Laplacian(gray_local, cv2.CV_64F)
    blur_score = laplacian.var()
    edges = cv2.Canny(gray_local, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / (h * w) if (h * w) > 0 else 0.0
    brightness = float(gray_local.mean())
    contrast = float(gray_local.std())
    blurred = cv2.GaussianBlur(gray_local, (5, 5), 0)
    noise = cv2.absdiff(gray_local, blurred)
    noise_ratio = float(noise.mean())
    left = gray_local[:, :w//2]
    right = gray_local[:, w//2:]
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        left = left[:, :min_w]
        right = right[:, :min_w]
        left_flat = left.flatten()
        right_flat = right.flatten()
        if len(left_flat) >= 2 and len(right_flat) >= 2:
            corr = np.corrcoef(left_flat, right_flat)[0, 1]
            symmetry_score = float(corr) if not np.isnan(corr) else 0.0
        else:
            symmetry_score = 0.0
    else:
        symmetry_score = 0.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)
    color_variance = float(np.mean([r.var(), g.var(), b.var()]))
    return {
        "blur_score": float(blur_score),
        "edge_density": edge_density,
        "brightness": brightness,
        "contrast": contrast,
        "noise_ratio": noise_ratio,
        "symmetry_score": symmetry_score,
        "color_variance": color_variance
    }


def _compute_ela_from_pil(pil_img: Image.Image, quality: int = 90) -> float:
    buffer1 = io.BytesIO()
    pil_img.save(buffer1, format='JPEG', quality=quality)
    buffer1.seek(0)
    compressed_img = Image.open(buffer1).convert('RGB')
    original = np.array(pil_img, dtype=np.int16)
    compressed = np.array(compressed_img, dtype=np.int16)
    diff = np.abs(original - compressed)
    return float(diff.max())


def _extract_ocr_from_pil(pil_img: Image.Image) -> Dict[str, float]:
    ocr_confidence = 0.0
    try:
        import easyocr
        reader = easyocr.Reader(['es', 'en'], gpu=False)
        result = reader.readtext(np.array(pil_img))
        if result:
            confidences = [item[2] for item in result if len(item) >= 3]
            if confidences:
                ocr_confidence = np.mean(confidences)
    except Exception:
        pass
    return {"ocr_confidence": float(ocr_confidence)}


def calculate_fraud_score(features: Dict[str, float], capture_meta: Dict = None) -> Tuple[float, List[str]]:
    signals = []
    score = 0.0
    ela_norm = min(features.get("ela_score", 0) / 50.0, 1.0)
    score += WEIGHTS["ela"] * 100 * ela_norm
    if ela_norm > 0.5:
        signals.append(f"ELA anomalia alta ({features.get('ela_score', 0):.1f})")
    moire_norm = min(features.get("moire_score", 0) / 15.0, 1.0)
    score += WEIGHTS["moire"] * 100 * moire_norm
    if moire_norm > 0.5:
        signals.append(f"Posible screen capture (Moire: {features.get('moire_score', 0):.1f})")
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
            signals.append(f"Multiples intentos ({capture_meta.get('repeated_attempts', 0)})")
    score = min(score, 100.0)
    return score, signals


def verify_document(base64_image: str, capture_meta: Dict = None) -> Dict:
    start_time = time.time()
    bgr, gray, pil_img = decode_base64_image(base64_image)
    try:
        features = extract_all_features(bgr, gray, pil_img)
        del bgr, gray, pil_img

        # Usar el nuevo motor de riesgo
        risk_result = calculate_risk_score(features, capture_meta)

        processing_ms = int((time.time() - start_time) * 1000)

        # Convertir señales dict a lista de strings para compatibilidad
        signals_list = [f"{k}: {v:.2f}" for k, v in risk_result["signals"].items()]

        return {
            "is_fraud": risk_result["is_fraud"],
            "confidence": risk_result["confidence"],
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "signals": signals_list,
            "signal_details": risk_result["signals"],
            "processing_ms": processing_ms,
            "features": features
        }
    except Exception as e:
        logger.error(f"Error en verificacion: {e}")
        raise
    finally:
        if 'bgr' in locals():
            del bgr
        if 'gray' in locals():
            del gray
        if 'pil_img' in locals():
            del pil_img
