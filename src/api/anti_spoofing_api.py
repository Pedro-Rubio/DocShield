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

# Limpieza: Eliminar lógica muerta tras migrar a risk_engine
# (calculate_fraud_score y WEIGHTS ya no se usan aquí)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str, max_size_mb: float = 5.0) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Validar tamaño para prevenir DoS
        estimated_bytes = len(base64_string) * 3 / 4
        if estimated_bytes > max_size_mb * 1024 * 1024:
            raise ValueError(f"Imagen excede el tamaño máximo de {max_size_mb}MB")
        
        image_data = base64.b64decode(base64_string)
        if len(image_data) > max_size_mb * 1024 * 1024:
            raise ValueError(f"Imagen excede el tamaño máximo de {max_size_mb}MB")
            
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


# Singleton para EasyOCR (evita reinicialización en cada request)
_EASYOCR_READERS = {}

def _get_ocr_reader(langs=['es', 'en']):
    key = tuple(langs)
    if key not in _EASYOCR_READERS:
        try:
            import easyocr
            _EASYOCR_READERS[key] = easyocr.Reader(langs, gpu=False)
        except ImportError:
            return None
    return _EASYOCR_READERS.get(key)

def _extract_ocr_from_pil(pil_img: Image.Image) -> Dict[str, float]:
    ocr_confidence = 0.0
    reader = _get_ocr_reader(['es', 'en'])
    if reader:
        try:
            result = reader.readtext(np.array(pil_img))
            if result:
                confidences = [item[2] for item in result if len(item) >= 3]
                if confidences:
                    ocr_confidence = np.mean(confidences)
        except Exception:
            pass
    return {"ocr_confidence": float(ocr_confidence)}


def verify_document(base64_image: str, capture_meta: Dict = None) -> Dict:
    start_time = time.time()
    bgr, gray, pil_img = decode_base64_image(base64_image)
    try:
        features = extract_all_features(bgr, gray, pil_img)
        del bgr, gray, pil_img

        risk_result = calculate_risk_score(features, capture_meta)

        processing_ms = int((time.time() - start_time) * 1000)

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
