import numpy as np
from typing import Dict
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Singleton para EasyOCR
_EASYOCR_READERS = {}

def _get_easyocr_reader(langs=['es', 'en']):
    key = tuple(langs)
    if key not in _EASYOCR_READERS:
        try:
            import easyocr
            _EASYOCR_READERS[key] = easyocr.Reader(langs, gpu=False)
        except (ImportError, Exception):
            return None
    return _EASYOCR_READERS.get(key)

def extract_ocr_features(image_path: str) -> Dict[str, float]:
    """
    Extrae features de OCR (confianza de reconocimiento de texto).

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Diccionario con ocr_confidence (confianza promedio de OCR).
    """
    ocr_confidence = 0.0
    
    # Intentar con EasyOCR (singleton)
    reader = _get_easyocr_reader(['es', 'en'])
    if reader:
        try:
            result = reader.readtext(image_path)
            if result:
                confidences = [item[2] for item in result if len(item) >= 3]
                if confidences:
                    ocr_confidence = np.mean(confidences)
        except Exception:
            pass
    
    # Fallback a Tesseract
    if ocr_confidence == 0.0 and pytesseract is not None:
        try:
            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            if confidences:
                ocr_confidence = np.mean(confidences) / 100.0
        except Exception:
            pass
    
    return {"ocr_confidence": float(ocr_confidence)}
