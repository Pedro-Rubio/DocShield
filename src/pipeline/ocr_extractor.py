"""
Módulo de extracción de texto mediante OCR.

Utiliza EasyOCR y Tesseract para extraer texto de documentos
de identidad y calcular métricas de confianza que ayudan
a detectar documentos fraudulentos.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_ocr_features(
    image_path: str,
    use_easyocr: bool = True,
    use_tesseract: bool = False,
) -> dict:
    """
    Extrae features OCR de una imagen de documento.

    Args:
        image_path: Ruta al archivo de imagen.
        use_easyocr: Si usar EasyOCR como motor principal.
        use_tesseract: Si usar Tesseract como motor adicional.

    Returns:
        Diccionario con las siguientes keys:
        - ocr_confidence: Confianza promedio del OCR (0-1)
        - ocr_field_count: Número de campos de texto detectados
        - ocr_has_id_number: Si se detectó un posible número de documento
        - ocr_text_length: Longitud total del texto extraído
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    del image

    # Intentar con EasyOCR
    easyocr_result = _extract_easyocr(gray) if use_easyocr else None

    # Intentar con Tesseract
    tesseract_result = _extract_tesseract(gray) if use_tesseract else None

    del gray

    # Combinar resultados (priorizar EasyOCR)
    if easyocr_result is not None:
        result = easyocr_result
    elif tesseract_result is not None:
        result = tesseract_result
    else:
        result = _default_empty_result()

    return result


def _extract_easyocr(gray: np.ndarray) -> Optional[dict]:
    """
    Extrae texto y confianza usando EasyOCR.

    Args:
        gray: Imagen en escala de grises.

    Returns:
        Diccionario con features OCR o None si falla.
    """
    try:
        import easyocr

        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        results = reader.readtext(gray)

        if not results:
            return None

        confidences = [r[2] for r in results if len(r) >= 3]
        if not confidences:
            return None

        all_text = " ".join([r[1] for r in results])

        return {
            "ocr_confidence": float(np.mean(confidences)),
            "ocr_field_count": len(results),
            "ocr_has_id_number": _looks_like_id_number(all_text),
            "ocr_text_length": len(all_text),
        }
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
        return None


def _extract_tesseract(gray: np.ndarray) -> Optional[dict]:
    """
    Extrae texto y confianza usando Tesseract.

    Args:
        gray: Imagen en escala de grises.

    Returns:
        Diccionario con features OCR o None si falla.
    """
    try:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(gray, output_type=Output.DICT)

        # Filtrar entries con confianza > 0
        valid_confidences = [
            float(c)
            for c, t in zip(data["conf"], data["text"])
            if t.strip() and float(c) > 0
        ]

        if not valid_confidences:
            return None

        avg_confidence = np.mean(valid_confidences) / 100.0  # Normalizar a 0-1
        field_count = len([t for t in data["text"] if t.strip()])
        all_text = " ".join(data["text"])

        return {
            "ocr_confidence": float(avg_confidence),
            "ocr_field_count": field_count,
            "ocr_has_id_number": _looks_like_id_number(all_text),
            "ocr_text_length": len(all_text),
        }
    except Exception as e:
        logger.warning(f"Tesseract failed: {e}")
        return None


def _looks_like_id_number(text: str) -> bool:
    """
    Verifica si el texto contiene un posible número de documento.

    Busca patrones comunes de números de documento latinoamericanos:
    - 7-9 dígitos consecutivos
    - Patrones con puntos o guiones (XX.XXX.XXX)

    Args:
        text: Texto extraído por OCR.

    Returns:
        True si parece contener un número de documento.
    """
    import re

    # Patrón: 7-9 dígitos consecutivos
    pattern_consecutive = r"\b\d{7,9}\b"
    # Patrón: con puntos o guiones (ej: 12.345.678 o 12-3456789)
    pattern_formatted = r"\b\d{1,3}[.\-]\d{2,3}[.\-]\d{2,4}\b"

    return bool(re.search(pattern_consecutive, text) or re.search(pattern_formatted, text))


def _default_empty_result() -> dict:
    """Retorna un resultado vacío con valores por defecto."""
    return {
        "ocr_confidence": 0.0,
        "ocr_field_count": 0,
        "ocr_has_id_number": False,
        "ocr_text_length": 0,
    }
