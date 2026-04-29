"""
Módulo de extracción de features visuales de documentos de identidad.

Utiliza OpenCV para extraer características forenses de la imagen
que ayudan a detectar manipulación o fraude documental.
"""

import cv2
import numpy as np


def extract_visual_features(image_path: str) -> dict:
    """
    Extrae features visuales de una imagen de documento.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Diccionario con las siguientes keys:
        - blur_score: Varianza del Laplaciano (mayor = más nítido)
        - edge_density: Proporción de píxeles de borde
        - brightness: Media del canal gris (0-255)
        - contrast: Desviación estándar del canal gris
        - noise_ratio: Diferencia con GaussianBlur
        - symmetry_score: Correlación entre mitad izquierda y derecha
        - color_variance: Varianza entre canales RGB
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Blur score: varianza del Laplaciano
    blur_score = _laplacian_variance(gray)

    # Edge density: proporción de bordes detectados
    edge_density = _edge_density(gray)

    # Brightness y contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Noise ratio: diferencia con la versión suavizada
    noise_ratio = _noise_ratio(gray)

    # Symmetry score: correlación izquierda/derecha
    symmetry_score = _symmetry_score(gray)

    # Color variance: varianza entre canales RGB
    color_variance = _color_variance(bgr)

    # Liberar memoria de la imagen inmediatamente
    del bgr, gray

    return {
        "blur_score": blur_score,
        "edge_density": edge_density,
        "brightness": brightness,
        "contrast": contrast,
        "noise_ratio": noise_ratio,
        "symmetry_score": symmetry_score,
        "color_variance": color_variance,
    }


def compute_ela(image_path: str, quality: int = 90) -> float:
    """
    Error Level Analysis (ELA) para detectar recompresión JPEG.

    Args:
        image_path: Ruta al archivo de imagen.
        quality: Calidad de recompresión JPEG (0-100).

    Returns:
        Score ELA (mayor = más probable manipulación).
    """
    from PIL import Image

    original = Image.open(image_path)
    original = original.convert("RGB")

    # Recomprimir
    recompressed_path = image_path + ".ela_tmp.jpg"
    original.save(recompressed_path, "JPEG", quality=quality)
    recompressed = Image.open(recompressed_path)

    # Calcular diferencia máxima
    import os

    img_diff = Image.fromarray(
        np.abs(
            np.array(original, dtype=np.int16)
            - np.array(recompressed, dtype=np.int16)
        ).astype(np.uint8)
    )
    img_diff_gray = np.array(img_diff.convert("L"))
    ela_score = float(np.max(img_diff_gray))

    # Limpiar archivo temporal
    os.remove(recompressed_path)
    del original, recompressed, img_diff, img_diff_gray

    return ela_score


def _laplacian_variance(gray: np.ndarray) -> float:
    """Calcula la varianza del Laplaciano como medida de nitidez."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _edge_density(gray: np.ndarray) -> float:
    """Calcula la proporción de píxeles que son bordes (Canny)."""
    edges = cv2.Canny(gray, 50, 150)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_pixels = np.count_nonzero(edges)
    return float(edge_pixels / total_pixels)


def _noise_ratio(gray: np.ndarray) -> float:
    """Calcula el ratio de ruido comparando con una versión suavizada."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blurred)
    return float(np.mean(diff))


def _symmetry_score(gray: np.ndarray) -> float:
    """Calcula la correlación entre la mitad izquierda y derecha."""
    height, width = gray.shape
    mid = width // 2
    left = gray[:, :mid]
    right = np.fliplr(gray[:, mid : mid * 2])
    if left.size == 0 or right.size == 0:
        return 0.0
    min_width = min(left.shape[1], right.shape[1])
    left = left[:, :min_width]
    right = right[:, :min_width]
    correlation = np.corrcoef(left.flatten(), right.flatten())
    return float(correlation[0, 1])


def _color_variance(bgr: np.ndarray) -> float:
    """Calcula la varianza entre los canales RGB."""
    b, g, r = cv2.split(bgr.astype(np.float64))
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    channel_means = np.array([mean_b, mean_g, mean_r])
    return float(np.var(channel_means))
