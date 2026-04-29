import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict

def extract_visual_features(image_path: str) -> Dict[str, float]:
    """
    Extrae features visuales forenses de un documento de identidad.

    Args:
        image_path: Ruta al archivo de imagen del documento.

    Returns:
        Diccionario con las features extraídas: blur_score, edge_density,
        brightness, contrast, noise_ratio, symmetry_score, color_variance.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"No se pudo leer la imagen en {image_path}")
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / (h * w) if (h * w) > 0 else 0.0
    
    brightness = float(gray.mean())
    contrast = float(gray.std())
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blurred)
    noise_ratio = float(noise.mean())
    
    left = gray[:, :w//2]
    right = gray[:, w//2:]
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

def compute_ela(image_path: str, quality: int = 90) -> float:
    """
    Realiza Error Level Analysis (ELA) sobre una imagen.

    Args:
        image_path: Ruta al archivo de imagen.
        quality: Calidad de recompresión JPEG (por defecto 90).

    Returns:
        Diferencia máxima entre la imagen original y la recomprimida.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error al abrir la imagen {image_path}: {e}")
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    compressed_img = Image.open(buffer).convert('RGB')
    
    original = np.array(img, dtype=np.int16)
    compressed = np.array(compressed_img, dtype=np.int16)
    
    diff = np.abs(original - compressed)
    return float(diff.max())
