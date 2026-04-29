"""
Módulo de detección anti-spoofing para documentos de identidad.

Implementa técnicas forenses para detectar:
- Screen replay attacks (patrón de Moiré via FFT)
- Ediciones en zonas del documento (análisis DCT 8x8)
- Reflexión especular (pantalla vs documento físico)
"""

import numpy as np
import cv2
from scipy import fft


def detect_moire(gray: np.ndarray) -> float:
    """
    Detección de screen replay attack mediante análisis de patrón de Moiré.

    Aplica FFT 2D a la imagen en escala de grises, enmascara la componente
    DC (centro), y calcula el ratio entre la energía del pico periférico
    y la energía media periférica.

    Args:
        gray: Imagen en escala de grises (numpy array).

    Returns:
        Score de Moiré. Valores > 8.5 indican posible captura de pantalla.
    """
    h, w = gray.shape

    # Aplicar FFT 2D
    f_transform = fft.fft2(gray.astype(np.float64))
    f_shift = np.abs(fft.fftshift(f_transform))

    # Crear máscara para eliminar componente DC (centro)
    center_y, center_x = h // 2, w // 2
    radius = max(h, w) // 20  # Radio del círculo DC

    mask = np.ones((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 0, -1)

    # Aplicar máscara
    f_shift_masked = f_shift * mask

    # Calcular energía periférica
    peripheral = f_shift_masked[mask > 0]
    if peripheral.size == 0:
        return 0.0

    media_periferica = np.mean(peripheral)
    pico_periferico = np.max(peripheral)

    if media_periferica == 0:
        return 0.0

    ratio = pico_periferico / media_periferica
    return float(ratio)


def analyze_dct_blocks(gray: np.ndarray) -> float:
    """
    Análisis de bloques DCT 8x8 para detectar zonas editadas.

    Calcula la energía AC de cada bloque 8x8 y evalúa la
    consistencia de la distribución de energías. Inconsistencias
    revelan zonas editadas con diferente historia de compresión.

    Args:
        gray: Imagen en escala de grises (numpy array).

    Returns:
        Score de anomalía DCT. Valores altos indican inconsistencias.
    """
    h, w = gray.shape

    # Recortar a múltiplos de 8
    h_cut = (h // 8) * 8
    w_cut = (w // 8) * 8
    cropped = gray[:h_cut, :w_cut]

    ac_energies = []

    for y in range(0, h_cut, 8):
        for x in range(0, w_cut, 8):
            block = cropped[y : y + 8, x : x + 8].astype(np.float64)
            # DCT 2D simple usando FFT
            dct_block = fft.dct(fft.dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
            # Energía AC (excluir el coeficiente DC en [0,0])
            ac_energy = np.sum(dct_block[1:, 1:] ** 2)
            ac_energies.append(ac_energy)

    if len(ac_energies) == 0:
        return 0.0

    energies = np.array(ac_energies)
    mean_energy = np.mean(energies)

    if mean_energy == 0:
        return 0.0

    # Coeficiente de variación (std/mean) como medida de inconsistencia
    std_energy = np.std(energies)
    cv_val = std_energy / mean_energy

    return float(cv_val)


def analyze_reflection(bgr: np.ndarray) -> float:
    """
    Análisis de reflexión especular para distinguir pantalla de documento físico.

    Detecta zonas de alta reflectividad (> 240 en escala de grises) y calcula
    el gradiente promedio en esas zonas. Las pantallas suelen tener gradiente
    alto mientras que un documento físico tiene gradiente suave.

    Args:
        bgr: Imagen en color (numpy array, formato BGR).

    Returns:
        Score de reflexión. Valores altos sugieren captura de pantalla.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detectar zonas de alta reflectividad
    bright_mask = gray > 240
    bright_pixels = np.count_nonzero(bright_mask)

    if bright_pixels == 0:
        return 0.0

    # Calcular gradientes
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Gradiente promedio en zonas brillantes
    avg_gradient = float(np.mean(grad_magnitude[bright_mask]))

    return avg_gradient
