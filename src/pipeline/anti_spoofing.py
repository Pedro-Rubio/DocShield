import cv2
import numpy as np
from typing import Dict

def detect_moire(gray: np.ndarray) -> float:
    """
    Detección de screen replay attack mediante análisis de frecuencia (FFT).

    Args:
        gray: Imagen en escala de grises (numpy array).

    Returns:
        Ratio entre el pico periférico y la media periférica en el espectro FFT.
    """
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    mask = np.ones_like(magnitude, dtype=bool)
    mask[center_h-1:center_h+2, center_w-1:center_w+2] = False
    
    peripheral = magnitude[mask]
    if len(peripheral) == 0:
        return 0.0
    
    peak = peripheral.max()
    mean = peripheral.mean()
    return float(peak / mean) if mean != 0 else 0.0

def analyze_dct_blocks(gray: np.ndarray) -> float:
    """
    Análisis de bloques DCT 8x8 para detectar inconsistencias de edición.

    Args:
        gray: Imagen en escala de grises (numpy array).

    Returns:
        Relación entre la desviación estándar y la media de las energías AC de los bloques.
    """
    h, w = gray.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    gray_padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='constant') if pad_h > 0 or pad_w > 0 else gray
    
    h_pad, w_pad = gray_padded.shape
    block_energies = []
    
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            block = gray_padded[i:i+8, j:j+8].astype(np.float32)
            dct_block = cv2.dct(block)
            ac_coeffs = [dct_block[m, n] for m in range(8) for n in range(8) if not (m == 0 and n == 0)]
            energy = np.sum(np.array(ac_coeffs) ** 2)
            block_energies.append(energy)
    
    if not block_energies:
        return 0.0
    
    block_energies = np.array(block_energies)
    mean_energy = block_energies.mean()
    std_energy = block_energies.std()
    return float(std_energy / mean_energy) if mean_energy != 0 else 0.0

def analyze_reflection(bgr: np.ndarray) -> float:
    """
    Análisis de reflexión especular para distinguir pantallas de documentos físicos.

    Args:
        bgr: Imagen en formato BGR (numpy array).

    Returns:
        Gradiente promedio en zonas de alta reflectividad (>240 de gris).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 240
    if not np.any(mask):
        return 0.0
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    return float(grad_mag[mask].mean())
